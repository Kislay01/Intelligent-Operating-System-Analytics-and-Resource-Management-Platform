"""Closed-loop resource controller: observe -> decide -> act -> measure.

Every ``interval`` seconds the controller
1. observes the managed batch processes, other CPU hogs and the interactive
   processes through /proc (CPU demand, runnable hogs),
2. maps that to the learned policy's state and chooses a batch nice level,
3. applies it through nice values or a cgroup (cpu.weight equivalent of the
   nice level),
4. measures CPU pressure (PSI) and rolls the action back if pressure got worse
   by more than ``psi_margin`` (safety guard), and logs every decision.

Policies: "none" (observe only), "rule", "q" (Q-learning table trained in the
simulator) or "bandit".
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from ..config import settings
from ..monitoring import procfs
from ..scheduling.proportional import NICE_0_WEIGHT, nice_weight
from .cgroup import Cgroup, detect
from .env import ACTIONS, Scenario, context_of, load_calibration, state_of
from .learning import LinUCB, QLearner, rule_policy, train


@dataclass
class Decision:
    ts: float
    state: tuple
    nice: int
    applied: bool
    psi_before: float | None
    psi_after: float | None
    rolled_back: bool = False
    note: str = ""


@dataclass
class ControllerConfig:
    policy: str = "q"  # none | rule | q | bandit
    mode: str = "nice"  # nice | cgroup
    interval: float = 2.0
    psi_margin: float = 15.0  # percentage points of PSI 'some' avg10
    q_path: Path = field(default_factory=lambda: settings.data_dir / "q_policy.json")


def _cpu_pct(pids: list[int], window: float = 0.3) -> dict[int, float]:
    procs = {}
    for p in pids:
        try:
            procs[p] = psutil.Process(p)
            procs[p].cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(window)
    out = {}
    for p, pr in procs.items():
        try:
            out[p] = pr.cpu_percent(None)
        except psutil.Error:
            pass
    return out


def _psi_cpu() -> float | None:
    p = procfs.read_pressure("cpu")
    return p.get("some", {}).get("avg10") if p else None


class ClosedLoopController:
    def __init__(self, batch_pids: list[int], interactive_pids: list[int] | None = None,
                 cfg: ControllerConfig | None = None, other_hog_threshold: float = 50.0) -> None:
        self.cfg = cfg or ControllerConfig()
        self.batch = list(batch_pids)
        self.interactive = list(interactive_pids or [])
        self.other_hog_threshold = other_hog_threshold
        self.decisions: list[Decision] = []
        self.current_nice = 0
        self.is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        self.cgroup: Cgroup | None = None
        self.qlearner: QLearner | None = None
        self.bandit: LinUCB | None = None
        if self.cfg.policy in ("q", "bandit"):
            load_calibration()
            self._load_policy()
        if self.cfg.mode == "cgroup":
            info = detect()
            if info is None or not info.writable:
                raise RuntimeError("cgroup mode needs a writable cgroup hierarchy (run as root or delegate)")
            self.cgroup = Cgroup(f"batch-{os.getpid()}", info).create()
            for p in self.batch:
                try:
                    self.cgroup.add(p)
                except OSError:
                    pass

    def _load_policy(self) -> None:
        try:
            self.qlearner = QLearner.load(self.cfg.q_path)
        except (OSError, ValueError, KeyError):
            bandit, q, _ = train()
            self.qlearner, self.bandit = q, bandit
            try:
                q.save(self.cfg.q_path)
            except OSError:
                pass
        if self.cfg.policy == "bandit" and self.bandit is None:
            self.bandit, _, _ = train()

    # -------------------------------------------------------------- observe
    def observe(self) -> Scenario:
        live_batch = [p for p in self.batch if psutil.pid_exists(p)]
        others = []
        for proc in psutil.process_iter(["pid"]):
            if proc.pid not in live_batch and proc.pid not in self.interactive and proc.pid != os.getpid():
                others.append(proc.pid)
        cpu = _cpu_pct(live_batch + self.interactive + others)
        n_other = sum(1 for p in others if cpu.get(p, 0) >= self.other_hog_threshold)
        inter_cpu = sum(cpu.get(p, 0) for p in self.interactive) / 100
        load = min(max(inter_cpu, 0.05), 0.5)
        period = 10
        burst = max(1, round(load * period))
        # the simulator models one CPU: express contention per CPU
        ncpu = os.cpu_count() or 1
        per_cpu = lambda n: 0 if n == 0 else max(1, round(n / ncpu))  # noqa: E731
        return Scenario(n_batch=per_cpu(len(live_batch)), burst=burst, period=period, n_other=per_cpu(n_other))

    # --------------------------------------------------------------- decide
    def decide(self, sc: Scenario) -> int:
        if self.cfg.policy == "none":
            return self.current_nice
        if self.cfg.policy == "rule":
            return rule_policy(sc)
        if self.cfg.policy == "bandit" and self.bandit is not None:
            return ACTIONS[self.bandit.choose(context_of(sc), explore=False)]
        return ACTIONS[self.qlearner.choose(state_of(sc), explore=False)]

    # ------------------------------------------------------------------ act
    def apply(self, nice: int) -> bool:
        if self.cfg.mode == "cgroup" and self.cgroup is not None:
            self.cgroup.set_cpu_weight(round(100 * nice_weight(nice) / NICE_0_WEIGHT))
            return True
        ok = True
        for p in self.batch:
            try:
                proc = psutil.Process(p)
                if nice < proc.nice() and not self.is_root:
                    ok = False  # lowering nice needs CAP_SYS_NICE
                    continue
                proc.nice(nice)
            except psutil.Error:
                ok = False
        return ok

    # ----------------------------------------------------------------- loop
    def step(self) -> Decision:
        sc = self.observe()
        nice = self.decide(sc)
        before = _psi_cpu()
        applied = False
        if nice != self.current_nice:
            applied = self.apply(nice)
        prev = self.current_nice
        self.current_nice = nice if applied or nice == self.current_nice else self.current_nice
        time.sleep(self.cfg.interval)
        after = _psi_cpu()
        d = Decision(time.time(), state_of(sc), nice, applied, before, after)
        if applied and before is not None and after is not None and after > before + self.cfg.psi_margin:
            self.apply(prev)
            self.current_nice = prev
            d.rolled_back = True
            d.note = f"CPU pressure rose {before:.1f} -> {after:.1f}; rolled back to nice {prev}"
        self.decisions.append(d)
        return d

    def run(self, duration: float) -> list[Decision]:
        end = time.time() + duration
        while time.time() < end:
            self.step()
        return self.decisions

    def close(self, restore: bool = True) -> None:
        if restore and self.current_nice != 0:
            self.apply(0)
        if self.cgroup is not None:
            self.cgroup.remove()
            self.cgroup = None
