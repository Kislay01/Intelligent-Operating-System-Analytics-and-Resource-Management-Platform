"""CPU contention environment for learning allocation policies.

A tick-based CFS model of one CPU shared by
* one latency-sensitive *interactive* task: wakes every ``period`` ticks and
  needs ``burst`` ticks of CPU (think: a UI thread or a request handler), and
* ``n_batch`` CPU-bound *batch* tasks that are always runnable, and
* ``n_other`` other CPU-bound tasks at nice 0 (services the policy must not
  touch) that compete with the batch work.

The agent chooses the nice value applied to the batch tasks. CFS picks the task
with the smallest vruntime; vruntime grows by NICE_0_WEIGHT / weight, so a
higher batch nice lets the interactive task run sooner after it wakes.

Reward trades a latency SLO for the interactive task against batch throughput:
    reward = batch_share / fair_share  -  lam * max(0, latency_p95 - slo)  -  0.02 * latency_mean
where fair_share is the batch share at nice 0. Lowering batch priority is only
worth it when the interactive task actually misses its SLO, so the best action
depends on the workload - which is what the learner has to discover.
The same simulator is used to train and evaluate policies (bandit, Q-learning)
before they act on the real system through nice values or cgroup cpu.weight.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..scheduling.proportional import NICE_0_WEIGHT, nice_weight

ACTIONS = [0, 5, 10, 15, 19]  # batch nice levels


@dataclass
class Scenario:
    n_batch: int
    burst: int
    period: int
    n_other: int = 0
    slo: int = 1  # ticks (ms) the interactive task may wait after waking

    @property
    def interactive_load(self) -> float:
        return self.burst / self.period


@dataclass
class StepResult:
    latency_p95: float
    latency_mean: float
    batch_share: float
    reward: float


SCHED_LATENCY = 6      # ticks in which every runnable task should run once
MIN_GRANULARITY = 1    # minimum slice
SLEEPER_CREDIT = 3     # vruntime credit given to a task that slept (capped, like CFS)
# Wakeup preemption: a woken nice-0 task preempts the running task with probability
# (w_woken / w_running - 1) / (PREEMPT_RATIO - 1), clipped to [0, 1]. Below that it waits
# for the current slice to end. PREEMPT_RATIO is calibrated against measured wake-up
# latency on the real machine (see calibrate()); the default 4 was fitted on a 2-CPU Linux 6.x
# VM where nice-0 and nice-5 batch work delayed a waking task by milliseconds and nice >= 10 did not.
PREEMPT_RATIO = 4.0


def _run(sc: Scenario, batch_nice: int, horizon: int, seed: int):
    """Tick-based CFS with time slices, sleeper credit and wakeup-preemption granularity."""
    rng = random.Random(seed)
    tasks: dict = {}  # key -> [vruntime, weight]
    for k in range(sc.n_batch):
        tasks[("b", k)] = [0.0, nice_weight(batch_nice)]
    for k in range(sc.n_other):
        tasks[("o", k)] = [0.0, nice_weight(0)]
    w_int = nice_weight(0)
    vr_int = 0.0
    remaining = 0
    woke_at = None
    started = False
    latencies = []
    batch_ticks = 0
    running = None
    slice_left = 0
    next_wake = rng.randrange(sc.period)
    for t in range(horizon):
        woke_now = False
        if t >= next_wake and remaining == 0:
            remaining, woke_at, started, woke_now = sc.burst, t, False, True
            pool = [v[0] for v in tasks.values()]
            vr_int = max(vr_int, min(pool) - SLEEPER_CREDIT) if pool else vr_int
            next_wake = t + sc.period + rng.randint(-1, 1)
        runnable = dict(tasks)
        if remaining > 0:
            runnable["i"] = [vr_int, w_int]
        if not runnable:
            continue
        total_w = sum(w for _, w in runnable.values())
        need_pick = running not in runnable or slice_left <= 0
        if not need_pick and woke_now and running != "i":
            ratio = w_int / runnable[running][1]
            p_preempt = min(1.0, max(0.0, (ratio - 1) / (PREEMPT_RATIO - 1)))
            if rng.random() < p_preempt:
                need_pick = True
        if need_pick:
            running = min(runnable, key=lambda k: runnable[k][0])
            slice_left = max(MIN_GRANULARITY, round(SCHED_LATENCY * runnable[running][1] / total_w))
        if running == "i":
            if not started:
                latencies.append(t - woke_at)
                started = True
            remaining -= 1
            vr_int += NICE_0_WEIGHT / w_int
            if remaining == 0:
                running, slice_left = None, 0
                continue
        else:
            tasks[running][0] += NICE_0_WEIGHT / tasks[running][1]
            if running[0] == "b":
                batch_ticks += 1
        slice_left -= 1
    lat = np.array(latencies or [0])
    return float(np.percentile(lat, 95)), float(lat.mean()), batch_ticks / horizon


_FAIR_CACHE: dict = {}


def simulate(sc: Scenario, batch_nice: int, horizon: int = 2000, lam: float = 0.6, seed: int = 0) -> StepResult:
    p95, mean, share = _run(sc, batch_nice, horizon, seed)
    key = (sc.n_batch, sc.burst, sc.period, sc.n_other, horizon, seed)
    if key not in _FAIR_CACHE:
        _FAIR_CACHE[key] = _run(sc, 0, horizon, seed)[2] or 1e-9
    fair = _FAIR_CACHE[key]
    # small mean-latency term breaks ties in favour of responsiveness when throughput is equal
    reward = share / fair - lam * max(0.0, p95 - sc.slo) - 0.02 * mean
    return StepResult(p95, mean, share, reward)


def random_scenario(rng: random.Random) -> Scenario:
    return Scenario(n_batch=rng.randint(0, 5), burst=rng.randint(1, 4), period=rng.randint(6, 24),
                    n_other=rng.randint(0, 3))


def state_of(sc: Scenario) -> tuple[int, int, int]:
    """Discretised state: (runnable CPU hogs bucket, interactive load bucket, batch share of hogs bucket)."""
    hogs = sc.n_batch + sc.n_other
    hb = 0 if hogs == 0 else 1 if hogs <= 2 else 2 if hogs <= 4 else 3
    load = sc.interactive_load
    lb = 0 if load < 0.1 else 1 if load < 0.25 else 2
    frac = sc.n_batch / hogs if hogs else 0.0
    fb = 0 if frac == 0 else 1 if frac < 0.6 else 2
    return hb, lb, fb


def context_of(sc: Scenario) -> np.ndarray:
    hogs = sc.n_batch + sc.n_other
    return np.array([1.0, sc.n_batch / 5, sc.n_other / 3, sc.interactive_load * 4, sc.burst / 4,
                     (sc.n_batch / hogs) if hogs else 0.0])


def load_calibration(path=None) -> None:
    """Apply a saved PREEMPT_RATIO calibration if one exists."""
    import json
    from pathlib import Path

    from ..config import settings

    global PREEMPT_RATIO
    p = Path(path) if path else settings.data_dir / "sim_calibration.json"
    try:
        PREEMPT_RATIO = float(json.loads(p.read_text())["preempt_ratio"])
        _FAIR_CACHE.clear()
    except (OSError, ValueError, KeyError):
        pass


def calibrate(measured_p95_ms: dict[int, float], n_batch: int, burst: int = 1, period: int = 5,
              grid=(1.5, 2, 3, 4, 6, 9.3, 15, 30, 70)) -> dict:
    """Fit PREEMPT_RATIO so simulated p95 wake-up latency (1 tick = 1 ms) matches measurements.

    ``measured_p95_ms``: {batch nice: p95 latency in ms} from the latency probe on the real machine.
    """
    global PREEMPT_RATIO
    best = None
    for r in grid:
        PREEMPT_RATIO = r
        _FAIR_CACHE.clear()
        err = 0.0
        for nice, real in measured_p95_ms.items():
            sim = _run(Scenario(n_batch, burst, period), nice, 3000, 0)[0]
            err += (sim - real) ** 2
        if best is None or err < best[1]:
            best = (r, err)
    PREEMPT_RATIO = best[0]
    _FAIR_CACHE.clear()
    return {"preempt_ratio": best[0], "sse": round(best[1], 3)}
