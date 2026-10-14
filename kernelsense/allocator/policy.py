"""Allocation policy: workload class + system state -> concrete OS actions.

Linux levers used (all available to unprivileged users for their own
processes, except lowering nice below its current value):

* nice value (-20..19)  - CFS weight; each step changes CPU share by ~10%
* CPU affinity          - pin processes to cores (sched_setaffinity)
* I/O priority (ionice) - best-effort level 0..7 or the idle class
* scheduling class      - SCHED_BATCH for throughput-oriented background work
                          (suggested only; changing it is left to Phase B)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psutil

SYSTEM_PIDS = {0, 1, 2}
PROTECTED_NAMES = {"systemd", "init", "kthreadd", "sshd", "Xorg", "gnome-shell", "dbus-daemon", "streamlit"}


@dataclass
class Action:
    pid: int
    name: str
    kind: str  # nice | affinity | ionice | sched_class | memory_watch
    current: object
    target: object
    reason: str
    applicable: bool = True  # False = advisory only (needs root or Phase B)

    def describe(self) -> str:
        return f"{self.kind}: {self.current} -> {self.target}"


@dataclass
class SystemState:
    cpu_percent: float
    mem_percent: float
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count(logical=True) or 1)


@dataclass
class PolicyConfig:
    batch_nice: int = 10
    io_bound_ionice_level: int = 7
    cpu_busy_pct: float = 70.0
    mem_busy_pct: float = 85.0
    pin_batch_to_last_cores: bool = True
    min_confidence: float = 0.6


def plan(classified, system: SystemState, cfg: PolicyConfig | None = None, current: dict | None = None) -> list[Action]:
    """Build actions for classified processes.

    ``classified``: DataFrame with pid, name, predicted, confidence, cpu_mean, rss_mb.
    ``current``: optional {pid: {"nice": int, "affinity": list[int]}} of present settings.
    """
    cfg = cfg or PolicyConfig()
    current = current or {}
    actions: list[Action] = []
    cpu_busy = system.cpu_percent >= cfg.cpu_busy_pct
    mem_busy = system.mem_percent >= cfg.mem_busy_pct
    cores = list(range(system.cpu_count))
    batch_cores = cores[len(cores) // 2 :] if len(cores) > 1 else cores
    has_interactive = (classified["predicted"] == "Interactive").any() if not classified.empty else False

    for row in classified.itertuples(index=False):
        if row.pid in SYSTEM_PIDS or row.name in PROTECTED_NAMES or row.confidence < cfg.min_confidence:
            continue
        cur = current.get(row.pid, {})
        nice_now = cur.get("nice", 0)
        aff_now = cur.get("affinity", cores)

        if row.predicted == "CPU-bound":
            if (cpu_busy or has_interactive) and nice_now < cfg.batch_nice:
                actions.append(Action(row.pid, row.name, "nice", nice_now, cfg.batch_nice,
                                      "CPU-bound background work competes with interactive tasks; a higher nice "
                                      "value lowers its CFS weight so foreground tasks get the CPU first."))
            if cfg.pin_batch_to_last_cores and len(cores) > 1 and sorted(aff_now) != batch_cores and cpu_busy:
                actions.append(Action(row.pid, row.name, "affinity", aff_now, batch_cores,
                                      "Pin batch work to a subset of cores, leaving the rest free for "
                                      "latency-sensitive tasks and improving cache locality."))
            actions.append(Action(row.pid, row.name, "sched_class", "SCHED_OTHER", "SCHED_BATCH",
                                  "SCHED_BATCH tells the scheduler the task is non-interactive, reducing "
                                  "wakeup preemption.", applicable=False))
        elif row.predicted == "I/O-bound":
            actions.append(Action(row.pid, row.name, "ionice", "best-effort/4", f"best-effort/{cfg.io_bound_ionice_level}",
                                  "Heavy disk I/O; lowering its I/O priority keeps the disk responsive for "
                                  "other processes. Its CPU priority stays unchanged because it mostly sleeps."))
        elif row.predicted == "Interactive":
            if nice_now > 0:
                actions.append(Action(row.pid, row.name, "nice", nice_now, 0,
                                      "Interactive task should not be de-prioritised; restore default weight.",
                                      applicable=False))
        elif row.predicted == "Memory-heavy":
            actions.append(Action(row.pid, row.name, "memory_watch", f"{row.rss_mb:.0f} MB", "monitor / limit",
                                  "Large or growing resident set. Under memory pressure this process is the most "
                                  "likely OOM-killer target; a cgroup memory.high limit would throttle it "
                                  "before the system runs out of RAM.", applicable=False))
            if mem_busy and nice_now < 5:
                actions.append(Action(row.pid, row.name, "nice", nice_now, 5,
                                      "System memory is under pressure; slowing the memory-heavy process "
                                      "reduces its allocation rate.", ))
    return actions
