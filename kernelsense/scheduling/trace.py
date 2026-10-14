"""CPU-burst trace capture from the live kernel and trace-driven replay.

A process alternates between CPU bursts (running) and waits (blocked or
preempted). We sample each process's cumulative CPU time
(``utime + stime`` jiffies in /proc/[pid]/stat) at a fixed interval: a sample
in which the counter advanced means the process was on a CPU. Consecutive
running samples form one burst; a gap ends it.

The trace is turned into a scheduling workload in which every observed burst
is a job arriving at the moment it was observed, so the simulators replay what
the machine actually did instead of hand-made numbers.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..monitoring import procfs
from .models import Job


@dataclass
class Burst:
    pid: int
    name: str
    start_ms: float
    length_ms: float
    cpu_ms: float  # CPU time actually consumed during the burst (from jiffies)


@dataclass
class BurstTrace:
    interval_ms: float
    duration_s: float
    bursts: list[Burst] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(b) for b in self.bursts])

    def sequences(self, min_len: int = 3) -> dict[int, list[float]]:
        """Per-process sequence of burst lengths (ms) for burst prediction."""
        seq: dict[int, list[float]] = {}
        for b in sorted(self.bursts, key=lambda x: x.start_ms):
            seq.setdefault(b.pid, []).append(b.length_ms)
        return {p: s for p, s in seq.items() if len(s) >= min_len}

    def to_jobs(self, unit_ms: float | None = None, max_jobs: int = 60) -> list[Job]:
        """Each burst becomes a job (arrival = observed start, burst = observed length)."""
        unit = unit_ms or self.interval_ms
        ordered = sorted(self.bursts, key=lambda b: b.start_ms)[:max_jobs]
        jobs, counts = [], {}
        t0 = ordered[0].start_ms if ordered else 0.0
        for b in ordered:
            counts[b.pid] = counts.get(b.pid, 0) + 1
            jobs.append(Job(
                pid=f"{b.name[:10]}:{b.pid}#{counts[b.pid]}",
                arrival=int(round((b.start_ms - t0) / unit)),
                burst=max(1, int(round(b.length_ms / unit))),
                priority=0,
                label=b.name,
            ))
        return jobs

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"interval_ms": self.interval_ms, "duration_s": self.duration_s,
                                    "captured_at": self.captured_at, "bursts": [asdict(b) for b in self.bursts]}))
        return path

    @staticmethod
    def load(path: str | Path) -> "BurstTrace":
        d = json.loads(Path(path).read_text())
        return BurstTrace(d["interval_ms"], d["duration_s"], [Burst(**b) for b in d["bursts"]], d.get("captured_at", 0))


def _cpu_ticks_all() -> dict[int, tuple[str, int]]:
    out = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            st = procfs.read_stat(int(entry))
            if st:
                out[int(entry)] = (st["comm"], st["utime_ticks"] + st["stime_ticks"])
    return out


def capture(duration_s: float = 5.0, interval_ms: float = 20.0, pids: list[int] | None = None,
            min_cpu_ticks: int = 1) -> BurstTrace:
    """Sample CPU-time counters and segment them into bursts.

    Resolution is bounded by the kernel's accounting granularity (1 jiffy =
    1000/CLK_TCK ms, usually 10 ms), so ``interval_ms`` below that adds no detail.
    """
    if not procfs.is_linux_procfs():
        raise RuntimeError("burst capture needs Linux /proc")
    hz = procfs.clock_ticks()
    interval = max(interval_ms, 1000 / hz) / 1000
    open_bursts: dict[int, list] = {}  # pid -> [start_ms, last_ms, ticks_at_start, ticks_last, name]
    prev = _cpu_ticks_all() if pids is None else {p: v for p, v in _cpu_ticks_all().items() if p in pids}
    t0 = time.perf_counter()
    bursts: list[Burst] = []
    while (now := time.perf_counter()) - t0 < duration_s:
        time.sleep(interval)
        now_ms = (time.perf_counter() - t0) * 1000
        cur = _cpu_ticks_all() if pids is None else {p: v for p, v in _cpu_ticks_all().items() if p in pids}
        for pid, (name, ticks) in cur.items():
            last = prev.get(pid, (name, ticks))[1]
            running = ticks - last >= min_cpu_ticks
            if running:
                if pid in open_bursts:
                    open_bursts[pid][1] = now_ms
                    open_bursts[pid][3] = ticks
                else:
                    open_bursts[pid] = [now_ms - interval * 1000, now_ms, last, ticks, name]
            elif pid in open_bursts:
                s, e, tk0, tk1, nm = open_bursts.pop(pid)
                bursts.append(Burst(pid, nm, round(s, 1), round(e - s, 1), round((tk1 - tk0) * 1000 / hz, 1)))
        prev = cur
    end_ms = (time.perf_counter() - t0) * 1000
    for pid, (s, e, tk0, tk1, nm) in open_bursts.items():
        bursts.append(Burst(pid, nm, round(s, 1), round(max(e, end_ms) - s, 1), round((tk1 - tk0) * 1000 / hz, 1)))
    return BurstTrace(interval * 1000, duration_s, sorted(bursts, key=lambda b: b.start_ms))
