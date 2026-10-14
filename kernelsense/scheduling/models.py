"""Data models for CPU scheduling simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

IDLE = "IDLE"


@dataclass(frozen=True)
class Job:
    """A process as seen by the CPU scheduler.

    ``priority``: lower number = higher priority (Unix convention).
    """

    pid: str
    arrival: int
    burst: int
    priority: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.burst <= 0:
            raise ValueError(f"{self.pid}: burst must be positive")
        if self.arrival < 0:
            raise ValueError(f"{self.pid}: arrival must be non-negative")


@dataclass
class Segment:
    pid: str
    start: int
    end: int
    queue: int | None = None  # MLFQ level, when relevant

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ProcessStats:
    pid: str
    arrival: int
    burst: int
    priority: int
    start: int
    completion: int

    @property
    def turnaround(self) -> int:
        return self.completion - self.arrival

    @property
    def waiting(self) -> int:
        return self.turnaround - self.burst

    @property
    def response(self) -> int:
        return self.start - self.arrival


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


@dataclass
class ScheduleResult:
    algorithm: str
    gantt: list[Segment]
    stats: list[ProcessStats]
    context_switches: int
    params: dict = field(default_factory=dict)

    @property
    def makespan(self) -> int:
        return max((s.end for s in self.gantt), default=0)

    @property
    def busy_time(self) -> int:
        return sum(s.length for s in self.gantt if s.pid != IDLE)

    def summary(self) -> dict:
        n = len(self.stats) or 1
        tat = [s.turnaround for s in self.stats]
        wt = [s.waiting for s in self.stats]
        rt = [s.response for s in self.stats]
        span = self.makespan or 1
        start_time = min((s.arrival for s in self.stats), default=0)
        effective_span = max(span - start_time, 1)
        # Jain's fairness index on slowdown (turnaround / burst): 1.0 = perfectly fair
        slowdown = [s.turnaround / s.burst for s in self.stats] or [1.0]
        jain = (sum(slowdown) ** 2) / (len(slowdown) * sum(x * x for x in slowdown))
        return {
            "algorithm": self.algorithm,
            "avg_waiting": round(sum(wt) / n, 2),
            "avg_turnaround": round(sum(tat) / n, 2),
            "avg_response": round(sum(rt) / n, 2),
            "max_waiting": max(wt, default=0),
            "p95_response": round(_percentile(rt, 95), 2),
            "p99_turnaround": round(_percentile(tat, 99), 2),
            "throughput": round(len(self.stats) / effective_span, 4),
            "cpu_utilization": round(100 * self.busy_time / effective_span, 2),
            "context_switches": self.context_switches,
            "fairness": round(jain, 4),
            "makespan": span,
        }

    def stats_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pid": s.pid, "arrival": s.arrival, "burst": s.burst, "priority": s.priority,
                    "start": s.start, "completion": s.completion, "turnaround": s.turnaround,
                    "waiting": s.waiting, "response": s.response,
                }
                for s in sorted(self.stats, key=lambda x: (x.arrival, x.pid))
            ]
        )

    def gantt_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"pid": g.pid, "start": g.start, "end": g.end, "length": g.length, "queue": g.queue} for g in self.gantt]
        )
