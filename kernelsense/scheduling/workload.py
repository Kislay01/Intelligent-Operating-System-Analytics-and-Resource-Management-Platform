"""Workload sources for the scheduling simulator: synthetic, CSV and live system."""

from __future__ import annotations

import io
import random
from typing import Iterable

import pandas as pd

from .models import Job

PROFILES = {
    "balanced": "Mix of short and long jobs arriving over time",
    "cpu_heavy": "Few long CPU-bound jobs (batch processing)",
    "interactive": "Many short jobs arriving frequently (desktop / server requests)",
    "convoy": "One very long job arrives first, followed by many short ones",
    "priority_mix": "Jobs with widely different priorities",
}


def synthetic(profile: str = "balanced", n: int = 8, seed: int | None = 42) -> list[Job]:
    rng = random.Random(seed)
    jobs: list[Job] = []
    if profile == "cpu_heavy":
        for i in range(n):
            jobs.append(Job(f"P{i+1}", rng.randint(0, 6), rng.randint(15, 40), rng.randint(1, 5), "CPU-bound"))
    elif profile == "interactive":
        t = 0
        for i in range(n):
            t += rng.randint(0, 3)
            jobs.append(Job(f"P{i+1}", t, rng.randint(1, 5), rng.randint(1, 5), "Interactive"))
    elif profile == "convoy":
        jobs.append(Job("P1", 0, 40, 3, "CPU-bound"))
        for i in range(1, n):
            jobs.append(Job(f"P{i+1}", rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 5), "Interactive"))
    elif profile == "priority_mix":
        t = 0
        for i in range(n):
            t += rng.randint(0, 4)
            jobs.append(Job(f"P{i+1}", t, rng.randint(2, 15), rng.choice([0, 1, 5, 10, 20]), "Mixed"))
    else:  # balanced
        t = 0
        for i in range(n):
            t += rng.randint(0, 5)
            long = rng.random() < 0.3
            burst = rng.randint(12, 25) if long else rng.randint(1, 8)
            jobs.append(Job(f"P{i+1}", t, burst, rng.randint(1, 5), "CPU-bound" if long else "Interactive"))
    return jobs


def from_frame(df: pd.DataFrame) -> list[Job]:
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["pid", "arrival", "burst"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected pid, arrival, burst[, priority]")
    jobs = []
    for _, row in df.iterrows():
        jobs.append(
            Job(
                pid=str(row[cols["pid"]]),
                arrival=int(row[cols["arrival"]]),
                burst=int(row[cols["burst"]]),
                priority=int(row[cols["priority"]]) if "priority" in cols and pd.notna(row[cols["priority"]]) else 0,
            )
        )
    return jobs


def from_csv(text_or_path) -> list[Job]:
    if isinstance(text_or_path, str) and "\n" in text_or_path:
        return from_frame(pd.read_csv(io.StringIO(text_or_path)))
    return from_frame(pd.read_csv(text_or_path))


def to_frame(jobs: Iterable[Job]) -> pd.DataFrame:
    return pd.DataFrame([{"pid": j.pid, "arrival": j.arrival, "burst": j.burst, "priority": j.priority} for j in jobs])


def from_live_processes(procs, limit: int = 10, scale: float = 1.0) -> list[Job]:
    """Turn a live process snapshot into a scheduling workload.

    The OS does not expose future bursts, so we derive an *estimate*:
      * burst  ~ the process's current CPU demand (CPU% * scale), min 1
      * arrival staggered by process age (older processes arrive first)
      * priority = nice value shifted to be non-negative (nice -20..19 -> 0..39)
    The goal is a realistic mix of CPU-bound and interactive jobs derived from
    what is actually running, not an exact replay.
    """
    active = sorted(
        [p for p in procs if p.cpu_percent > 0 or getattr(p, "vol_ctx_rate", 0) > 0],
        key=lambda p: p.cpu_percent,
        reverse=True,
    )[:limit]
    if not active:
        return []
    oldest = min(p.create_time for p in active)
    newest = max(p.create_time for p in active)
    spread = max(newest - oldest, 1.0)
    jobs = []
    for p in active:
        arrival = int(round(10 * (p.create_time - oldest) / spread))
        burst = max(1, int(round(p.cpu_percent * scale / 5)) or 1)
        if getattr(p, "workload_class", "") == "CPU-bound":
            burst = max(burst, 10)
        jobs.append(Job(f"{p.name[:12]}:{p.pid}", arrival, burst, p.nice + 20, getattr(p, "workload_class", "")))
    return jobs
