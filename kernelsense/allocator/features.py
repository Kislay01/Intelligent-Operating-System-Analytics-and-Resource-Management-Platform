"""Feature extraction for workload classification.

Features describe *how a process uses the OS*, not what it is called:

    cpu_mean, cpu_std      CPU demand and burstiness
    rss_mb, rss_slope      memory footprint and growth (MB/min)
    vol_rate, invol_rate   voluntary / involuntary context switches per second
    vol_ratio              share of switches that were voluntary (blocking)
    io_rate_kb             disk I/O throughput (KB/s)
    threads                number of threads
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["cpu_mean", "cpu_std", "rss_mb", "rss_slope", "vol_rate", "invol_rate", "vol_ratio", "io_rate_kb", "threads"]


def from_history(history: pd.DataFrame, min_samples: int = 3) -> pd.DataFrame:
    """Aggregate MetricStore.process_history() rows into one feature row per PID."""
    if history.empty:
        return pd.DataFrame(columns=["pid", "name", *FEATURES])
    rows = []
    for pid, g in history.sort_values("ts").groupby("pid"):
        if len(g) < min_samples:
            continue
        dt = g["ts"].iloc[-1] - g["ts"].iloc[0]
        if dt <= 0:
            continue
        dvol = max(g["vol_ctx"].iloc[-1] - g["vol_ctx"].iloc[0], 0)
        dinv = max(g["invol_ctx"].iloc[-1] - g["invol_ctx"].iloc[0], 0)
        io = (g["read_bytes"].fillna(0) + g["write_bytes"].fillna(0))
        dio = max(io.iloc[-1] - io.iloc[0], 0)
        t_min = (g["ts"] - g["ts"].iloc[0]) / 60
        slope = float(np.polyfit(t_min, g["rss_mb"], 1)[0]) if t_min.nunique() > 1 else 0.0
        rows.append({
            "pid": int(pid),
            "name": g["name"].iloc[-1],
            "cpu_mean": g["cpu_percent"].mean(),
            "cpu_std": g["cpu_percent"].std(ddof=0),
            "rss_mb": g["rss_mb"].iloc[-1],
            "rss_slope": slope,
            "vol_rate": dvol / dt,
            "invol_rate": dinv / dt,
            "vol_ratio": dvol / (dvol + dinv) if (dvol + dinv) else 0.0,
            "io_rate_kb": dio / dt / 1024,
            "threads": g["num_threads"].iloc[-1],
        })
    return pd.DataFrame(rows)


def from_snapshot(procs) -> pd.DataFrame:
    """Single-snapshot features from ProcessInfo objects (rates come from the collector)."""
    rows = []
    for p in procs:
        total = p.vol_ctx_rate + p.invol_ctx_rate
        rows.append({
            "pid": p.pid, "name": p.name,
            "cpu_mean": p.cpu_percent, "cpu_std": 0.0,
            "rss_mb": p.rss_mb, "rss_slope": 0.0,
            "vol_rate": p.vol_ctx_rate, "invol_rate": p.invol_ctx_rate,
            "vol_ratio": p.vol_ctx_rate / total if total else 0.0,
            "io_rate_kb": p.io_rate / 1024, "threads": p.num_threads,
        })
    return pd.DataFrame(rows, columns=["pid", "name", *FEATURES])
