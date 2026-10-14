"""Memory leak detection from per-process RSS history.

A leaking process shows resident memory that keeps growing steadily instead of
plateauing. We fit a linear trend to RSS over time and flag a process when:

* the slope is above ``min_slope_mb_per_min``
* the trend explains most of the variation (R^2 >= ``min_r2``)
* most consecutive samples are non-decreasing (``min_monotonic``)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LeakFinding:
    pid: int
    name: str
    samples: int
    start_mb: float
    end_mb: float
    slope_mb_per_min: float
    r2: float
    monotonic_ratio: float
    suspected: bool
    projected_mb_1h: float


def analyze_series(ts: np.ndarray, rss_mb: np.ndarray) -> tuple[float, float, float]:
    """Return (slope MB/min, R^2, fraction of non-decreasing steps)."""
    t_min = (ts - ts[0]) / 60.0
    if np.ptp(t_min) == 0:
        return 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(t_min, rss_mb, 1)
    fitted = slope * t_min + intercept
    ss_res = float(np.sum((rss_mb - fitted) ** 2))
    ss_tot = float(np.sum((rss_mb - rss_mb.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    diffs = np.diff(rss_mb)
    mono = float(np.mean(diffs >= -1e-6)) if len(diffs) else 0.0
    return float(slope), float(r2), mono


def detect_leaks(
    history: pd.DataFrame,
    min_samples: int = 10,
    min_slope_mb_per_min: float = 1.0,
    min_r2: float = 0.8,
    min_monotonic: float = 0.7,
) -> list[LeakFinding]:
    """``history`` has columns ts, pid, name, rss_mb (as stored by MetricStore)."""
    if history.empty:
        return []
    findings = []
    for pid, grp in history.groupby("pid"):
        grp = grp.sort_values("ts")
        if len(grp) < min_samples:
            continue
        ts = grp["ts"].to_numpy(float)
        rss = grp["rss_mb"].to_numpy(float)
        slope, r2, mono = analyze_series(ts, rss)
        suspected = slope >= min_slope_mb_per_min and r2 >= min_r2 and mono >= min_monotonic
        findings.append(
            LeakFinding(
                pid=int(pid),
                name=str(grp["name"].iloc[-1]),
                samples=len(grp),
                start_mb=round(float(rss[0]), 2),
                end_mb=round(float(rss[-1]), 2),
                slope_mb_per_min=round(slope, 3),
                r2=round(r2, 3),
                monotonic_ratio=round(mono, 3),
                suspected=suspected,
                projected_mb_1h=round(float(rss[-1] + slope * 60), 1),
            )
        )
    findings.sort(key=lambda f: (not f.suspected, -f.slope_mb_per_min))
    return findings
