"""Memory leak detection from per-process RSS history.

A leaking process shows resident memory that keeps growing steadily instead of
plateauing. We fit a linear trend to RSS over time and flag a process when:

* the slope is above ``min_slope_mb_per_min``
* the trend explains most of the variation (R^2 >= ``min_r2``)
* most consecutive samples are non-decreasing (``min_monotonic``)
* growth is sustained (positive slope in both halves of the window, which
  rejects one-off step changes)
* the Mann-Kendall test confirms a statistically significant upward trend
  (non-parametric, so it is robust to garbage-collection sawtooth patterns
  and outliers that break the linear fit)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mann_kendall(x: np.ndarray) -> tuple[float, float, float]:
    """Mann-Kendall trend test. Returns (S, z, two-sided p-value).

    S = sum over i<j of sign(x_j - x_i). Under "no trend", S is approximately
    normal with variance n(n-1)(2n+5)/18 minus a correction for tied values.
    """
    import math

    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 1.0
    diff = np.sign(x[None, :] - x[:, None])
    s = float(np.triu(diff, k=1).sum())
    _, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    var = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    if var <= 0:
        return s, 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return s, float(z), float(p)


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
    mk_z: float = 0.0
    mk_p: float = 1.0


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
    max_p_value: float = 0.01,
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
        _, mk_z, mk_p = mann_kendall(rss)
        significant = mk_z > 0 and mk_p <= max_p_value
        # Reject one-off step changes (e.g. a cache warming up): a leak keeps growing in both halves.
        half = len(rss) // 2
        s1, _, _ = analyze_series(ts[:half], rss[:half])
        s2, _, _ = analyze_series(ts[half:], rss[half:])
        sustained = min(s1, s2) >= 0.25 * min_slope_mb_per_min
        # A steady linear fit OR a significant monotonic trend (robust to GC sawtooth) qualifies.
        suspected = slope >= min_slope_mb_per_min and significant and sustained and (
            (r2 >= min_r2 and mono >= min_monotonic) or mk_z >= 4)
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
                mk_z=round(mk_z, 2),
                mk_p=float(f"{mk_p:.3g}"),
            )
        )
    findings.sort(key=lambda f: (not f.suspected, -f.slope_mb_per_min))
    return findings
