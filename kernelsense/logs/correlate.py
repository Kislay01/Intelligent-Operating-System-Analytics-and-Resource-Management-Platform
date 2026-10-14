"""Cross-signal correlation: OS log events x metric anomalies.

Root-cause analysis usually needs two signals side by side: *what the metrics
did* (memory climbing, CPU saturated, pressure stalls) and *what the system
said* (OOM kill, hung task, I/O error). This module detects metric anomalies
with a robust z-score on each metric's sample-to-sample change, matches every
OS log event to metric anomalies within a time window, and produces an ordered
incident timeline with a one-line narrative.
"""

from __future__ import annotations

import pandas as pd

from .anomaly import robust_z

METRICS = {
    "cpu_percent": "CPU",
    "mem_percent": "Memory",
    "swap_percent": "Swap",
    "load1": "Load",
}

# Which metric movements plausibly explain which log events.
EXPLAINS = {
    "OOM kill": ["Memory", "Swap"],
    "Memory pressure": ["Memory", "Swap"],
    "Hung task": ["Load"],
    "Soft/hard lockup": ["CPU", "Load"],
    "Thermal throttling": ["CPU"],
    "Disk I/O error": ["Load"],
}


def metric_anomalies(history: pd.DataFrame, z: float = 3.5) -> pd.DataFrame:
    """Rows where a metric jumped unusually (|robust z| of its change >= z) or crossed a hard limit."""
    if history.empty:
        return pd.DataFrame(columns=["time", "metric", "value", "change", "z", "kind"])
    rows = []
    h = history.sort_values("time")
    for col, label in METRICS.items():
        if col not in h:
            continue
        s = h[col].astype(float)
        d = s.diff().fillna(0)
        zz = robust_z(d)
        for t, v, dv, zv in zip(h["time"], s, d, zz):
            if abs(zv) >= z and abs(dv) > 0:
                rows.append({"time": t, "metric": label, "value": round(v, 2), "change": round(dv, 2),
                             "z": round(float(zv), 1), "kind": "spike" if dv > 0 else "drop"})
        limit = {"cpu_percent": 95, "mem_percent": 90, "swap_percent": 50}.get(col)
        if limit is not None:
            for t, v in zip(h["time"], s):
                if v >= limit:
                    rows.append({"time": t, "metric": label, "value": round(v, 2), "change": 0.0, "z": 0.0,
                                 "kind": f">= {limit}%"})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["time", "metric", "value", "change", "z", "kind"])


def correlate(events: pd.DataFrame, anomalies: pd.DataFrame, window: str = "2min") -> pd.DataFrame:
    """For each OS event, attach metric anomalies seen in [event - window, event + 30s]."""
    if events.empty:
        return pd.DataFrame(columns=["timestamp", "category", "message", "related_metrics", "narrative"])
    w = pd.Timedelta(window)
    out = []
    for ev in events.dropna(subset=["timestamp"]).itertuples(index=False):
        near = anomalies[(anomalies["time"] >= ev.timestamp - w) &
                         (anomalies["time"] <= ev.timestamp + pd.Timedelta("30s"))] if not anomalies.empty else anomalies
        expected = EXPLAINS.get(ev.category, [])
        related = near[near["metric"].isin(expected)] if expected else near
        if not related.empty:
            first = related.iloc[0]
            lead = (ev.timestamp - first["time"]).total_seconds()
            metrics = ", ".join(sorted(set(related["metric"])))
            narrative = (f"{metrics} anomaly ({first['metric']} {first['kind']} to {first['value']}) "
                         f"{abs(lead):.0f}s {'before' if lead >= 0 else 'after'} → {ev.category}: {ev.message[:90]}")
        else:
            metrics = ""
            narrative = f"{ev.category} with no matching metric anomaly: {ev.message[:90]}"
        out.append({"timestamp": ev.timestamp, "category": ev.category, "message": ev.message,
                    "related_metrics": metrics, "narrative": narrative})
    return pd.DataFrame(out).sort_values("timestamp").reset_index(drop=True)


def incident_timeline(history: pd.DataFrame, log_df: pd.DataFrame, z: float = 3.5, window: str = "2min") -> dict:
    from .events import detect_events

    events = detect_events(log_df)
    anomalies = metric_anomalies(history, z)
    corr = correlate(events, anomalies, window)
    explained = int((corr["related_metrics"] != "").sum()) if not corr.empty else 0
    return {"events": events, "metric_anomalies": anomalies, "correlated": corr,
            "explained": explained, "total_events": len(corr)}
