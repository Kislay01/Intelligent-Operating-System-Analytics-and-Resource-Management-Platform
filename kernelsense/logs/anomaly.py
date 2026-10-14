"""Time-window anomaly detection on log activity.

Logs are bucketed into fixed windows. Each window becomes a feature vector
(volume, errors, warnings, critical lines, distinct sources, OS event counts).
An Isolation Forest flags windows that look unlike the rest, and a robust
z-score on error volume gives a simple, explainable second opinion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .events import detect_events

ERROR_LEVELS = {"emerg", "alert", "crit", "err"}


def window_features(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    d = df.dropna(subset=["timestamp"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["sev"] = d["severity"].astype(str)
    d = d.set_index("timestamp").sort_index()
    grp = d.groupby(pd.Grouper(freq=window))
    feats = pd.DataFrame({
        "lines": grp.size(),
        "errors": grp["sev"].apply(lambda s: s.isin(ERROR_LEVELS).sum()),
        "warnings": grp["sev"].apply(lambda s: (s == "warning").sum()),
        "critical": grp["sev"].apply(lambda s: s.isin({"emerg", "alert", "crit"}).sum()),
        "sources": grp["source"].nunique(),
    }).fillna(0)
    events = detect_events(df)
    if not events.empty:
        ev = events.dropna(subset=["timestamp"]).set_index("timestamp").groupby(pd.Grouper(freq=window)).size()
        feats["os_events"] = ev.reindex(feats.index, fill_value=0)
    else:
        feats["os_events"] = 0
    return feats.astype(float)


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else (x.std() or 1.0)
    return (x - med) / scale


def detect_anomalies(df: pd.DataFrame, window: str = "1min", contamination: float = 0.05, seed: int = 0) -> pd.DataFrame:
    feats = window_features(df, window)
    if feats.empty:
        return feats
    feats["error_z"] = robust_z(feats["errors"]).round(2)
    if len(feats) >= 8:
        cols = ["lines", "errors", "warnings", "critical", "sources", "os_events"]
        X = np.log1p(feats[cols].to_numpy())
        model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
        model.fit(X)
        feats["iforest_score"] = (-model.score_samples(X)).round(3)
        feats["iforest_anomaly"] = model.predict(X) == -1
    else:
        feats["iforest_score"] = 0.0
        feats["iforest_anomaly"] = False
    feats["zscore_anomaly"] = feats["error_z"] >= 3.5
    feats["anomaly"] = feats["iforest_anomaly"] | feats["zscore_anomaly"]
    return feats


def top_sources(df: pd.DataFrame, n: int = 10, errors_only: bool = False) -> pd.DataFrame:
    d = df
    if errors_only:
        d = df[df["severity"].astype(str).isin(ERROR_LEVELS)]
    return d["source"].value_counts().head(n).rename_axis("source").reset_index(name="count")
