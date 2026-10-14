"""Evaluation on Loghub benchmark datasets (He et al., "Loghub", ISSRE 2023).

The 2k-line samples from github.com/logpai/loghub come with ground-truth event
ids (for template-mining accuracy) and, for BGL (Blue Gene/L supercomputer),
per-line alert labels (for anomaly detection).

Reported metrics:
* Drain grouping accuracy on BGL, HDFS and Linux system logs
* anomaly detection precision / recall / F1 on BGL windows for
  - the n-gram DeepLog-style sequence model (trained on normal windows)
  - Isolation Forest over template-count vectors (unsupervised)
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import settings
from .drain import Drain, grouping_accuracy
from .sequence import NGramSequenceModel

BASE = "https://raw.githubusercontent.com/logpai/loghub/master"
DATASETS = {"BGL": "BGL/BGL_2k.log_structured.csv", "HDFS": "HDFS/HDFS_2k.log_structured.csv",
            "Linux": "Linux/Linux_2k.log_structured.csv"}


def load(name: str, cache_dir: Path | None = None) -> pd.DataFrame:
    cache = (cache_dir or settings.data_dir / "loghub")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(DATASETS[name]).name
    if not path.exists():
        with urllib.request.urlopen(f"{BASE}/{DATASETS[name]}", timeout=60) as resp:  # noqa: S310 - fixed URL
            path.write_bytes(resp.read())
    return pd.read_csv(path)


def parsing_accuracy(df: pd.DataFrame, depth: int = 4, sim: float = 0.5) -> dict:
    d = Drain(depth=depth, sim_threshold=sim)
    d.fit(df["Content"].astype(str).tolist())
    pred = d.final_assignments()
    return {"lines": len(df), "templates_found": len(d.clusters), "templates_truth": df["EventId"].nunique(),
            "grouping_accuracy": round(grouping_accuracy(pred, df["EventId"].tolist()), 4)}


def _prf(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def bgl_anomaly_detection(df: pd.DataFrame | None = None, window: int = 20, train_frac: float = 0.4,
                          history: int = 3, top_k: int = 5, seed: int = 0) -> dict:
    df = load("BGL") if df is None else df
    drain = Drain()
    keys = drain.fit(df["Content"].astype(str).tolist())
    keys = drain.final_assignments()
    labels = (df["Label"].astype(str) != "-").to_numpy()
    n_win = len(df) // window
    seqs = [keys[i * window:(i + 1) * window] for i in range(n_win)]
    y = np.array([labels[i * window:(i + 1) * window].any() for i in range(n_win)])

    split = int(n_win * train_frac)
    train_idx = [i for i in range(split) if not y[i]]  # DeepLog trains on normal executions only
    test_idx = list(range(split, n_win))

    model = NGramSequenceModel(history=history, top_k=top_k).fit([seqs[i] for i in train_idx])
    seq_pred = [model.is_anomalous(seqs[i]) for i in test_idx]

    vocab = sorted(set(keys))
    index = {k: j for j, k in enumerate(vocab)}
    X = np.zeros((n_win, len(vocab)))
    for i, s in enumerate(seqs):
        for k in s:
            X[i, index[k]] += 1
    iso = IsolationForest(n_estimators=300, contamination=max(0.01, min(0.5, y.mean())), random_state=seed)
    iso.fit(np.log1p(X))
    iso_pred = iso.predict(np.log1p(X[test_idx])) == -1

    y_test = y[test_idx]
    return {
        "windows": n_win,
        "window_lines": window,
        "anomalous_windows": int(y.sum()),
        "test_windows": len(test_idx),
        "test_anomalous": int(y_test.sum()),
        "ngram_deeplog": _prf(y_test, seq_pred),
        "isolation_forest": _prf(y_test, iso_pred),
    }


def run_all(cache_dir: Path | None = None) -> dict:
    out = {"parsing": {}}
    for name in DATASETS:
        out["parsing"][name] = parsing_accuracy(load(name, cache_dir))
    out["bgl_anomaly"] = bgl_anomaly_detection(load("BGL", cache_dir))
    return out


def save_report(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path
