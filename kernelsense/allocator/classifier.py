"""ML workload classifier (Random Forest).

Training data: real processes rarely come with ground-truth labels, so the
model is trained on a *synthetic, OS-informed* dataset. Each class is generated
from the scheduler/memory behaviour that defines it (e.g. CPU-bound = high CPU,
involuntary switches dominate). Live processes can also be added with labels
from the rule-based classifier (weak supervision). Accuracy is reported with
stratified cross-validation and the model exposes feature importances so its
decisions can be explained.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..config import settings
from .features import FEATURES

LABELS = ["CPU-bound", "I/O-bound", "Interactive", "Memory-heavy", "Idle"]


def synthetic_dataset(n_per_class: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def clip(x, lo=0.0, hi=None):
        return np.clip(x, lo, hi)

    frames = []
    n = n_per_class
    # CPU-bound: high CPU, preempted often, few voluntary switches, little I/O
    vol = clip(rng.gamma(2, 3, n)); inv = clip(rng.gamma(4, 12, n))
    frames.append(pd.DataFrame({
        "cpu_mean": clip(rng.normal(85, 15, n), 25, 400), "cpu_std": clip(rng.normal(6, 4, n)),
        "rss_mb": clip(rng.lognormal(4.5, 0.8, n), 5), "rss_slope": rng.normal(0, 0.3, n),
        "vol_rate": vol, "invol_rate": inv, "io_rate_kb": clip(rng.exponential(20, n)),
        "threads": clip(rng.integers(1, 16, n), 1), "label": "CPU-bound"}))
    # I/O-bound: low CPU, blocks frequently, high disk throughput
    vol = clip(rng.gamma(5, 40, n)); inv = clip(rng.gamma(1.5, 2, n))
    frames.append(pd.DataFrame({
        "cpu_mean": clip(rng.normal(12, 7, n), 0.5, 45), "cpu_std": clip(rng.normal(5, 3, n)),
        "rss_mb": clip(rng.lognormal(4, 0.8, n), 5), "rss_slope": rng.normal(0, 0.3, n),
        "vol_rate": vol, "invol_rate": inv, "io_rate_kb": clip(rng.lognormal(8, 1.0, n), 300),
        "threads": clip(rng.integers(1, 12, n), 1), "label": "I/O-bound"}))
    # Interactive: short bursts, sleeps waiting for input, bursty CPU, low I/O
    vol = clip(rng.gamma(3, 10, n)); inv = clip(rng.gamma(1.5, 1.5, n))
    frames.append(pd.DataFrame({
        "cpu_mean": clip(rng.normal(6, 4, n), 0.5, 30), "cpu_std": clip(rng.normal(8, 4, n)),
        "rss_mb": clip(rng.lognormal(5, 0.7, n), 10), "rss_slope": rng.normal(0, 0.2, n),
        "vol_rate": vol, "invol_rate": inv, "io_rate_kb": clip(rng.exponential(15, n), 0, 250),
        "threads": clip(rng.integers(2, 40, n), 1), "label": "Interactive"}))
    # Memory-heavy: either a large resident set, or a resident set that keeps growing.
    # A leak can start small and use almost no CPU - its defining feature is the growth
    # rate, so the "growing" half deliberately looks idle on every other feature.
    large = rng.random(n) < 0.5
    frames.append(pd.DataFrame({
        "cpu_mean": np.where(large, clip(rng.normal(25, 15, n), 0.5, 100), clip(rng.exponential(1.0, n), 0, 30)),
        "cpu_std": np.where(large, clip(rng.normal(4, 3, n)), clip(rng.exponential(0.3, n))),
        "rss_mb": np.where(large, clip(rng.lognormal(7.3, 0.6, n), 600), clip(rng.lognormal(4.0, 1.2, n), 5)),
        "rss_slope": np.where(large, clip(rng.normal(3, 3, n), -1), clip(rng.lognormal(3, 1.2, n), 3)),
        "vol_rate": np.where(large, clip(rng.gamma(2, 6, n)), clip(rng.exponential(2.0, n))),
        "invol_rate": np.where(large, clip(rng.gamma(2, 5, n)), clip(rng.exponential(0.3, n))),
        "io_rate_kb": np.where(large, clip(rng.exponential(60, n)), clip(rng.exponential(3, n))),
        "threads": clip(rng.integers(1, 32, n), 1), "label": "Memory-heavy"}))
    # Idle: nearly no CPU, rare switches
    vol = clip(rng.exponential(0.3, n)); inv = clip(rng.exponential(0.05, n))
    frames.append(pd.DataFrame({
        "cpu_mean": clip(rng.exponential(0.15, n), 0, 1.5), "cpu_std": clip(rng.exponential(0.1, n)),
        "rss_mb": clip(rng.lognormal(3.5, 1.0, n), 1), "rss_slope": rng.normal(0, 0.02, n),
        "vol_rate": vol, "invol_rate": inv, "io_rate_kb": clip(rng.exponential(0.5, n)),
        "threads": clip(rng.integers(1, 6, n), 1), "label": "Idle"}))
    df = pd.concat(frames, ignore_index=True)
    total = df["vol_rate"] + df["invol_rate"]
    df["vol_ratio"] = np.where(total > 0, df["vol_rate"] / total.replace(0, 1), 0.0)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class TrainingReport:
    accuracy: float
    report: str
    confusion: pd.DataFrame
    importances: pd.Series
    n_samples: int


class WorkloadClassifier:
    def __init__(self, n_estimators: int = 200, seed: int = 0) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=12, min_samples_leaf=2, random_state=seed, class_weight="balanced"
        )
        self.seed = seed
        self.trained = False
        self.report: TrainingReport | None = None

    @staticmethod
    def _X(df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].astype(float).copy()
        # log-scale heavy-tailed rates so the trees split on orders of magnitude
        for c in ("rss_mb", "vol_rate", "invol_rate", "io_rate_kb"):
            X[c] = np.log1p(X[c].clip(lower=0))
        return X.to_numpy()

    def train(self, data: pd.DataFrame | None = None, extra: pd.DataFrame | None = None) -> TrainingReport:
        data = synthetic_dataset(seed=self.seed) if data is None else data
        if extra is not None and not extra.empty:
            data = pd.concat([data, extra[FEATURES + ["label"]]], ignore_index=True)
        X, y = self._X(data), data["label"].to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        pred = cross_val_predict(self.model, X, y, cv=cv)
        self.model.fit(X, y)
        self.trained = True
        labels = [l for l in LABELS if l in set(y)]
        self.report = TrainingReport(
            accuracy=float(accuracy_score(y, pred)),
            report=classification_report(y, pred, labels=labels, zero_division=0),
            confusion=pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels),
            importances=pd.Series(self.model.feature_importances_, index=FEATURES).sort_values(ascending=False),
            n_samples=len(data),
        )
        return self.report

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.trained:
            self.train()
        if features.empty:
            return features.assign(predicted=[], confidence=[])
        X = self._X(features)
        proba = self.model.predict_proba(X)
        out = features.copy()
        out["predicted"] = self.model.classes_[proba.argmax(axis=1)]
        out["confidence"] = proba.max(axis=1).round(3)
        return out

    def save(self, path: Path | None = None) -> Path:
        path = path or settings.data_dir / "workload_classifier.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path | None = None) -> "WorkloadClassifier":
        path = path or settings.data_dir / "workload_classifier.pkl"
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @classmethod
    def load_or_train(cls, path: Path | None = None) -> "WorkloadClassifier":
        try:
            return cls.load(path)
        except (OSError, pickle.UnpicklingError, AttributeError, EOFError):
            clf = cls()
            clf.train()
            try:
                clf.save(path)
            except OSError:
                pass
            return clf
