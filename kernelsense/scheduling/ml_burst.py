"""Machine-learned next-burst prediction compared against exponential averaging.

Exponential averaging (tau = a*t + (1-a)*tau) adapts slowly to *phase changes*
(a process that alternates between computation and I/O phases). A gradient
boosting regressor over lag features (last k bursts, their mean, trend and the
exponential average itself) can learn such patterns. Both are evaluated with
the same protocol: for every sequence, predict each burst from the bursts
before it, on sequences the model was not trained on.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .burst_predictor import best_alpha, exponential_average

K = 4  # number of lag features


def _features(history: list[float], alpha: float = 0.5) -> list[float]:
    h = list(history[-K:])
    h = [h[0]] * (K - len(h)) + h  # left-pad with the oldest value
    ea = exponential_average(history, alpha).next_prediction
    arr = np.array(history[-8:], dtype=float)
    trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) >= 2 else 0.0
    return [*h, float(arr.mean()), float(arr.std()), trend, ea, float(len(history) % 2)]


def _xy(sequences: list[list[float]], min_history: int = 2):
    X, y = [], []
    for seq in sequences:
        for i in range(min_history, len(seq)):
            X.append(_features(seq[:i]))
            y.append(seq[i])
    return np.array(X), np.array(y)


def synthetic_sequences(n: int = 120, length: int = 40, seed: int = 0) -> list[list[float]]:
    """Burst sequences with realistic structure: phases, periodic patterns, noise."""
    rng = random.Random(seed)
    seqs = []
    for i in range(n):
        kind = i % 4
        if kind == 0:  # phase changes: CPU phase then I/O phase
            seq = []
            while len(seq) < length:
                level = rng.choice([2, 4, 20, 40])
                seq += [max(0.5, rng.gauss(level, level * 0.15)) for _ in range(rng.randint(3, 8))]
        elif kind == 1:  # alternating short/long (e.g. read-compute loops)
            a, b = rng.uniform(1, 5), rng.uniform(15, 40)
            seq = [max(0.5, rng.gauss(a if j % 2 else b, 1)) for j in range(length)]
        elif kind == 2:  # slow drift
            base = rng.uniform(5, 15)
            seq = [max(0.5, base + 0.4 * j + rng.gauss(0, 1.5)) for j in range(length)]
        else:  # stationary noisy
            m = rng.uniform(3, 30)
            seq = [max(0.5, rng.gauss(m, m * 0.3)) for _ in range(length)]
        seqs.append(seq[:length])
    return seqs


class MLBurstPredictor:
    def __init__(self, seed: int = 0) -> None:
        self.model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, random_state=seed)
        self.trained = False

    def fit(self, sequences: list[list[float]]) -> "MLBurstPredictor":
        X, y = _xy(sequences)
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict_next(self, history: list[float]) -> float:
        return float(self.model.predict([_features(history)])[0])

    def walk_forward(self, seq: list[float], min_history: int = 2) -> list[float]:
        return [self.predict_next(seq[:i]) for i in range(min_history, len(seq))]


def compare_predictors(train: list[list[float]], test: list[list[float]], alpha: float = 0.5) -> pd.DataFrame:
    """MAE per method on held-out sequences (lower is better)."""
    ml = MLBurstPredictor().fit(train)
    rows = []
    for idx, seq in enumerate(test):
        if len(seq) < 4:
            continue
        actual = np.array(seq[2:])
        ea = np.array(exponential_average(seq, alpha).predicted[2:])
        a_best, _ = best_alpha(seq[:max(3, len(seq) // 2)])  # alpha tuned on the first half only
        ea_tuned = np.array(exponential_average(seq, a_best).predicted[2:])
        last = np.array(seq[1:-1])
        pred_ml = np.array(ml.walk_forward(seq))
        rows.append({
            "sequence": idx,
            "last_burst": float(np.mean(np.abs(actual - last))),
            f"exp_avg_a={alpha}": float(np.mean(np.abs(actual - ea))),
            "exp_avg_tuned": float(np.mean(np.abs(actual - ea_tuned))),
            "gradient_boosting": float(np.mean(np.abs(actual - pred_ml))),
        })
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    methods = [c for c in table.columns if c != "sequence"]
    mae = table[methods].mean()
    base = mae[[c for c in methods if c.startswith("exp_avg_a")][0]]
    return pd.DataFrame({
        "method": methods,
        "MAE": mae.values.round(3),
        "improvement_vs_exp_avg_%": ((base - mae.values) / base * 100).round(1),
        "wins": [(table[m] == table[methods].min(axis=1)).sum() for m in methods],
    }).sort_values("MAE").reset_index(drop=True)
