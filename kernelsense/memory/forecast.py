"""Memory usage forecasting.

All forecasters implement ``fit(y)`` and ``predict(steps)`` returning point
forecasts with a prediction interval. ``evaluate`` holds out the last part of
the series so every model is compared against the naive baseline on data it
has not seen. A model is only worth using if it beats the baseline.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    model: str
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class Forecaster(ABC):
    name = "Forecaster"

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._resid_std = 0.0

    def fit(self, y) -> "Forecaster":
        y = np.asarray(y, dtype=float)
        if len(y) < 3:
            raise ValueError("need at least 3 observations")
        self._y = y
        self._fit(y)
        return self

    @abstractmethod
    def _fit(self, y: np.ndarray) -> None: ...

    @abstractmethod
    def _point(self, steps: int) -> np.ndarray: ...

    def predict(self, steps: int, z: float = 1.96) -> Forecast:
        mean = self._point(steps)
        # interval widens with horizon (random-walk style growth of uncertainty)
        width = z * self._resid_std * np.sqrt(np.arange(1, steps + 1))
        # memory quantities cannot be negative
        return Forecast(self.name, mean, np.maximum(mean - width, 0.0), mean + width)


class NaiveForecaster(Forecaster):
    name = "Naive (last value)"

    def _fit(self, y):
        self._resid_std = float(np.std(np.diff(y))) if len(y) > 1 else 0.0

    def _point(self, steps):
        return np.full(steps, self._y[-1])


class MovingAverageForecaster(Forecaster):
    def __init__(self, window: int = 10) -> None:
        super().__init__()
        self.window = window
        self.name = f"Moving average ({window})"

    def _fit(self, y):
        w = min(self.window, len(y))
        ma = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
        self._level = ma[-1]
        self._resid_std = float(np.std(y - ma))

    def _point(self, steps):
        return np.full(steps, self._level)


class LinearTrendForecaster(Forecaster):
    """Least-squares trend on the most recent ``window`` points."""

    def __init__(self, window: int | None = 60) -> None:
        super().__init__()
        self.window = window
        self.name = "Linear trend"

    def _fit(self, y):
        seg = y[-self.window :] if self.window else y
        x = np.arange(len(seg))
        self._slope, self._intercept = np.polyfit(x, seg, 1)
        self._n = len(seg)
        self._resid_std = float(np.std(seg - (self._slope * x + self._intercept)))

    def _point(self, steps):
        x = np.arange(self._n, self._n + steps)
        return self._slope * x + self._intercept

    @property
    def slope(self) -> float:
        return float(self._slope)


class HoltForecaster(Forecaster):
    """Holt's linear exponential smoothing (level + damped trend) via statsmodels."""

    name = "Holt (damped trend)"

    def _fit(self, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.ptp(y) == 0:
                self._const = y[-1]
                self._model = None
                self._resid_std = 0.0
                return
            self._model = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
            self._resid_std = float(np.std(self._model.resid))

    def _point(self, steps):
        if self._model is None:
            return np.full(steps, self._const)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self._model.forecast(steps), dtype=float)


def default_models() -> list[Forecaster]:
    return [NaiveForecaster(), MovingAverageForecaster(10), LinearTrendForecaster(60), HoltForecaster()]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    err = actual - predicted
    nz = np.abs(actual) > 1e-9
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else float("nan"),
    }


def evaluate(y, holdout: float = 0.2, models: list[Forecaster] | None = None) -> pd.DataFrame:
    """Train on the first (1-holdout) of the series, score on the rest."""
    y = np.asarray(y, dtype=float)
    h = max(1, int(len(y) * holdout))
    train, test = y[:-h], y[-h:]
    rows = []
    for m in models or default_models():
        try:
            pred = m.fit(train).predict(h).mean
            rows.append({"model": m.name, **metrics(test, pred)})
        except Exception as exc:  # noqa: BLE001 - report per-model failures in the table
            rows.append({"model": m.name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "error": str(exc)})
    df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    base = df.loc[df["model"].str.startswith("Naive"), "MAE"]
    if not base.empty and base.iloc[0] > 0:
        df["improvement_vs_naive_%"] = ((base.iloc[0] - df["MAE"]) / base.iloc[0] * 100).round(1)
    return df


def best_model(y, holdout: float = 0.2) -> Forecaster:
    table = evaluate(y, holdout)
    name = table.dropna(subset=["MAE"]).iloc[0]["model"]
    for m in default_models():
        if m.name == name:
            return m.fit(y)
    return NaiveForecaster().fit(y)


def time_to_threshold(forecast: Forecast, threshold: float, interval_s: float) -> float | None:
    """Seconds until the point forecast crosses ``threshold`` (None if it does not)."""
    idx = np.where(forecast.mean >= threshold)[0]
    return float((idx[0] + 1) * interval_s) if len(idx) else None


def regularize(df: pd.DataFrame, column: str, interval_s: float) -> pd.Series:
    """Resample an irregularly sampled metric onto a fixed grid (needed by time-series models)."""
    s = df.set_index("time")[column].astype(float)
    rule = f"{max(int(interval_s), 1)}s"
    return s.resample(rule).mean().interpolate(limit_direction="both").dropna()
