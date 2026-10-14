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


class QuantileGBMForecaster(Forecaster):
    """Gradient-boosted quantile regression on lag features.

    Three HistGradientBoosting models (quantiles 5%, 50%, 95%) predict the next
    *change* in the series from the last ``lags`` values, rolling statistics
    and the recent slope. Forecasts are produced recursively: the median step
    is appended to the history and the model is applied again. The 5%/95%
    steps accumulate into the prediction interval, so the band reflects what
    the data says about uncertainty instead of an assumed distribution.
    """

    name = "Quantile GBM"

    def __init__(self, lags: int = 12, quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)) -> None:
        super().__init__()
        self.lags = lags
        self.quantiles = quantiles

    def _row(self, hist: np.ndarray) -> list[float]:
        w = hist[-self.lags:]
        d = np.diff(w) if len(w) > 1 else np.zeros(1)
        x = np.arange(len(w))
        slope = float(np.polyfit(x, w, 1)[0]) if len(w) > 2 else 0.0
        return [*(w - w[-1]), float(d.mean()), float(d.std()), float(d[-3:].mean()), slope]

    def _fit(self, y):
        from sklearn.ensemble import HistGradientBoostingRegressor

        if len(y) <= self.lags + 5:
            raise ValueError(f"Quantile GBM needs more than {self.lags + 5} observations")
        X = np.array([self._row(y[: i + 1]) for i in range(self.lags - 1, len(y) - 1)])
        target = np.diff(y)[self.lags - 1:]
        self._models = []
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150, learning_rate=0.05,
                                              max_leaf_nodes=15, min_samples_leaf=5, random_state=0)
            m.fit(X, target)
            self._models.append(m)
        self._resid_std = float(np.std(target - self._models[1].predict(X)))

    def _paths(self, steps: int):
        hist = list(self._y)
        lo_acc = hi_acc = 0.0
        mean, lo, hi = [], [], []
        for _ in range(steps):
            row = [self._row(np.asarray(hist))]
            d_lo, d_mid, d_hi = (float(m.predict(row)[0]) for m in self._models)
            nxt = hist[-1] + d_mid
            hist.append(nxt)
            lo_acc += min(d_lo - d_mid, 0.0)
            hi_acc += max(d_hi - d_mid, 0.0)
            mean.append(nxt)
            lo.append(nxt + lo_acc)
            hi.append(nxt + hi_acc)
        return np.array(mean), np.array(lo), np.array(hi)

    def _point(self, steps):
        return self._paths(steps)[0]

    def predict(self, steps: int, z: float = 1.96) -> Forecast:
        mean, lo, hi = self._paths(steps)
        return Forecast(self.name, mean, np.maximum(lo, 0.0), hi)


def default_models(include_ml: bool = True) -> list[Forecaster]:
    models: list[Forecaster] = [NaiveForecaster(), MovingAverageForecaster(10), LinearTrendForecaster(60),
                                HoltForecaster()]
    if include_ml:
        models.append(QuantileGBMForecaster())
    return models


def interval_coverage(actual, forecast: Forecast) -> float:
    """Share of actual values inside the prediction interval (target ~95%)."""
    a = np.asarray(actual, float)
    n = min(len(a), len(forecast.mean))
    return float(np.mean((a[:n] >= forecast.lower[:n]) & (a[:n] <= forecast.upper[:n])))


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
