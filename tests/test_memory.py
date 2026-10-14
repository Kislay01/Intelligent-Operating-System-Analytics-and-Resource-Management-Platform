import time

import numpy as np
import pandas as pd

from kernelsense.memory import (
    HoltForecaster, LinearTrendForecaster, NaiveForecaster, best_model, detect_leaks, evaluate, time_to_threshold,
)


def trend_series(n=300, slope=0.05, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return 40 + slope * np.arange(n) + rng.normal(0, noise, n)


def test_trend_models_beat_naive_on_trending_data():
    table = evaluate(trend_series())
    naive = table.loc[table["model"].str.startswith("Naive"), "MAE"].iloc[0]
    trend = table.loc[table["model"] == "Linear trend", "MAE"].iloc[0]
    assert trend < naive
    assert table.iloc[0]["model"] != "Naive (last value)"


def test_forecast_shapes_and_interval():
    for m in (NaiveForecaster(), LinearTrendForecaster(), HoltForecaster()):
        fc = m.fit(trend_series()).predict(20)
        assert len(fc.mean) == 20
        assert np.all(fc.lower <= fc.mean) and np.all(fc.mean <= fc.upper)
        assert np.all(fc.lower >= 0)


def test_time_to_threshold():
    y = trend_series(slope=0.1, noise=0.01)
    fc = LinearTrendForecaster().fit(y).predict(200)
    eta = time_to_threshold(fc, 80, interval_s=2)
    assert eta is not None and 0 < eta < 400
    assert time_to_threshold(fc, 1000, 2) is None


def test_constant_series_is_handled():
    m = best_model(np.full(50, 30.0))
    assert np.allclose(m.predict(5).mean, 30.0)


def test_leak_detection():
    rng = np.random.default_rng(1)
    ts = time.time() + np.arange(40) * 5.0
    df = pd.DataFrame({
        "ts": np.r_[ts, ts],
        "pid": [1] * 40 + [2] * 40,
        "name": ["leaky"] * 40 + ["steady"] * 40,
        "rss_mb": np.r_[100 + np.arange(40) * 0.5, 200 + rng.normal(0, 2, 40)],
    })
    findings = {f.name: f for f in detect_leaks(df)}
    assert findings["leaky"].suspected
    assert abs(findings["leaky"].slope_mb_per_min - 6.0) < 0.01
    assert not findings["steady"].suspected
    assert detect_leaks(pd.DataFrame()) == []
