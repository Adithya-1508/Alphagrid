from __future__ import annotations

import pandas as pd
import pytest

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.predict import predict_next_hours, predict_quantile_forecasts
from alphagrid.forecasting.train import train_model


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    # Generate sufficient synthetic data and train the model once for testing
    df = generate_synthetic("2026-01-01", "2026-01-20")
    train_model(df)


def test_predict_datetime_index():
    df = generate_synthetic("2026-01-21", "2026-01-23")
    res = predict_next_hours(df)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 24
    assert set(["forecast", "lower_bound", "upper_bound", "p10", "p50", "p90"]).issubset(
        res.columns
    )
    assert isinstance(res.index, pd.DatetimeIndex)


def test_predict_plain_index_strings():
    df = generate_synthetic("2026-01-21", "2026-01-23")
    # Convert index to plain strings
    df.index = df.index.astype(str)
    assert not isinstance(df.index, pd.DatetimeIndex)

    res = predict_next_hours(df)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 24
    assert isinstance(res.index, pd.DatetimeIndex)


def test_predict_plain_index_timestamps():
    df = generate_synthetic("2026-01-21", "2026-01-23")
    # Convert index to a plain object Index containing Timestamps
    df.index = pd.Index(list(df.index), dtype="object")
    assert not isinstance(df.index, pd.DatetimeIndex)

    res = predict_next_hours(df)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 24
    assert isinstance(res.index, pd.DatetimeIndex)


def test_quantile_ordering():
    df = generate_synthetic("2026-01-21", "2026-01-23")
    res = predict_quantile_forecasts(df, horizon_hours=24)
    assert (res["p10"] <= res["p50"]).all()
    assert (res["p50"] <= res["p90"]).all()
