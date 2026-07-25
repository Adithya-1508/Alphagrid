import pandas as pd
from alphagrid.forecasting.anomalies import detect_anomalies


def test_detect_anomalies_basic():
    # 5 days of hourly data (120 hours)
    idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    actual_values = [100.0] * 96 + [10.0] * 24  # Day 5 shortage anomaly
    forecast_values = [100.0] * 120

    actuals = pd.Series(actual_values, index=idx)
    forecasts = pd.Series(forecast_values, index=idx)

    anomalies = detect_anomalies(actuals, forecasts, threshold=1.5)
    assert len(anomalies) == 1
    assert anomalies[0]["date"] == "2026-01-05"
    assert anomalies[0]["direction"] == "Shortage"
    assert anomalies[0]["magnitude"] < 0


def test_detect_anomalies_empty():
    idx = pd.date_range("2026-01-01", periods=0, freq="h", tz="UTC")
    actuals = pd.Series([], index=idx, dtype=float)
    forecasts = pd.Series([], index=idx, dtype=float)

    anomalies = detect_anomalies(actuals, forecasts)
    assert anomalies == []


def test_detect_anomalies_string_index_coercion():
    idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    actual_values = [100.0] * 96 + [10.0] * 24
    forecast_values = [100.0] * 120

    actuals = pd.Series(actual_values, index=idx.astype(str))
    forecasts = pd.Series(forecast_values, index=idx.astype(str))

    # Passing string index should be automatically converted to DatetimeIndex
    anomalies = detect_anomalies(actuals, forecasts, threshold=1.5)
    assert len(anomalies) == 1
    assert anomalies[0]["date"] == "2026-01-05"
