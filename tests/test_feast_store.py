from __future__ import annotations

import pandas as pd
import pytest

from alphagrid.data.feast_store import FeastFeatureStore, get_feature_store


def test_feast_store_registration():
    fs = FeastFeatureStore()
    assert "wind_features" in fs.views
    assert "market_features" in fs.views
    assert "wind_mw" in fs.views["wind_features"].features


def test_feast_historical_features():
    fs = get_feature_store()
    df = pd.DataFrame(
        {"wind_mw": [8000.0, 8500.0]},
        index=pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="h", tz="UTC"),
    )
    res = fs.get_historical_features(df, ["wind_features:wind_mw"])
    assert len(res) == 2
    assert "wind_mw" in res.columns


def test_feast_online_materialization():
    fs = FeastFeatureStore()
    fs.materialize_online_store("DE_LU", {"wind_speed_ms": 12.5, "day_ahead_price_eur_mwh": 72.0})

    online = fs.get_online_features("DE_LU", ["wind_speed_ms", "day_ahead_price_eur_mwh"])
    assert online["wind_speed_ms"] == 12.5
    assert online["day_ahead_price_eur_mwh"] == 72.0


def test_feast_empty_df_raises():
    fs = FeastFeatureStore()
    with pytest.raises(ValueError):
        fs.get_historical_features(pd.DataFrame(), ["wind_mw"])
