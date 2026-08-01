from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from alphagrid.forecasting.explainability import compute_feature_explainability


def test_compute_feature_explainability_lgb():
    X = pd.DataFrame(
        {
            "wind_speed_ms": np.random.rand(100) * 15.0,
            "temperature_c": np.random.rand(100) * 25.0,
            "lag_24": np.random.rand(100) * 5000.0,
        }
    )
    y = X["wind_speed_ms"] * 500.0 + np.random.randn(100) * 50.0

    ds = lgb.Dataset(X, label=y)
    booster = lgb.train({"verbosity": -1}, ds, num_boost_round=10)

    explanation = compute_feature_explainability(booster, list(X.columns))

    assert explanation.total_features_count == 3
    assert len(explanation.top_features) == 3
    assert explanation.dominant_feature == "wind_speed_ms"
    assert explanation.top_features[0].feature_name == "wind_speed_ms"
    assert explanation.top_features[0].contribution_pct > 50.0
