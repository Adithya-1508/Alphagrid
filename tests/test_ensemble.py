from __future__ import annotations

import pandas as pd

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.ensemble import predict_ensemble, train_ensemble_models


def test_ensemble_training_and_prediction():
    df = generate_synthetic("2026-05-01T00:00:00Z", "2026-05-15T00:00:00Z")
    metrics = train_ensemble_models(df)
    assert "ensemble_mae" in metrics
    assert metrics["ensemble_mae"] >= 0.0

    preds = predict_ensemble(df, horizon_hours=24)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 24
    assert "forecast" in preds.columns
    assert isinstance(preds.index, pd.DatetimeIndex)
