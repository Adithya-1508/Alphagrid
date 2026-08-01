from __future__ import annotations

import pandas as pd
import pytest

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.sequence_models import SequenceForecaster


def test_sequence_forecaster_fit_and_predict():
    df = generate_synthetic("2026-01-01", "2026-01-10")
    forecaster = SequenceForecaster(sequence_length=24)

    metrics = forecaster.fit(df)
    assert "sequence_loss" in metrics
    assert metrics["sequence_loss"] >= 0.0

    preds = forecaster.predict(df, horizon_hours=24)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 24
    assert "sequence_forecast" in preds.columns
    assert (preds["sequence_forecast"] >= 0.0).all()


def test_sequence_forecaster_insufficient_history_raises():
    df = generate_synthetic("2026-01-01T00:00:00Z", "2026-01-01T05:00:00Z")
    forecaster = SequenceForecaster(sequence_length=24)
    with pytest.raises(ValueError):
        forecaster.fit(df)
