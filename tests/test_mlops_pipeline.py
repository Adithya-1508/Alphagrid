from __future__ import annotations

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.mlops_pipeline import check_and_trigger_retrain


def test_mlops_pipeline_no_drift():
    ref_df = generate_synthetic("2026-01-01", "2026-01-10")
    curr_df = generate_synthetic("2026-01-01", "2026-01-10")

    status = check_and_trigger_retrain(ref_df, curr_df)
    assert status.drift_detected is False
    assert status.retrain_triggered is False
    assert status.status_message == "No data drift detected. Models are healthy."


def test_mlops_pipeline_forced_retrain():
    ref_df = generate_synthetic("2026-01-01", "2026-01-15")
    curr_df = generate_synthetic("2026-01-16", "2026-01-30")

    status = check_and_trigger_retrain(ref_df, curr_df, force_retrain=True)
    assert status.retrain_triggered is True
    assert status.quantile_metrics is not None
    assert status.ensemble_metrics is not None
