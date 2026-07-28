from __future__ import annotations

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.drift import detect_data_drift


def test_drift_detection_no_drift():
    ref_df = generate_synthetic("2026-05-01T00:00:00Z", "2026-05-05T00:00:00Z")
    cur_df = generate_synthetic("2026-05-06T00:00:00Z", "2026-05-10T00:00:00Z")
    report = detect_data_drift(ref_df, cur_df, features=["wind_mw", "wind_speed_ms"])

    assert isinstance(report, dict)
    assert "drift_detected" in report
    assert "feature_reports" in report
    assert "wind_mw" in report["feature_reports"]


def test_drift_detection_with_drift():
    ref_df = generate_synthetic("2026-05-01T00:00:00Z", "2026-05-05T00:00:00Z")
    cur_df = ref_df.copy()
    cur_df["wind_mw"] = cur_df["wind_mw"] * 5.0  # Introduce massive drift

    report = detect_data_drift(ref_df, cur_df, features=["wind_mw"])
    assert report["drift_detected"] is True
    assert report["feature_reports"]["wind_mw"]["drift_detected"] is True
