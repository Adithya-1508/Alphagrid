from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from .drift import detect_data_drift
from .ensemble import train_ensemble_models
from .train import FEATURES, train_quantile_models


class RetrainStatus(BaseModel):
    drift_detected: bool
    retrain_triggered: bool
    drifted_features: list[str]
    quantile_metrics: dict[str, float] | None = None
    ensemble_metrics: dict[str, float] | None = None
    status_message: str


def check_and_trigger_retrain(
    ref_df: pd.DataFrame, curr_df: pd.DataFrame, force_retrain: bool = False
) -> RetrainStatus:
    """
    Evaluates dataset drift between reference and current data.
    If drift is detected (PSI > 0.20 or p < 0.05), automatically triggers retraining.
    """
    active_feats = [f for f in FEATURES if f in ref_df.columns and f in curr_df.columns]
    if not active_feats:
        active_feats = ["wind_speed_ms", "temperature_c"]

    drift_report = detect_data_drift(ref_df, curr_df, active_feats)
    is_drifted = bool(drift_report.get("drift_detected", False)) or force_retrain

    reports = drift_report.get("feature_reports", {})
    if isinstance(reports, dict):
        drifted_feats = [
            feat
            for feat, rep in reports.items()
            if isinstance(rep, dict) and rep.get("drift_detected", False)
        ]
    else:
        drifted_feats = []

    if not is_drifted:
        return RetrainStatus(
            drift_detected=False,
            retrain_triggered=False,
            drifted_features=[],
            status_message="No data drift detected. Models are healthy.",
        )

    # Trigger automatic retraining pipeline
    q_metrics = train_quantile_models(curr_df)
    ens_metrics = train_ensemble_models(curr_df)

    return RetrainStatus(
        drift_detected=True,
        retrain_triggered=True,
        drifted_features=drifted_feats,
        quantile_metrics=q_metrics,
        ensemble_metrics=ens_metrics,
        status_message="Data drift detected. Models successfully retrained.",
    )
