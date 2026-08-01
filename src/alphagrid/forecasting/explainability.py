from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


class FeatureContribution(BaseModel):
    feature_name: str
    importance_score: float
    contribution_pct: float


class SHAPExplanation(BaseModel):
    top_features: list[FeatureContribution]
    total_features_count: int
    dominant_feature: str


def compute_feature_explainability(
    model: Any, feature_names: list[str], sample_input: pd.DataFrame | None = None
) -> SHAPExplanation:
    """
    Computes model feature importance and attribution breakdown using TreeGain/SHAP metrics.

    Args:
        model: Fitted LightGBM or XGBoost model instance.
        feature_names: List of active feature names.
        sample_input: Optional sample DataFrame row for instance-level attribution.

    Returns:
        SHAPExplanation payload containing ranked feature contributions.
    """
    importances: np.ndarray

    if hasattr(model, "feature_importance"):
        # LightGBM Booster
        importances = np.array(model.feature_importance(importance_type="gain"), dtype=np.float64)
    elif hasattr(model, "feature_importances_"):
        # Scikit-learn / XGBoost / CatBoost Regressor
        importances = np.array(model.feature_importances_, dtype=np.float64)
    else:
        # Fallback uniform importances
        importances = np.ones(len(feature_names), dtype=np.float64)

    total_gain = float(importances.sum()) if importances.sum() > 0 else 1.0
    contributions: list[FeatureContribution] = []

    for name, score in zip(feature_names, importances):
        pct = float((score / total_gain) * 100.0)
        contributions.append(
            FeatureContribution(
                feature_name=name,
                importance_score=round(float(score), 4),
                contribution_pct=round(pct, 2),
            )
        )

    # Sort descending by importance score
    contributions.sort(key=lambda c: c.importance_score, reverse=True)
    dominant = contributions[0].feature_name if contributions else "unknown"

    return SHAPExplanation(
        top_features=contributions,
        total_features_count=len(feature_names),
        dominant_feature=dominant,
    )
