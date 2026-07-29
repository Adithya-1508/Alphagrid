from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

MODEL_DIR = Path("artifacts") / "models"
BASE_FEATURES = [
    "wind_speed_ms",
    "temperature_c",
    "lag_24",
    "wind_speed_lag_24",
    "hour",
    "dayofweek",
    "month",
]
PRICE_FEATURES = ["day_ahead_price_eur_mwh", "gas_price_eur_mwh", "carbon_price_eur_t"]
FEATURES = BASE_FEATURES
QUANTILES = [0.10, 0.50, 0.90]


def get_active_features(df: pd.DataFrame) -> list[str]:
    """Returns active feature list based on available columns in DataFrame."""
    active = list(BASE_FEATURES)
    for pf in PRICE_FEATURES:
        if pf in df.columns and pf not in active:
            active.append(pf)
    return active


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["lag_24"] = df_feat["wind_mw"].shift(24)
    df_feat["wind_speed_lag_24"] = df_feat["wind_speed_ms"].shift(24)
    assert isinstance(df_feat.index, pd.DatetimeIndex)
    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    return df_feat.dropna()


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility alias for feature preparation."""
    return _prepare_features(df)


def compute_pinball_loss(y_true: np.ndarray, y_pred: Any, alpha: float) -> float:
    """Computes Pinball (Quantile) Loss for a given quantile alpha."""
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    err = y_true - pred_arr
    return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))


def train_quantile_models(df: pd.DataFrame) -> dict[str, float]:
    """
    Trains 3 separate LightGBM models for P10, P50, and P90 quantile forecasts.
    Returns average validation Pinball Loss dictionary.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_features(df)
    if prepared.empty:
        raise ValueError("DataFrame contains insufficient history for quantile training.")

    active_feats = get_active_features(prepared)
    X = prepared[active_feats]
    y = prepared["wind_mw"]

    tscv = TimeSeriesSplit(n_splits=5)
    quantile_metrics: dict[str, float] = {}

    for q in QUANTILES:
        q_label = f"p{int(q * 100)}"
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbosity": -1,
        }

        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            booster = lgb.train(params, ds_tr, num_boost_round=150)
            preds = booster.predict(X_va)

            p_loss = compute_pinball_loss(y_va.to_numpy(), preds, alpha=q)
            scores.append(p_loss)

        # Train final model on full dataset
        ds_full = lgb.Dataset(X, label=y)
        final_model = lgb.train(params, ds_full, num_boost_round=150)
        joblib.dump(final_model, MODEL_DIR / f"lgb_model_{q_label}.pkl")

        quantile_metrics[f"pinball_loss_{q_label}"] = float(np.mean(scores))

    return quantile_metrics


def train_model(df: pd.DataFrame) -> float:
    """Backward compatibility wrapper returning P50 validation score."""
    metrics = train_quantile_models(df)
    return metrics.get("pinball_loss_p50", 0.0)
