from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = ["wind_speed_ms", "temperature_c", "lag_24", "wind_speed_lag_24", "hour", "month"]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["lag_24"] = df_feat["wind_mw"].shift(24)
    df_feat["wind_speed_lag_24"] = df_feat["wind_speed_ms"].shift(24)
    assert isinstance(df_feat.index, pd.DatetimeIndex)
    df_feat["hour"] = df_feat.index.hour
    df_feat["month"] = df_feat.index.month
    return df_feat.dropna()


def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 15) -> tuple[dict[str, Any], float]:
    """
    Automated hyperparameter optimization using Optuna & TimeSeriesSplit CV.

    Returns:
        (best_params_dict, best_mae_score)
    """
    prepared = _prepare_features(df)
    if prepared.empty:
        raise ValueError("DataFrame contains insufficient history after lag preparation.")

    X = prepared[FEATURES]
    y = prepared["wind_mw"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state": 42,
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            booster = lgb.train(params, ds_tr, num_boost_round=100)
            preds = booster.predict(X_va)
            mae = float(np.mean(np.abs(y_va - preds)))
            scores.append(mae)

        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_mae = study.best_value
    return best_params, best_mae
