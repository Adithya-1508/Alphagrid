from __future__ import annotations
import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

MODEL_DIR = Path("artifacts") / "models"
TARGET = "wind_mw"
FEATURES = [
    "lag_24",
    "roll_mean_24",
    "roll_std_24",
    "roll_mean_168",
    "wind_speed_lag_24",
    "temperature_lag_24",
    "hour",
    "dayofweek",
    "month",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_24"] = df[TARGET].shift(24)
    df["roll_mean_24"] = df[TARGET].shift(1).rolling(24).mean()
    df["roll_std_24"] = df[TARGET].shift(1).rolling(24).std()
    df["roll_mean_168"] = df[TARGET].shift(1).rolling(168).mean()
    df["wind_speed_lag_24"] = df["wind_speed_ms"].shift(24)
    df["temperature_lag_24"] = df["temperature_c"].shift(24)
    idx = pd.DatetimeIndex(df.index)
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    return df


def train_model(df: pd.DataFrame) -> float:
    feat_df = build_feature_matrix(df).dropna()
    if len(feat_df) < 50:
        raise ValueError(
            f"Insufficient data to train. Need at least 50 samples, got {len(feat_df)}."
        )

    X = feat_df[FEATURES]
    y = feat_df[TARGET]

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    residuals = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
        residuals.extend(y_val - preds)

    avg_mae = sum(maes) / len(maes)
    residual_std = pd.Series(residuals).std()

    final_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(MODEL_DIR / "wind_model.txt"))

    metadata = {
        "residual_std": residual_std,
        "mae": avg_mae,
        "features": FEATURES,
        "target": TARGET,
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return avg_mae
