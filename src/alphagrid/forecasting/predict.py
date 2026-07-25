from __future__ import annotations
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from .train import build_feature_matrix, MODEL_DIR, FEATURES

def predict_next_hours(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Loads the trained model, constructs out-of-sample features,
    and returns point forecasts alongside a 90% confidence interval.
    """
    if df.empty:
        raise ValueError("Input DataFrame df cannot be empty.")
    if len(df) < 24:
        raise ValueError(f"Input DataFrame must contain at least 24 rows of historical data, got {len(df)}.")

    model_path = MODEL_DIR / "wind_model.txt"
    meta_path = MODEL_DIR / "model_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Trained model or metadata not found. Run train_model first.")

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    residual_std = metadata["residual_std"]

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)

    assert isinstance(df.index, pd.DatetimeIndex)
    last_dt = df.index[-1]
    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
        tz=df.index.tz
    )
    
    future_df = pd.DataFrame(index=future_index, columns=df.columns, dtype=float)
    combined_df = pd.concat([df, future_df])

    feat_df = build_feature_matrix(combined_df)
    future_feat = feat_df.loc[future_index]

    X_future = future_feat[FEATURES].astype(float)
    predictions = model.predict(X_future)

    lower_bound = np.clip(predictions - 1.645 * residual_std, 0, None)
    upper_bound = predictions + 1.645 * residual_std

    return pd.DataFrame(
        {
            "forecast": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        },
        index=future_index
    )
