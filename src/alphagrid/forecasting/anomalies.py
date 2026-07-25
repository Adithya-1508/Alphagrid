from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np


def detect_anomalies(
    actuals: pd.Series, forecasts: pd.Series, threshold: float = 2.0
) -> list[dict[str, Any]]:
    """
    Identifies anomaly days based on standard deviations of forecast residuals.

    Returns:
        List of dicts representing flagged anomaly days.
    """
    if not isinstance(actuals.index, pd.DatetimeIndex):
        try:
            actuals = actuals.copy()
            actuals.index = pd.DatetimeIndex(actuals.index)
        except Exception as e:
            raise TypeError(f"Could not convert actuals.index to DatetimeIndex: {e}") from e

    if not isinstance(forecasts.index, pd.DatetimeIndex):
        try:
            forecasts = forecasts.copy()
            forecasts.index = pd.DatetimeIndex(forecasts.index)
        except Exception as e:
            raise TypeError(f"Could not convert forecasts.index to DatetimeIndex: {e}") from e

    # Align series and clean NaNs
    df = pd.DataFrame({"actual": actuals, "forecast": forecasts}).dropna()
    if df.empty:
        return []

    df["residual"] = df["actual"] - df["forecast"]

    # Compute normalized z-scores of the residuals
    mean_res = df["residual"].mean()
    std_res = df["residual"].std()
    if std_res == 0 or np.isnan(std_res):
        std_res = 1e-8

    df["z_score"] = (df["residual"] - mean_res) / std_res
    df["abs_z"] = df["z_score"].abs()

    # Enforce type narrowing for Pyright/Pylance
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)

    assert isinstance(df.index, pd.DatetimeIndex)

    # Group residuals by day to identify "anomaly days"
    daily = df.groupby(df.index.date).agg(
        mean_residual=("residual", "mean"), max_abs_z=("abs_z", "max"), mean_z=("z_score", "mean")
    )

    anomalies: list[dict[str, Any]] = []
    for date, row in daily.iterrows():
        if row["max_abs_z"] >= threshold:
            direction = "Surplus" if row["mean_residual"] > 0 else "Shortage"
            anomalies.append(
                {
                    "date": str(date),
                    "direction": direction,
                    "magnitude": float(row["mean_residual"]),
                    "zscore": float(row["mean_z"]),
                }
            )

    return anomalies
