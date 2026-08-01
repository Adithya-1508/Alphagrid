from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_conformal_margin(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, alpha: float = 0.10
) -> float:
    """
    Calculates the conformal non-conformity margin for a target coverage level (1 - alpha).

    Args:
        y_true: True historical ground truth targets from calibration set.
        y_pred: Model predictions on calibration set.
        alpha: Miscoverage probability (default 0.10 for 90% coverage).

    Returns:
        q_margin: Empirical quantile margin scalar.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0 or len(y_t) != len(y_p):
        raise ValueError("Inputs y_true and y_pred must be non-empty and equal length.")

    # Absolute non-conformity residuals
    residuals = np.abs(y_t - y_p)
    n = len(residuals)

    # Conformal finite-sample correction quantile level: (ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    q_margin = float(np.quantile(residuals, q_level))

    return q_margin


def predict_conformal_bounds(
    y_pred: np.ndarray | pd.Series, q_margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies calibrated conformal margin to produce lower and upper prediction bounds.

    Returns:
        (lower_bounds, upper_bounds) numpy arrays clipped to non-negative power generation.
    """
    preds = np.asarray(y_pred, dtype=np.float64)
    lower = np.clip(preds - q_margin, 0.0, None)
    upper = preds + q_margin
    return lower, upper
