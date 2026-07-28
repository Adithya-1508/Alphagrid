from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current distributions."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(ref, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    ref_counts = np.histogram(ref, buckets)[0]
    cur_counts = np.histogram(cur, buckets)[0]

    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts) / len(ref)
    cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts) / len(cur)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_val)


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Detects feature-level data drift using Kolmogorov-Smirnov (KS) test and PSI.

    Returns:
        Dictionary detailing overall drift status and feature metrics.
    """
    feature_reports = {}
    any_drift = False

    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue

        ref_vals = reference_df[feat].to_numpy()
        cur_vals = current_df[feat].to_numpy()

        # KS Test
        ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
        ks_drift = bool(p_val < ks_alpha)

        # PSI
        psi_score = calculate_psi(ref_vals, cur_vals)
        psi_drift = bool(psi_score >= psi_threshold)

        is_drifted = ks_drift or psi_drift
        if is_drifted:
            any_drift = True

        feature_reports[feat] = {
            "drift_detected": is_drifted,
            "ks_p_value": float(p_val),
            "psi_score": float(psi_score),
        }

    return {
        "drift_detected": any_drift,
        "feature_reports": feature_reports,
    }
