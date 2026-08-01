from __future__ import annotations

import numpy as np
import pytest

from alphagrid.forecasting.conformal import calibrate_conformal_margin, predict_conformal_bounds


def test_conformal_calibration_and_coverage():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(5000.0, 9000.0, size=500)
    y_pred = y_true + rng.normal(0.0, 200.0, size=500)

    # 90% coverage target (alpha = 0.10)
    margin = calibrate_conformal_margin(y_true, y_pred, alpha=0.10)
    assert margin > 0.0

    lower, upper = predict_conformal_bounds(y_pred, margin)
    covered = (y_true >= lower) & (y_true <= upper)
    empirical_coverage = float(np.mean(covered))

    # Empirical coverage must be at least 88%
    assert empirical_coverage >= 0.88
    assert (lower >= 0.0).all()


def test_conformal_invalid_inputs_raise():
    with pytest.raises(ValueError):
        calibrate_conformal_margin(np.array([]), np.array([]))
