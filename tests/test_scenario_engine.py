from __future__ import annotations

import pytest

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.scenario_engine import run_monte_carlo_scenarios


def test_monte_carlo_dunkelflaute_scenario():
    df = generate_synthetic("2026-01-01", "2026-01-05")
    res = run_monte_carlo_scenarios(
        df, scenario="Dunkelflaute", num_simulations=50, horizon_hours=24
    )

    assert res.scenario_name == "Dunkelflaute"
    assert res.num_simulations == 50
    assert len(res.simulated_paths) == 5
    assert len(res.simulated_paths[0]) == 24
    assert res.metrics.risk_level == "CRITICAL_DEFICIT"
    assert res.metrics.p05_deficit_mw <= res.metrics.p95_surplus_mw


def test_monte_carlo_empty_df_raises():
    import pandas as pd

    with pytest.raises(ValueError):
        run_monte_carlo_scenarios(pd.DataFrame())
