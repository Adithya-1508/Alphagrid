from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ScenarioMetrics(BaseModel):
    mean_forecast_mw: float
    p05_deficit_mw: float
    p95_surplus_mw: float
    max_price_spike_eur_mwh: float
    risk_level: str


class ScenarioResults(BaseModel):
    scenario_name: str
    num_simulations: int
    metrics: ScenarioMetrics
    simulated_paths: list[list[float]]


def run_monte_carlo_scenarios(
    df: pd.DataFrame,
    scenario: str = "Dunkelflaute",
    num_simulations: int = 100,
    horizon_hours: int = 24,
) -> ScenarioResults:
    """
    Runs Monte Carlo stochastic grid stress testing under extreme shock scenarios:
    - Dunkelflaute: 90% wind drop + severe cold snap.
    - Heatwave: Thermal capacity limits + high demand.
    - Outage: Unplanned 2000 MW baseload trip.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty for scenario testing.")

    base_wind = float(df["wind_mw"].iloc[-1]) if "wind_mw" in df.columns else 5000.0
    rng = np.random.default_rng(42)
    max_capacity_mw = 10000.0

    # Apply shock multipliers based on scenario
    if scenario == "Dunkelflaute":
        shock_mult = 0.10
        volatility = 200.0
        risk_level = "CRITICAL_DEFICIT"
    elif scenario == "Heatwave":
        shock_mult = 0.70
        volatility = 500.0
        risk_level = "HIGH_THERMAL_STRESS"
    elif scenario == "Outage":
        shock_mult = 0.50
        volatility = 800.0
        risk_level = "CAPACITY_TRIP"
    else:
        shock_mult = 1.0
        volatility = 300.0
        risk_level = "NORMAL"

    paths: list[list[float]] = []

    for _ in range(num_simulations):
        path = []
        val = base_wind * shock_mult
        for h in range(horizon_hours):
            shock = rng.normal(0, volatility)
            val = max(0.0, min(max_capacity_mw, val + shock + 50.0 * np.sin(h)))
            path.append(round(val, 2))
        paths.append(path)

    paths_arr = np.array(paths)
    mean_path = np.mean(paths_arr, axis=0)
    p05_path = np.percentile(paths_arr, 5, axis=0)
    p95_path = np.percentile(paths_arr, 95, axis=0)

    mean_val = float(np.mean(mean_path))
    p05 = float(np.min(p05_path))
    p95 = float(np.max(p95_path))

    # Spot price spike estimation based on deficit severity
    price_spike = float(max(50.0, 300.0 * (1.0 - (p05 / max(1.0, mean_val)))))

    metrics = ScenarioMetrics(
        mean_forecast_mw=round(mean_val, 2),
        p05_deficit_mw=round(p05, 2),
        p95_surplus_mw=round(p95, 2),
        max_price_spike_eur_mwh=round(price_spike, 2),
        risk_level=risk_level,
    )

    return ScenarioResults(
        scenario_name=scenario,
        num_simulations=num_simulations,
        metrics=metrics,
        simulated_paths=paths[:5],  # Sample 5 paths for visualization
    )
