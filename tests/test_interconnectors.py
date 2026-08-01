from __future__ import annotations

import pandas as pd

from alphagrid.data.interconnectors import INTERCONNECTOR_REGISTRY, calculate_power_flows


def test_interconnector_registry():
    assert len(INTERCONNECTOR_REGISTRY) >= 4
    names = [ic.name for ic in INTERCONNECTOR_REGISTRY]
    assert "DE-FR" in names
    assert "DE-NL" in names


def test_calculate_power_flows():
    df = calculate_power_flows("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", zone="DE_LU")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(["net_flow_mw", "utilization_pct", "congestion_price_eur_mwh"]).issubset(df.columns)
    assert (df["utilization_pct"] <= 100.0).all()
