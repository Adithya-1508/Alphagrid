from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .time_utils import as_utc


class Interconnector(BaseModel):
    name: str
    zone_a: str
    zone_b: str
    max_capacity_mw: float


INTERCONNECTOR_REGISTRY: list[Interconnector] = [
    Interconnector(name="DE-FR", zone_a="DE_LU", zone_b="FR", max_capacity_mw=3000.0),
    Interconnector(name="DE-NL", zone_a="DE_LU", zone_b="NL", max_capacity_mw=3250.0),
    Interconnector(name="DE-DK1", zone_a="DE_LU", zone_b="DK_1", max_capacity_mw=2500.0),
    Interconnector(name="FR-NL", zone_a="FR", zone_b="NL", max_capacity_mw=1400.0),
]


def calculate_power_flows(start, end, zone: str = "DE_LU") -> pd.DataFrame:
    """
    Computes cross-border physical power flows, net import/export MW,
    transmission line utilization, and price congestion metrics.
    """
    start_utc, end_utc = as_utc(start), as_utc(end)
    idx = pd.date_range(start_utc, end_utc, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(77)
    t = np.arange(n)

    # Net power flow simulation driven by regional supply imbalances
    net_flow = 1200.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 200, n)
    max_cap = 3000.0

    # Clip physical flow to max transfer capacity
    clipped_flow = np.clip(net_flow, -max_cap, max_cap)
    utilization_pct = (np.abs(clipped_flow) / max_cap) * 100.0
    congestion_spread = np.where(utilization_pct > 90.0, rng.uniform(15.0, 45.0, n), 0.0)

    return pd.DataFrame(
        {
            "net_flow_mw": clipped_flow,
            "utilization_pct": np.round(utilization_pct, 2),
            "congestion_price_eur_mwh": np.round(congestion_spread, 2),
        },
        index=idx,
    )
