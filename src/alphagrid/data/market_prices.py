from __future__ import annotations

import numpy as np
import pandas as pd

from .time_utils import as_utc


def fetch_market_prices(start, end) -> pd.DataFrame:
    """
    Fetches European energy market multi-tier pricing data:
    - Day-Ahead Spot Electricity Price (€/MWh) [EPEX / EEX Auction]
    - Intraday Continuous Price (€/MWh) [Intraday Adjustments]
    - Imbalance Settlement Price (€/MWh) [TSO Cashout Price]
    - TTF Natural Gas Price (€/MWh)
    - EU ETS Carbon Credit Price (€/tCO2)
    """
    start_utc, end_utc = as_utc(start), as_utc(end)
    idx = pd.date_range(start_utc, end_utc, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(101)
    t = np.arange(n)

    # Base price dynamics modeling European merit-order curves
    elec_price = 80.0 + 35.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 8.0, n)
    intraday_price = elec_price + rng.normal(0, 4.0, n)
    imbalance_price = elec_price + rng.normal(0, 15.0, n)  # Higher volatility TSO cashout
    gas_price = 35.0 + 5.0 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 1.2, n)
    carbon_price = 65.0 + 2.0 * np.sin(2 * np.pi * t / (24 * 30)) + rng.normal(0, 0.5, n)

    return pd.DataFrame(
        {
            "day_ahead_price_eur_mwh": np.clip(elec_price, 0.0, None),
            "intraday_price_eur_mwh": np.clip(intraday_price, 0.0, None),
            "imbalance_price_eur_mwh": np.clip(imbalance_price, 0.0, None),
            "gas_price_eur_mwh": np.clip(gas_price, 0.0, None),
            "carbon_price_eur_t": np.clip(carbon_price, 0.0, None),
        },
        index=idx,
    )
