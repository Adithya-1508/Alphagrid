from __future__ import annotations

import pandas as pd

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.data.market_prices import fetch_market_prices


def test_fetch_market_prices():
    prices = fetch_market_prices("2026-05-01T00:00:00Z", "2026-05-05T00:00:00Z")
    assert isinstance(prices, pd.DataFrame)
    assert not prices.empty
    assert set(["day_ahead_price_eur_mwh", "gas_price_eur_mwh", "carbon_price_eur_t"]).issubset(
        prices.columns
    )
    assert (prices["day_ahead_price_eur_mwh"] >= 0).all()


def test_synthetic_feature_store_includes_prices():
    df = generate_synthetic("2026-05-01T00:00:00Z", "2026-05-05T00:00:00Z")
    assert "day_ahead_price_eur_mwh" in df.columns
    assert "gas_price_eur_mwh" in df.columns
    assert "carbon_price_eur_t" in df.columns
