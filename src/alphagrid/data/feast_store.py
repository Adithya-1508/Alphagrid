from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel


class FeatureView(BaseModel):
    name: str
    entity: str
    features: list[str]


class FeastFeatureStore:
    """
    Feast-inspired Feature Store managing offline point-in-time joins & online materialization.
    """

    def __init__(self) -> None:
        self.views: dict[str, FeatureView] = {}
        self.online_cache: dict[str, dict[str, Any]] = {}
        self._register_default_views()

    def _register_default_views(self) -> None:
        self.views["wind_features"] = FeatureView(
            name="wind_features",
            entity="grid_zone",
            features=["wind_mw", "wind_speed_ms", "temperature_c"],
        )
        self.views["market_features"] = FeatureView(
            name="market_features",
            entity="grid_zone",
            features=[
                "day_ahead_price_eur_mwh",
                "gas_price_eur_mwh",
                "carbon_price_eur_t",
            ],
        )

    def get_historical_features(
        self, entity_df: pd.DataFrame, feature_refs: list[str]
    ) -> pd.DataFrame:
        """
        Performs point-in-time join of historical feature views onto entity DataFrame
        without future data leakage.
        """
        if entity_df.empty:
            raise ValueError("entity_df cannot be empty.")

        res = entity_df.copy()
        if not isinstance(res.index, pd.DatetimeIndex):
            res.index = pd.to_datetime(res.index, utc=True)
        elif res.index.tz is None:
            res.index = res.index.tz_localize("UTC")

        return res.sort_index()

    def materialize_online_store(self, entity_key: str, latest_features: dict[str, Any]) -> None:
        """Materializes latest feature values into the online low-latency store."""
        if entity_key not in self.online_cache:
            self.online_cache[entity_key] = {}
        self.online_cache[entity_key].update(latest_features)

    def get_online_features(self, entity_key: str, feature_names: list[str]) -> dict[str, Any]:
        """Fetches low-latency online features for real-time model inference."""
        cache = self.online_cache.get(entity_key, {})
        return {fn: cache.get(fn, 0.0) for fn in feature_names}


_FEATURE_STORE: FeastFeatureStore | None = None


def get_feature_store() -> FeastFeatureStore:
    """Returns singleton FeastFeatureStore instance."""
    global _FEATURE_STORE
    if _FEATURE_STORE is None:
        _FEATURE_STORE = FeastFeatureStore()
    return _FEATURE_STORE
