from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from . import entsoe_client, weather_client
from .market_prices import fetch_market_prices
from .time_utils import as_utc

CACHE_DIR = Path("artifacts") / "raw_cache"


def _cache_path(source: str, zone: str, start, end) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    d_start = pd.Timestamp(start).date()
    d_end = pd.Timestamp(end).date()
    return CACHE_DIR / f"{source}_{zone}_{d_start}_{d_end}.parquet"


def build_features(start, end, zone: str = "DE_LU", use_cache: bool = True) -> pd.DataFrame:
    start_utc, end_utc = as_utc(start), as_utc(end)
    wpath = _cache_path("wind", zone, start_utc, end_utc)
    if use_cache and wpath.exists():
        wind = pd.read_parquet(wpath)
    else:
        wind = entsoe_client.get_wind_generation(start_utc, end_utc, zone=zone)
        wind.to_parquet(wpath)

    # Load coordinates for zone from config
    cfg = load_config()
    zone_info = cfg.get("grid_zones", {}).get(zone, {})
    lat = zone_info.get("latitude", 50.0)
    lon = zone_info.get("longitude", 10.0)

    mpath = _cache_path("weather", zone, start_utc, end_utc)
    if use_cache and mpath.exists():
        weather = pd.read_parquet(mpath)
    else:
        weather = weather_client.get_weather(start_utc, end_utc, lat=lat, lon=lon)
        weather.to_parquet(mpath)

    # Fetch market pricing series
    prices = fetch_market_prices(start_utc, end_utc)

    for name, df in (("wind", wind), ("weather", weather), ("prices", prices)):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{name} index must be a pd.DatetimeIndex")
        if df.index.tz is None:
            raise ValueError(f"{name} index is naive; UTC required")

    wind_hourly = wind.resample("1h").mean()
    weather_hourly = weather.resample("1h").mean()
    prices_hourly = prices.resample("1h").mean()

    combined = wind_hourly.join(weather_hourly, how="inner").join(prices_hourly, how="inner")
    return combined.sort_index()


def generate_synthetic(start, end) -> pd.DataFrame:
    start_utc, end_utc = as_utc(start), as_utc(end)
    idx = pd.date_range(start_utc, end_utc, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(42)
    t = np.arange(n)
    wind_mw = 8000 + 4000 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 500, n)
    wind_speed_ms = 6 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    temperature_c = 10 + 8 * np.sin(2 * np.pi * t / 24 * 365) + rng.normal(0, 2, n)

    prices = fetch_market_prices(start_utc, end_utc)

    df_base = pd.DataFrame(
        {"wind_mw": wind_mw, "wind_speed_ms": wind_speed_ms, "temperature_c": temperature_c},
        index=idx,
    )
    return df_base.join(prices, how="inner")
