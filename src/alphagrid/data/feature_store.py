from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .time_utils import as_utc
from . import entsoe_client, weather_client

CACHE_DIR = Path("artifacts") / "raw_cache"


def _cache_path(source: str, start, end) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_DIR / f"{source}_DE_LU{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.parquet"
    )


def build_features(start, end, use_cache: bool = True) -> pd.DataFrame:
    start_utc, end_utc = as_utc(start), as_utc(end)
    wpath = _cache_path("wind", start_utc, end_utc)
    if use_cache and wpath.exists():
        wind = pd.read_parquet(wpath)
    else:
        wind = entsoe_client.get_wind_generation(start_utc, end_utc)
        wind.to_parquet(wpath)
    mpath = _cache_path("weather", start_utc, end_utc)
    if use_cache and mpath.exists():
        weather = pd.read_parquet(mpath)
    else:
        weather = weather_client.get_weather(start_utc, end_utc)
        weather.to_parquet(mpath)
    for name, df in (("wind", wind), ("weather", weather)):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{name} index must be a pd.DatetimeIndex")
        if df.index.tz is None:
            raise ValueError(f"{name} index is naive; UTC required")
    
    wind_hourly = wind.resample("1h").mean()
    weather_hourly = weather.resample("1h").mean()
    return wind_hourly.join(weather_hourly, how="inner").sort_index()


def generate_synthetic(start, end) -> pd.DataFrame:
    start_utc, end_utc = as_utc(start), as_utc(end)
    idx = pd.date_range(start_utc, end_utc, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(42)
    t = np.arange(n)
    wind_mw = 8000 + 4000 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 500, n)
    wind_speed_ms = 6 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    temperature_c = 10 + 8 * np.sin(2 * np.pi * t / 24 * 365) + rng.normal(0, 2, n)
    return pd.DataFrame(
        {"wind_mw": wind_mw, "wind_speed_ms": wind_speed_ms, "temperature_c": temperature_c},
        index=idx,
    )
