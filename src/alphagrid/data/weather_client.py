from __future__ import annotations

import pandas as pd
import requests

from ..config import load_config
from .time_utils import to_utc


def get_weather(start, end, lat: float | None = None, lon: float | None = None) -> pd.DataFrame:
    start_utc, end_utc = to_utc(start), to_utc(end)

    if lat is None or lon is None:
        try:
            cfg = load_config()
            weather_cfg = cfg.get("weather", {})
            default_lat = weather_cfg.get("latitude", 50.0)
            default_lon = weather_cfg.get("longitude", 10.0)
        except Exception:  # noqa: BLE001
            default_lat, default_lon = 50.0, 10.0

        if lat is None:
            lat = default_lat
        if lon is None:
            lon = default_lon

    current_time = pd.Timestamp.now(tz="UTC")
    # Archive data has a ~2 day lag. Use forecast API if end_utc is newer than 2 days ago.
    if end_utc > current_time - pd.Timedelta(days=2):
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_10m", "temperature_2m"],
        "start_date": start_utc.strftime("%Y-%m-%d"),
        "end_date": end_utc.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    resp_data = resp.json()
    if "hourly" not in resp_data:
        raise KeyError(f"Expected 'hourly' key in Open-Meteo response, got: {resp_data}")
    data = resp_data["hourly"]
    df = pd.DataFrame(
        {
            "time": data["time"],
            "wind_speed_ms": data["wind_speed_10m"],
            "temperature_c": data["temperature_2m"],
        }
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df
