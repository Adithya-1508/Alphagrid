from __future__ import annotations
import pandas as pd
import requests
from .time_utils import to_utc

LAT, LON = 50.0, 10.0


def get_weather(start, end) -> pd.DataFrame:
    start_utc, end_utc = to_utc(start), to_utc(end)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ["wind_speed-10m", "temperature_2m"],
        "start_date": start_utc.strftime("%Y-%m-%d"),
        "end_date": end_utc.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "time": data["time"],
            "wind_speed_ms": data["wind_speed_10m"],
            "temperature_c": data["temperature_2m"],
        }
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()
