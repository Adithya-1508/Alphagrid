from __future__ import annotations

import os

import pandas as pd
from entsoe import EntsoePandasClient

from ..config import load_config
from .time_utils import to_utc


def get_wind_generation(start, end, zone: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    token = os.getenv(cfg.get("entsoe_token_env", "ENTSOE_TOKEN"))
    if not token:
        raise RuntimeError("ENTSOE_TOKEN env var not set")
    client = EntsoePandasClient(api_key=token)
    start_utc, end_utc = to_utc(start), to_utc(end)
    country_code = zone or cfg.get("grid_zone", "DE_LU")
    df = client.query_generation(
        start=start_utc, end=end_utc, country_code=country_code, psr_type="B16"
    )
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty):
        return pd.DataFrame(columns=["wind_mw"], index=pd.DatetimeIndex([], tz="UTC"))

    if isinstance(df, pd.DataFrame):
        series = df.sum(axis=1) if df.shape[1] > 1 else df.iloc[:, 0]
    else:
        series = df
    out = pd.DataFrame({"wind_mw": series})
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        out.index = idx.tz_localize("UTC")
    else:
        out.index = idx.tz_convert("UTC")
    return out
