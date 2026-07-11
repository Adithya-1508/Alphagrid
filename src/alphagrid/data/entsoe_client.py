from __future__ import annotations
import os
import pandas as pd
from entsoe import EntsoePandasClient
from .time_utils import to_utc
from ..config import load_config


def get_wind_generation(start, end) -> pd.DataFrame:
    cfg = load_config()
    token = os.getenv(cfg.get("entsoe_token_env", "ENTSOE_TOKEN"))
    if not token:
        raise RuntimeError("ENTSOE_TOKEN env var not set")
    client = EntsoePandasClient(api_key=token)
    start_utc, end_utc = to_utc(start), to_utc(end)
    zone = cfg.get("grid_zone", "DE_LU")
    df = client.query_generation(start=start_utc, end=end_utc, country_code=zone, psr_type="B16")
    if isinstance(df, pd.DataFrame):
        series = df.sum(axis=1) if df.shape[1] > 1 else df.iloc[:, 0]
    else:
        series = df
    out = pd.DataFrame({"wind_mw": series})
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC")
    return out
