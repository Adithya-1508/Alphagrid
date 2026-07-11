from __future__ import annotations
import pandas as pd
from datetime import datetime

UTC = "UTC"


def to_utc(ts) -> pd.Timestamp:
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    elif isinstance(ts, datetime):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"Naive timestamp forbidden: {ts}. Must be timezone-aware.")
    return ts.tz_convert(UTC)


def as_utc(ts) -> pd.Timestamp:
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    elif isinstance(ts, datetime):
        ts = pd.Timestamp(ts)
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def utc_range(start, end):
    return to_utc(start), to_utc(end)
