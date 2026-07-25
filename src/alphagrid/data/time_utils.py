from __future__ import annotations

from datetime import datetime

import pandas as pd

UTC = "UTC"


def to_utc(ts) -> pd.Timestamp:
    if isinstance(ts, (str, datetime)):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"Naive timestamp forbidden: {ts}. Must be timezone-aware.")
    return ts.tz_convert(UTC)


def as_utc(ts) -> pd.Timestamp:
    if isinstance(ts, (str, datetime)):
        ts = pd.Timestamp(ts)
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def utc_range(start, end):
    return to_utc(start), to_utc(end)
