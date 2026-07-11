import pandas as pd
import pytest
from alphagrid.data.time_utils import to_utc
from alphagrid.data.feature_store import generate_synthetic


def test_utc_normalization():
    out = to_utc("2026-07-11T12:00:00+02:00")
    assert out == pd.Timestamp("2026-07-11T10:00:00+00:00"), out


def test_naive_rejected():
    with pytest.raises(ValueError):
        to_utc("2026-07-11T12:00:00")


def test_synthetic_schema():
    df = generate_synthetic("2026-01-01", "2026-01-03")
    assert list(df.columns) == ["wind_mw", "wind_speed_ms", "temperature_c"]
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
