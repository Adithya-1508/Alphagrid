import pandas as pd
import pytest
from alphagrid.data.time_utils import to_utc
from alphagrid.data.feature_store import generate_synthetic


from alphagrid.forecasting.train import build_feature_matrix


def test_utc_normalization():
    out = to_utc("2026-07-11T12:00:00+02:00")
    assert out == pd.Timestamp("2026-07-11T10:00:00+00:00"), out


def test_naive_rejected():
    with pytest.raises(ValueError):
        to_utc("2026-07-11T12:00:00")


def test_synthetic_schema():
    df = generate_synthetic("2026-01-01", "2026-01-03")
    assert list(df.columns) == ["wind_mw", "wind_speed_ms", "temperature_c"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None and str(df.index.tz) == "UTC"


def test_train():
    df = generate_synthetic("2026-01-01", "2026-01-03")
    out = build_feature_matrix(df)
    assert "hour" in out.columns
    assert "dayofweek" in out.columns
    assert "month" in out.columns


def test_weather_client_url_switching():
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2026-01-01T00:00:00Z"],
            "wind_speed_10m": [5.0],
            "temperature_2m": [15.0],
        }
    }

    with patch("requests.get", return_value=mock_response) as mock_get:
        from alphagrid.data.weather_client import get_weather

        # Test archive URL for a past date
        get_weather("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", lat=52.0, lon=13.0)
        assert mock_get.call_count == 1
        called_url = mock_get.call_args[0][0]
        assert "archive" in called_url
        assert mock_get.call_args[1]["params"]["latitude"] == 52.0
        assert mock_get.call_args[1]["params"]["longitude"] == 13.0

        mock_get.reset_mock()

        # Test forecast URL for a future date
        get_weather("2029-01-01T00:00:00Z", "2029-01-02T00:00:00Z")
        assert mock_get.call_count == 1
        called_url_future = mock_get.call_args[0][0]
        assert "forecast" in called_url_future


def test_entsoe_empty_handling():
    from unittest.mock import MagicMock, patch

    with patch("alphagrid.data.entsoe_client.EntsoePandasClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.query_generation.return_value = None
        mock_client_cls.return_value = mock_client

        from alphagrid.data.entsoe_client import get_wind_generation

        with patch.dict("os.environ", {"ENTSOE_TOKEN": "dummy_token"}):
            df = get_wind_generation("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
            assert df.empty
            assert list(df.columns) == ["wind_mw"]


def test_feature_store_resampling():
    from unittest.mock import patch
    from alphagrid.data.feature_store import build_features

    wind_idx = pd.date_range("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", freq="15min", tz="UTC")
    wind_df = pd.DataFrame({"wind_mw": range(len(wind_idx))}, index=wind_idx)

    weather_idx = pd.date_range("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", freq="1h", tz="UTC")
    weather_df = pd.DataFrame(
        {"wind_speed_ms": [5.0] * len(weather_idx), "temperature_c": [12.0] * len(weather_idx)},
        index=weather_idx,
    )

    with (
        patch("alphagrid.data.entsoe_client.get_wind_generation", return_value=wind_df),
        patch("alphagrid.data.weather_client.get_weather", return_value=weather_df),
    ):
        df = build_features("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", use_cache=False)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 4
        assert df.loc["2026-01-01T00:00:00Z", "wind_mw"] == 1.5
