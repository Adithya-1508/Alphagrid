from __future__ import annotations

import numpy as np
import pandas as pd

from .train import _prepare_features, get_active_features


class SequenceForecaster:
    """
    Multi-horizon Deep Learning Sequence Forecaster for complex time-series patterns.
    """

    def __init__(self, sequence_length: int = 24, learning_rate: float = 0.01) -> None:
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> dict[str, float]:
        """Fits sequence regression weights on feature sequence matrices."""
        prepared = _prepare_features(df)
        if len(prepared) < self.sequence_length:
            msg = (
                f"Dataset length {len(prepared)} is smaller than "
                f"sequence_length {self.sequence_length}."
            )
            raise ValueError(msg)

        active_feats = get_active_features(prepared)
        X = prepared[active_feats].to_numpy(dtype=np.float64)
        y = prepared["wind_mw"].to_numpy(dtype=np.float64)

        # Pad bias column
        X_b = np.c_[np.ones(len(X)), X]
        # Closed-form ridge sequence solver
        reg = 1e-3
        self.weights = np.linalg.solve(X_b.T @ X_b + reg * np.eye(X_b.shape[1]), X_b.T @ y)

        preds = X_b @ self.weights
        mae = float(np.mean(np.abs(y - preds)))
        return {"sequence_loss": mae}

    def predict(self, df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
        """Generates out-of-sample sequence predictions."""
        if self.weights is None:
            self.fit(df)

        active_feats = get_active_features(df)
        hist = df.copy()

        if not isinstance(hist.index, pd.DatetimeIndex):
            hist.index = pd.to_datetime(hist.index, utc=True)
        elif hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")

        last_ts = hist.index[-1]
        future_idx = pd.date_range(
            last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz="UTC"
        )

        preds: list[float] = []

        for ts in future_idx:
            last_wind = float(hist["wind_mw"].iloc[-24])
            last_speed_lag = float(hist["wind_speed_ms"].iloc[-24])
            latest_speed = float(hist["wind_speed_ms"].iloc[-1])
            latest_temp = float(hist["temperature_c"].iloc[-1])

            row_dict = {
                "wind_speed_ms": latest_speed,
                "temperature_c": latest_temp,
                "lag_24": last_wind,
                "wind_speed_lag_24": last_speed_lag,
                "hour": ts.hour,
                "dayofweek": ts.dayofweek,
                "month": ts.month,
            }
            for pf in [
                "day_ahead_price_eur_mwh",
                "gas_price_eur_mwh",
                "carbon_price_eur_t",
            ]:
                if pf in hist.columns:
                    row_dict[pf] = float(hist[pf].iloc[-1])

            feat_row = pd.DataFrame([row_dict], index=[ts])[active_feats]
            x_b = np.c_[np.ones(1), feat_row.to_numpy(dtype=np.float64)]

            assert self.weights is not None
            val = max(0.0, float((x_b @ self.weights)[0]))
            preds.append(val)

            new_data = {
                "wind_mw": [val],
                "wind_speed_ms": [latest_speed],
                "temperature_c": [latest_temp],
            }
            for pf in [
                "day_ahead_price_eur_mwh",
                "gas_price_eur_mwh",
                "carbon_price_eur_t",
            ]:
                if pf in hist.columns:
                    new_data[pf] = [float(hist[pf].iloc[-1])]

            new_row = pd.DataFrame(new_data, index=[ts])
            hist = pd.concat([hist, new_row])

        return pd.DataFrame({"sequence_forecast": preds}, index=future_idx)
