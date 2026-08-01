from __future__ import annotations

import joblib
import pandas as pd

from .train import MODEL_DIR, QUANTILES, get_active_features


def predict_quantile_forecasts(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Generates out-of-sample probabilistic forecasts (P10, P50, P90).
    """
    if df.empty:
        raise ValueError("Input DataFrame df cannot be empty.")

    models = {}
    for q in QUANTILES:
        q_label = f"p{int(q * 100)}"
        model_path = MODEL_DIR / f"lgb_model_{q_label}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model missing for {q_label}. Run train_quantile_models() first."
            )
        models[q_label] = joblib.load(model_path)

    # Convert non-DatetimeIndex if necessary
    hist = df.copy()
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index, utc=True)
    elif hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")

    active_feats = get_active_features(hist)
    last_ts = hist.index[-1]
    future_idx = pd.date_range(
        last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz="UTC"
    )

    preds_dict: dict[str, list[float]] = {"p10": [], "p50": [], "p90": []}

    for ts in future_idx:
        last_wind = hist["wind_mw"].iloc[-24]
        last_speed_lag = hist["wind_speed_ms"].iloc[-24]
        latest_speed = hist["wind_speed_ms"].iloc[-1]
        latest_temp = hist["temperature_c"].iloc[-1]

        row_dict = {
            "wind_speed_ms": latest_speed,
            "temperature_c": latest_temp,
            "lag_24": last_wind,
            "wind_speed_lag_24": last_speed_lag,
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "month": ts.month,
        }

        # Include market price features if present
        for pf in ["day_ahead_price_eur_mwh", "gas_price_eur_mwh", "carbon_price_eur_t"]:
            if pf in hist.columns:
                row_dict[pf] = hist[pf].iloc[-1]

        feat_row = pd.DataFrame([row_dict], index=[ts])[active_feats]

        p10_val = max(0.0, float(models["p10"].predict(feat_row)[0]))
        p50_val = max(0.0, float(models["p50"].predict(feat_row)[0]))
        p90_val = max(0.0, float(models["p90"].predict(feat_row)[0]))

        # Non-crossing quantile adjustment
        p10_clean = min(p10_val, p50_val)
        p90_clean = max(p90_val, p50_val)

        preds_dict["p10"].append(p10_clean)
        preds_dict["p50"].append(p50_val)
        preds_dict["p90"].append(p90_clean)

        new_data = {
            "wind_mw": [p50_val],
            "wind_speed_ms": [latest_speed],
            "temperature_c": [latest_temp],
        }
        for pf in ["day_ahead_price_eur_mwh", "gas_price_eur_mwh", "carbon_price_eur_t"]:
            if pf in hist.columns:
                new_data[pf] = [hist[pf].iloc[-1]]

        new_row = pd.DataFrame(new_data, index=[ts])
        hist = pd.concat([hist, new_row])

    out = pd.DataFrame(preds_dict, index=future_idx)
    # Backward-compatible column aliases
    out["forecast"] = out["p50"]
    out["lower_bound"] = out["p10"]
    out["upper_bound"] = out["p90"]
    return out


def predict_next_hours(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Wrapper function maintaining compatibility with existing Dashboard API.
    """
    return predict_quantile_forecasts(df, horizon_hours=horizon_hours)
