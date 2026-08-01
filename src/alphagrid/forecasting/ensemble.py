from __future__ import annotations

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .train import MODEL_DIR, _prepare_features, get_active_features


def train_ensemble_models(df: pd.DataFrame) -> dict[str, float]:
    """
    Trains an ensemble of LightGBM, XGBoost, and CatBoost regressors,
    and fits a Ridge Meta-Learner on out-of-fold predictions to blend outputs.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_features(df)
    if prepared.empty:
        raise ValueError("DataFrame contains insufficient history for ensemble training.")

    active_feats = get_active_features(prepared)
    X = prepared[active_feats]
    y = prepared["wind_mw"]

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(X), 3))

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        m_lgb = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_estimators=100)
        m_xgb = xgb.XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
        m_cb = cb.CatBoostRegressor(random_state=42, verbose=0, iterations=100)

        m_lgb.fit(X_tr, y_tr)
        m_xgb.fit(X_tr, y_tr)
        m_cb.fit(X_tr, y_tr)

        oof_preds[val_idx, 0] = m_lgb.predict(X_va)
        oof_preds[val_idx, 1] = m_xgb.predict(X_va)
        oof_preds[val_idx, 2] = m_cb.predict(X_va)

    # Train final base models on full dataset
    final_lgb = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_estimators=100)
    final_xgb = xgb.XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
    final_cb = cb.CatBoostRegressor(random_state=42, verbose=0, iterations=100)

    final_lgb.fit(X, y)
    final_xgb.fit(X, y)
    final_cb.fit(X, y)

    # Fit Ridge meta-learner on valid OOF indices
    val_indices = np.where(oof_preds.sum(axis=1) != 0)[0]
    meta_learner = Ridge(alpha=1.0, positive=True)
    meta_learner.fit(oof_preds[val_indices], y.iloc[val_indices])

    joblib.dump(final_lgb, MODEL_DIR / "ensemble_lgb.pkl")
    joblib.dump(final_xgb, MODEL_DIR / "ensemble_xgb.pkl")
    joblib.dump(final_cb, MODEL_DIR / "ensemble_cb.pkl")
    joblib.dump(meta_learner, MODEL_DIR / "ensemble_meta.pkl")

    oof_blend = meta_learner.predict(oof_preds[val_indices])
    blend_mae = float(np.mean(np.abs(y.iloc[val_indices] - oof_blend)))

    return {"ensemble_mae": blend_mae}


def predict_ensemble(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Generates blended predictions from LightGBM, XGBoost, and CatBoost models.
    """
    lgb_path = MODEL_DIR / "ensemble_lgb.pkl"
    xgb_path = MODEL_DIR / "ensemble_xgb.pkl"
    cb_path = MODEL_DIR / "ensemble_cb.pkl"
    meta_path = MODEL_DIR / "ensemble_meta.pkl"

    if not (lgb_path.exists() and xgb_path.exists() and cb_path.exists() and meta_path.exists()):
        train_ensemble_models(df)

    m_lgb = joblib.load(lgb_path)
    m_xgb = joblib.load(xgb_path)
    m_cb = joblib.load(cb_path)
    meta = joblib.load(meta_path)

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

    preds: list[float] = []

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
        for pf in ["day_ahead_price_eur_mwh", "gas_price_eur_mwh", "carbon_price_eur_t"]:
            if pf in hist.columns:
                row_dict[pf] = hist[pf].iloc[-1]

        feat_row = pd.DataFrame([row_dict], index=[ts])[active_feats]

        p_lgb = float(m_lgb.predict(feat_row)[0])
        p_xgb = float(m_xgb.predict(feat_row)[0])
        p_cb = float(m_cb.predict(feat_row)[0])

        base_preds = np.array([[p_lgb, p_xgb, p_cb]])
        blend_val = max(0.0, float(meta.predict(base_preds)[0]))
        preds.append(blend_val)

        new_data = {
            "wind_mw": [blend_val],
            "wind_speed_ms": [latest_speed],
            "temperature_c": [latest_temp],
        }
        for pf in ["day_ahead_price_eur_mwh", "gas_price_eur_mwh", "carbon_price_eur_t"]:
            if pf in hist.columns:
                new_data[pf] = [hist[pf].iloc[-1]]

        new_row = pd.DataFrame(new_data, index=[ts])
        hist = pd.concat([hist, new_row])

    return pd.DataFrame({"forecast": preds}, index=future_idx)
