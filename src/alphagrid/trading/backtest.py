from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel


class BacktestMetrics(BaseModel):
    total_pnl_eur: float
    sharpe_ratio: float
    max_drawdown_pct: float
    var_95_eur: float
    profit_factor: float
    total_trades: int
    win_rate_pct: float


class BacktestResult(BaseModel):
    metrics: BacktestMetrics
    equity_curve: list[float]
    trade_log: list[dict[str, float | str]]


def run_backtest(
    df_features: pd.DataFrame,
    df_predictions: pd.DataFrame,
    initial_capital_eur: float = 100_000.0,
    max_capacity_mw: float = 10.0,
) -> BacktestResult:
    """
    Simulates Day-Ahead / Intraday energy trading & battery storage arbitrage strategy.

    Trading Strategy Logic:
    - Position Sizing is inversely proportional to uncertainty spread (P90 - P10).
    - If forecast generation > median threshold -> Go Short power / Charge storage.
    - If forecast generation < median threshold -> Go Long power / Discharge storage.
    """
    if df_predictions.empty:
        raise ValueError("df_predictions cannot be empty for backtesting.")

    p50 = df_predictions["p50"] if "p50" in df_predictions.columns else df_predictions["forecast"]
    p10 = df_predictions.get("p10", p50 * 0.9)
    p90 = df_predictions.get("p90", p50 * 1.1)

    prices = df_features.get("day_ahead_price_eur_mwh", pd.Series(60.0, index=df_predictions.index))

    # Reindex prices to align with forecast horizon, filling NaN with default price 60.0 EUR/MWh
    aligned_prices = prices.reindex(df_predictions.index).ffill().bfill().fillna(60.0)
    median_gen = float(p50.median()) if pd.notna(p50.median()) else 0.0

    capital = initial_capital_eur
    equity_curve = [capital]
    trade_log: list[dict[str, float | str]] = []
    hourly_pnls: list[float] = []

    for ts, gen in p50.items():
        raw_price = aligned_prices.loc[ts] if ts in aligned_prices.index else 60.0
        price = float(raw_price) if pd.notna(raw_price) else 60.0

        p10_val = float(p10.loc[ts]) if ts in p10.index and pd.notna(p10.loc[ts]) else gen * 0.9
        p90_val = float(p90.loc[ts]) if ts in p90.index and pd.notna(p90.loc[ts]) else gen * 1.1
        unc = max(0.0, p90_val - p10_val)

        # Position sizing scaled by inverse uncertainty
        size_mw = max_capacity_mw * (1.0 / (1.0 + unc / 2000.0))

        if float(gen) > median_gen:
            # Over-supply expected -> Short power position
            pnl = size_mw * (price - 50.0)  # Arbitrage vs baseline 50 EUR/MWh
            action = "SHORT"
        else:
            # Under-supply expected -> Long power position
            pnl = size_mw * (70.0 - price)
            action = "LONG"

        if np.isnan(pnl):
            pnl = 0.0

        capital += pnl
        equity_curve.append(capital)
        hourly_pnls.append(pnl)

        trade_log.append(
            {
                "timestamp": str(ts),
                "action": action,
                "size_mw": round(size_mw, 2),
                "price_eur_mwh": round(price, 2),
                "pnl_eur": round(pnl, 2),
                "capital_eur": round(capital, 2),
            }
        )

    # Compute financial metrics
    total_pnl = capital - initial_capital_eur
    pnls_arr = np.array(hourly_pnls)

    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr < 0]
    win_rate = float(len(wins) / len(pnls_arr) * 100.0) if len(pnls_arr) > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-6
    profit_factor = gross_profit / gross_loss

    # Sharpe Ratio (annualized hourly return)
    std_pnl = float(pnls_arr.std())
    sharpe = float((pnls_arr.mean() / std_pnl) * np.sqrt(8760)) if std_pnl > 0 else 0.0

    # Max Drawdown
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdown = np.where(peak > 0, (peak - eq_arr) / peak, 0.0)
    max_dd = float(drawdown.max() * 100.0) if len(drawdown) > 0 else 0.0

    # Value-at-Risk (95% confidence level)
    var_95 = float(np.percentile(pnls_arr, 5)) if len(pnls_arr) > 0 else 0.0

    metrics = BacktestMetrics(
        total_pnl_eur=round(total_pnl, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        var_95_eur=round(var_95, 2),
        profit_factor=round(profit_factor, 2),
        total_trades=len(trade_log),
        win_rate_pct=round(win_rate, 2),
    )

    return BacktestResult(
        metrics=metrics,
        equity_curve=[round(e, 2) for e in equity_curve],
        trade_log=trade_log,
    )
