from __future__ import annotations

import pandas as pd
import pytest

from alphagrid.data.feature_store import generate_synthetic
from alphagrid.trading.backtest import run_backtest


def test_run_backtest_execution():
    df_feat = generate_synthetic("2026-01-01", "2026-01-05")
    df_preds = pd.DataFrame(
        {
            "p10": [7000.0] * 24,
            "p50": [8000.0] * 24,
            "p90": [9000.0] * 24,
            "forecast": [8000.0] * 24,
        },
        index=pd.date_range("2026-01-01T00:00:00Z", periods=24, freq="h", tz="UTC"),
    )

    res = run_backtest(df_feat, df_preds, initial_capital_eur=100_000.0)

    assert res.metrics.total_trades == 24
    assert len(res.equity_curve) == 25
    assert res.metrics.sharpe_ratio is not None
    assert res.metrics.max_drawdown_pct >= 0.0
    assert 0.0 <= res.metrics.win_rate_pct <= 100.0


def test_run_backtest_empty_raises():
    df_feat = generate_synthetic("2026-01-01", "2026-01-02")
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError):
        run_backtest(df_feat, df_empty)
