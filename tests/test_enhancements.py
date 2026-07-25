from __future__ import annotations

import pandas as pd

from alphagrid.dashboard.pdf_report import generate_executive_pdf
from alphagrid.data.feature_store import generate_synthetic
from alphagrid.forecasting.tune import tune_hyperparameters


def test_synthetic_multi_zone():
    df = generate_synthetic("2026-05-01T00:00:00Z", "2026-05-05T00:00:00Z")
    assert not df.empty
    assert "wind_mw" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_optuna_tuning():
    df = generate_synthetic("2026-05-01T00:00:00Z", "2026-05-10T00:00:00Z")
    best_params, best_mae = tune_hyperparameters(df, n_trials=2)
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert best_mae >= 0.0


def test_pdf_report_generation():
    anomaly_event = {
        "market_symbol": "DE_LU",
        "date": "2026-05-15",
        "direction": "Shortage",
        "magnitude": -2500.0,
        "zscore": -2.8,
    }
    thesis_data = {
        "market_symbol": "DE_LU",
        "position_direction": "Long",
        "target_horizon_hours": 24,
        "reasoning": "Wind generation is severely curtailed due to offshore storm activity.",
        "verbatim_citations": ["Wind generation is severely curtailed"],
    }
    chunks = ["Wind generation is severely curtailed"]
    pdf_bytes = generate_executive_pdf(anomaly_event, thesis_data, source_chunks=chunks)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")
