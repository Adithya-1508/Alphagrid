from __future__ import annotations

import pytest

from alphagrid.agents.guardrail import MarketThesis
from alphagrid.agents.memory_store import get_historical_accuracy, log_thesis, update_thesis_outcome
from alphagrid.agents.reflection_agent import reflect_on_thesis


def test_memory_store_lifecycle(tmp_path, monkeypatch):
    db_file = tmp_path / "test_memory.db"
    monkeypatch.setattr("alphagrid.agents.memory_store.DB_PATH", db_file)

    tid = log_thesis("DE_LU", "Short", 24, "High wind generation forecast", 0.85)
    assert tid > 0

    update_thesis_outcome(tid, "Success", 0.90)
    acc = get_historical_accuracy("DE_LU")
    assert acc == pytest.approx(0.90, 0.01)


def test_reflection_agent_approval():
    thesis = MarketThesis(
        market_symbol="DE_LU",
        position_direction="Long",
        target_horizon_hours=24,
        verbatim_citations=["Severe storm curtailment"],
        reasoning="Curtailment reduces active capacity",
    )
    reflected = reflect_on_thesis(thesis, {"gas_price_eur_mwh": 30.0})
    assert reflected.is_approved is True
    assert reflected.adjusted_confidence >= 0.50
    assert "Historical accuracy" in reflected.critique
