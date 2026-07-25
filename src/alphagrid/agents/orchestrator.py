from __future__ import annotations

from typing import Any

from .debate_agent import run_market_debate
from .guardrail import validate_thesis
from .synthesis_agent import synthesize_thesis


def process_anomaly_event(
    anomaly_event: dict[str, Any],
    threshold: float | None = None,
    use_debate: bool = True,
) -> dict[str, Any]:
    """
    Orchestrates the agentic market-intel loop:
    1. Runs Multi-Agent Bull vs. Bear Debate (or single candidate synthesis).
    2. Validates candidate against guardrail (schema + verbatim citations + cosine sim >= 0.85).
    3. Returns structured output indicating approval or rejection.
    """
    if use_debate:
        debate_res = run_market_debate(anomaly_event, threshold=threshold)
        return {
            "status": debate_res["status"],
            "reason": debate_res["reason"],
            "thesis": debate_res["winning_thesis"],
            "bull_thesis": debate_res["bull_thesis"],
            "bull_valid": debate_res.get("bull_valid", False),
            "bear_thesis": debate_res["bear_thesis"],
            "bear_valid": debate_res.get("bear_valid", False),
            "source_chunks": debate_res["source_chunks"],
        }

    candidate, source_chunks = synthesize_thesis(anomaly_event)

    if candidate is None:
        return {
            "status": "REJECTED",
            "reason": "Failed candidate synthesis or invalid JSON payload.",
            "thesis": None,
            "source_chunks": source_chunks,
        }

    is_valid = validate_thesis(candidate, source_chunks, threshold=threshold)

    if is_valid:
        return {
            "status": "APPROVED",
            "reason": "Guardrail passed: citations verified and cosine similarity >= threshold.",
            "thesis": candidate.model_dump(),
            "source_chunks": source_chunks,
        }
    else:
        return {
            "status": "REJECTED",
            "reason": "Guardrail failed: cosine similarity below threshold or citation unverified.",
            "thesis": candidate.model_dump(),
            "source_chunks": source_chunks,
        }
