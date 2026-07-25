from __future__ import annotations

from typing import Any

from .guardrail import validate_thesis
from .synthesis_agent import synthesize_thesis


def process_anomaly_event(
    anomaly_event: dict[str, Any], threshold: float | None = None
) -> dict[str, Any]:
    """
    Orchestrates the agentic market-intel loop:
    1. Synthesizes a candidate MarketThesis from ChromaDB news chunks.
    2. Validates candidate against guardrail (schema + verbatim citations + cosine sim >= 0.85).
    3. Returns structured output indicating approval or rejection.
    """
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
