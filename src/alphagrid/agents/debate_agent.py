from __future__ import annotations

import json
from typing import Any

from ..config import load_config
from ..llm.ollama_client import generate_completion
from .guardrail import MarketThesis, validate_thesis
from .ingestion_agent import query_news_chunks


def _synthesize_directional_thesis(
    anomaly_event: dict[str, Any],
    source_chunks: list[str],
    perspective: str,
    position: str,
) -> MarketThesis | None:
    cfg = load_config()
    market_symbol = anomaly_event.get("market_symbol") or cfg.get("grid_zone", "DE_LU")
    date_str = anomaly_event.get("date", "")
    direction = anomaly_event.get("direction", "Shortage")

    chunks_text = "\n---\n".join(source_chunks)

    prompt = f"""
You are a senior energy market {perspective} trader.

Anomaly Event:
- Grid Zone: {market_symbol}
- Date: {date_str}
- Direction: {direction}
- Magnitude (MW): {anomaly_event.get("magnitude", 0)}

Source News Chunks:
{chunks_text}

Analyze the market from a strict {perspective} perspective (target position '{position}').
Output ONLY a valid JSON object with the following exact keys:
{{
  "market_symbol": "{market_symbol}",
  "position_direction": "{position}",
  "target_horizon_hours": 24,
  "verbatim_citations": [<exact verbatim substring quotes from source news chunks>],
  "reasoning": "<explanation of the {perspective} trade rationale>"
}}
"""

    system_prompt = f"You are an expert {perspective} energy trader. Respond ONLY with valid JSON."
    raw_response = generate_completion(prompt=prompt, system_prompt=system_prompt)

    try:
        cleaned_json = raw_response.strip()
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

        payload = json.loads(cleaned_json)
        return MarketThesis.model_validate(payload)
    except Exception:  # noqa: BLE001
        return None


def run_market_debate(
    anomaly_event: dict[str, Any], threshold: float | None = None
) -> dict[str, Any]:
    """
    Executes a Multi-Agent Debate between Bull Agent and Bear Agent.

    Returns:
        Dictionary containing Bull Thesis, Bear Thesis, validation results, and winning output.
    """
    date_str = anomaly_event.get("date", "")
    direction = anomaly_event.get("direction", "Shortage")

    query = f"German power grid wind generation {direction} {date_str}"
    source_chunks = query_news_chunks(query=query, n_results=3)

    if not source_chunks:
        source_chunks = query_news_chunks(query=f"German energy grid {direction}", n_results=3)

    if not source_chunks:
        return {
            "status": "REJECTED",
            "reason": "No relevant news chunks found in vector database.",
            "bull_thesis": None,
            "bear_thesis": None,
            "winning_thesis": None,
            "source_chunks": [],
        }

    bull_candidate = _synthesize_directional_thesis(
        anomaly_event, source_chunks, perspective="Bullish", position="Long"
    )
    bear_candidate = _synthesize_directional_thesis(
        anomaly_event, source_chunks, perspective="Bearish", position="Short"
    )

    bull_valid = (
        validate_thesis(bull_candidate, source_chunks, threshold=threshold)
        if bull_candidate
        else False
    )
    bear_valid = (
        validate_thesis(bear_candidate, source_chunks, threshold=threshold)
        if bear_candidate
        else False
    )

    # Determine winner based on grid direction and guardrail pass
    winning_thesis = None
    if direction == "Shortage" and bull_valid:
        winning_thesis = bull_candidate
    elif direction == "Surplus" and bear_valid:
        winning_thesis = bear_candidate
    elif bull_valid:
        winning_thesis = bull_candidate
    elif bear_valid:
        winning_thesis = bear_candidate

    status = "APPROVED" if winning_thesis is not None else "REJECTED"

    return {
        "status": status,
        "reason": (
            "Guardrail approved winning thesis."
            if status == "APPROVED"
            else "Neither debate candidate passed guardrail requirements."
        ),
        "bull_thesis": bull_candidate.model_dump() if bull_candidate else None,
        "bull_valid": bull_valid,
        "bear_thesis": bear_candidate.model_dump() if bear_candidate else None,
        "bear_valid": bear_valid,
        "winning_thesis": winning_thesis.model_dump() if winning_thesis else None,
        "source_chunks": source_chunks,
    }
