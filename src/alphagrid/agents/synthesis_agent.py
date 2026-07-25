from __future__ import annotations

import json
from typing import Any

from ..config import load_config
from ..llm.ollama_client import generate_completion
from .guardrail import MarketThesis
from .ingestion_agent import query_news_chunks


def synthesize_thesis(anomaly_event: dict[str, Any]) -> tuple[MarketThesis | None, list[str]]:
    """
    Retrieves relevant news chunks for an anomaly event and uses Ollama
    to generate a candidate MarketThesis JSON object.

    Returns:
        (Candidate MarketThesis or None, list of retrieved source text chunks)
    """
    cfg = load_config()
    market_symbol = cfg.get("grid_zone", "DE_LU")
    date_str = anomaly_event.get("date", "")
    direction = anomaly_event.get("direction", "Shortage")

    query = f"German power grid wind generation {direction} {date_str}"
    source_chunks = query_news_chunks(query=query, n_results=3)

    if not source_chunks:
        # Fallback to general market query if date-specific chunks are empty
        source_chunks = query_news_chunks(query=f"German wind grid {direction}", n_results=3)

    if not source_chunks:
        return None, []

    chunks_text = "\n---\n".join(source_chunks)

    prompt = f"""
Given the anomaly event and market news, generate a JSON object representing a market thesis.

Anomaly Event:
- Grid Zone: {market_symbol}
- Date: {date_str}
- Direction: {direction}
- Magnitude (MW): {anomaly_event.get("magnitude", 0)}

Source News Chunks:
{chunks_text}

Output ONLY a valid JSON object with the following exact keys:
{{
  "market_symbol": "{market_symbol}",
  "position_direction": "Long" or "Short",
  "target_horizon_hours": 24,
  "verbatim_citations": [<exact verbatim substring quotes from the source news chunks>],
  "reasoning": "<explanation of the trade rationale based on source chunks>"
}}
"""

    system_prompt = "You are an expert energy market trader. Respond ONLY with a valid JSON object."

    raw_response = generate_completion(prompt=prompt, system_prompt=system_prompt)

    try:
        # Extract JSON from code fences if present
        cleaned_json = raw_response.strip()
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

        payload = json.loads(cleaned_json)
        candidate = MarketThesis.model_validate(payload)
        return candidate, source_chunks
    except Exception:  # noqa: BLE001
        return None, source_chunks
