from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ..config import load_config


class MarketThesis(BaseModel):
    market_symbol: str = Field(description="Bidding zone or grid symbol, e.g. DE_LU")
    position_direction: Literal["Long", "Short"] = Field(description="Market position direction")
    target_horizon_hours: int = Field(default=24, description="Forecast horizon in hours")
    verbatim_citations: list[str] = Field(description="Literal citations from source text")
    reasoning: str = Field(description="Synthesized market rationale")


_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def validate_thesis(
    candidate: MarketThesis, source_chunks: list[str], threshold: float | None = None
) -> bool:
    """
    Validates a MarketThesis payload against source text chunks.

    Checks:
    1. Cosine similarity between reasoning and chunks >= threshold (default 0.85).
    2. Every verbatim_citation must be a literal substring of a source chunk.
    """
    if threshold is None:
        cfg = load_config()
        threshold = float(cfg.get("guardrail", {}).get("cosine_threshold", 0.85))

    if not source_chunks or not candidate.reasoning:
        return False

    # 1. Substring citation check
    for citation in candidate.verbatim_citations:
        if not citation:
            continue
        if not any(citation in chunk for chunk in source_chunks):
            return False

    # 2. Embedding Cosine Similarity check
    embedder = get_embedder()
    thesis_embedding = embedder.encode(candidate.reasoning, convert_to_numpy=True)
    chunk_embeddings = embedder.encode(source_chunks, convert_to_numpy=True)

    max_sim = 0.0
    for chunk_emb in chunk_embeddings:
        sim = cosine_similarity(thesis_embedding, chunk_emb)
        if sim > max_sim:
            max_sim = sim

    return max_sim >= threshold
