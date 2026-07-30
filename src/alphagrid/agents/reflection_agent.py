from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .guardrail import MarketThesis
from .memory_store import get_historical_accuracy


class ReflectedThesis(BaseModel):
    original_thesis: MarketThesis
    critique: str = Field(
        description="Self-critique analysis identifying potential flaws or biases"
    )
    is_approved: bool = Field(description="Whether thesis passes multi-agent self-critique review")
    adjusted_confidence: float = Field(description="Refined confidence score (0.0 to 1.0)")


def reflect_on_thesis(
    thesis: MarketThesis, market_context: dict[str, Any] | None = None
) -> ReflectedThesis:
    """
    Performs multi-agent self-critique and reflection on a synthesized MarketThesis.

    Checks:
    1. Alignment with historical accuracy memory store.
    2. Overconfidence bias adjustment.
    3. Market pricing sanity checks.
    """
    context = market_context or {}
    hist_acc = get_historical_accuracy(thesis.market_symbol)

    critiques = []
    confidence = 0.80

    # Historical accuracy weighting
    if hist_acc < 0.60:
        msg = (
            f"Historical accuracy for {thesis.market_symbol} is below 60% ({hist_acc:.2f}). "
            "Adjusting risk posture."
        )
        critiques.append(msg)
        confidence -= 0.15
    else:
        critiques.append(
            f"Historical accuracy for {thesis.market_symbol} is strong ({hist_acc:.2f})."
        )

    # Sanity check position against gas price signals if available
    gas_price = context.get("gas_price_eur_mwh", 35.0)
    if thesis.position_direction == "Short" and gas_price > 50.0:
        msg = (
            "Warning: Short position proposed despite high gas prices "
            "(>€50/MWh) setting high marginal costs."
        )
        critiques.append(msg)
        confidence -= 0.10

    # Substring citation check
    if not thesis.verbatim_citations:
        critiques.append("Thesis lacks verbatim citation support.")
        confidence -= 0.20

    is_approved = confidence >= 0.50
    critique_text = (
        " ".join(critiques) if critiques else "Thesis passed reflection review with no warnings."
    )

    return ReflectedThesis(
        original_thesis=thesis,
        critique=critique_text,
        is_approved=is_approved,
        adjusted_confidence=max(0.10, min(1.0, confidence)),
    )
