import numpy as np
import pytest

from alphagrid.agents.guardrail import MarketThesis, cosine_similarity, validate_thesis
from alphagrid.agents.ingestion_agent import clean_html


def test_clean_html():
    raw_html = "<p>Wind power generation in <a>Germany</a> dropped by <br>1500 MW today.</p>"
    cleaned = clean_html(raw_html)
    assert cleaned == "Wind power generation in Germany dropped by 1500 MW today."


def test_cosine_similarity():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_similarity(vec_a, vec_b), 0.001) == 1.0

    vec_c = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(cosine_similarity(vec_a, vec_c), 0.001) == 0.0


def test_guardrail_validation_success():
    chunk = "A severe wind storm forced German wind turbines to curtail 2000 MW on 2026-01-05."
    thesis = MarketThesis(
        market_symbol="DE_LU",
        position_direction="Short",
        target_horizon_hours=24,
        verbatim_citations=["A severe wind storm forced German wind turbines to curtail 2000 MW"],
        reasoning="Wind storm forced German wind turbines to curtail 2000 MW on 2026-01-05.",
    )

    # Valid thesis with verbatim citation and identical reasoning text
    assert validate_thesis(thesis, [chunk], threshold=0.85) is True


def test_guardrail_validation_rejection_bad_citation():
    chunk = "A severe wind storm forced German wind turbines to curtail 2000 MW on 2026-01-05."
    thesis = MarketThesis(
        market_symbol="DE_LU",
        position_direction="Short",
        target_horizon_hours=24,
        verbatim_citations=["Fabricated citation not present in source text"],
        reasoning="Wind storm forced German wind turbines to curtail 2000 MW on 2026-01-05.",
    )

    # Rejects because citation is not a verbatim substring of source chunk
    assert validate_thesis(thesis, [chunk], threshold=0.85) is False


def test_guardrail_validation_rejection_low_similarity():
    chunk = "Solar panels in Bavaria produced peak output during midday sun."
    thesis = MarketThesis(
        market_symbol="DE_LU",
        position_direction="Short",
        target_horizon_hours=24,
        verbatim_citations=["Solar panels in Bavaria"],
        reasoning="Nuclear reactors in France were shut down due to coolant leak.",
    )

    # Rejects because cosine similarity between thesis reasoning and source chunk is low (< 0.85)
    assert validate_thesis(thesis, [chunk], threshold=0.85) is False
