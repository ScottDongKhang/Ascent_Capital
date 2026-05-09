# tests/test_factor_proposer.py
import pytest
import json
import numpy as np
from unittest.mock import patch


_CURRENT_WEIGHTS = {
    "trend": 0.41, "meanrev": 0.05, "volatility": 0.05, "statarb": 0.15,
    "ml": 0.10, "fundamental": 0.05, "earnings": 0.05, "analyst": 0.05,
    "options_flow": 0.02, "insider": 0.02, "short_interest": 0.02, "llm_fundamental": 0.03,
}


def test_propose_hypotheses_returns_list(tmp_path):
    from ascent.research.factor_proposer import propose_hypotheses
    mock_response = json.dumps([
        {"narrative": "In stressed regime, quality trumps momentum",
         "weight_biases": {"fundamental": 0.08, "trend": -0.08}},
        {"narrative": "Credit stress means low-vol names outperform",
         "weight_biases": {"volatility": 0.06, "trend": -0.06}},
    ])
    with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=2)
    assert isinstance(result, list)
    assert len(result) >= 1
    for h in result:
        assert "narrative" in h
        assert "weight_biases" in h


def test_duplicate_hypotheses_are_deduplicated():
    from ascent.research.factor_proposer import deduplicate_hypotheses
    h1 = {"narrative": "A", "weight_biases": {"fundamental": 0.10, "trend": -0.10}}
    h2 = {"narrative": "B", "weight_biases": {"fundamental": 0.10, "trend": -0.10}}  # identical biases
    h3 = {"narrative": "C", "weight_biases": {"volatility": 0.10, "trend": -0.10}}
    result = deduplicate_hypotheses([h1, h2, h3], similarity_threshold=0.85)
    assert len(result) < 3, "Identical bias vectors must be deduplicated"
    assert len(result) >= 2, "Genuinely different hypotheses must be kept"


def test_generate_guided_variants_sums_to_one():
    from ascent.research.factor_proposer import generate_guided_variants
    hypotheses = [
        {"narrative": "More quality", "weight_biases": {"fundamental": 0.10, "trend": -0.10}},
        {"narrative": "More vol",    "weight_biases": {"volatility": 0.08, "trend": -0.08}},
    ]
    variants = generate_guided_variants(_CURRENT_WEIGHTS, hypotheses, perturb_range=0.03)
    for v in variants:
        w = v["alpha_weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total}"


def test_generate_guided_variants_respects_floor():
    from ascent.research.factor_proposer import generate_guided_variants
    hypotheses = [
        {"narrative": "Zero out trend", "weight_biases": {"trend": -0.99, "fundamental": 0.50}},
    ]
    variants = generate_guided_variants(_CURRENT_WEIGHTS, hypotheses, perturb_range=0.03)
    for v in variants:
        w = v["alpha_weights"]
        assert w.get("trend", 0) >= 0.05, "Trend must never drop below minimum floor 5%"


def test_llm_failure_returns_empty_list():
    from ascent.research.factor_proposer import propose_hypotheses
    with patch("ascent.research.factor_proposer._call_llm", return_value=None):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=3)
    assert result == []


def test_generate_variants_uses_proposer_when_available():
    from ascent.research.self_improve import generate_variants
    import ascent.research.self_improve as si_mod
    mock_hyp = [
        {"narrative": "Quality over momentum", "weight_biases": {"fundamental": 0.08, "trend": -0.08}},
        {"narrative": "Vol regime favors stability", "weight_biases": {"volatility": 0.07, "trend": -0.07}},
        {"narrative": "Mean reversion in stress",   "weight_biases": {"meanrev": 0.06, "trend": -0.06}},
        {"narrative": "ML signals outperform",      "weight_biases": {"ml": 0.06, "trend": -0.06}},
        {"narrative": "Earnings drive alpha",       "weight_biases": {"earnings": 0.06, "trend": -0.06}},
    ]
    with patch("ascent.research.self_improve.SELF_MODIFY_ENABLED", True):
        with patch("ascent.research.factor_proposer.propose_hypotheses", return_value=mock_hyp):
            with patch("ascent.research.factor_proposer.generate_guided_variants",
                       wraps=lambda w, h, **kw: [{"variant_id": f"guided_{i}", "alpha_weights": w} for i, _ in enumerate(h)]):
                variants = generate_variants({"alpha_weights": _CURRENT_WEIGHTS}, n=5, regime="stressed")
    assert len(variants) == 5
    assert any(v["variant_id"].startswith("guided_") for v in variants), \
        "generate_variants must use guided proposer when regime is provided"
