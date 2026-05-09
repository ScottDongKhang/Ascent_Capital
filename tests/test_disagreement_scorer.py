# tests/test_disagreement_scorer.py
import pytest
import json
from pathlib import Path
from unittest.mock import patch


def test_identical_texts_score_zero_disagreement():
    from debate.disagreement_scorer import compute_disagreement_score
    text = "The portfolio is overweight technology with high momentum and strong fundamentals."
    score = compute_disagreement_score(text, text, text)
    assert score < 0.05, f"Identical texts should score near 0 disagreement, got {score:.4f}"


def test_completely_different_texts_score_high_disagreement():
    from debate.disagreement_scorer import compute_disagreement_score
    bull  = "Strong momentum signals across technology and industrials. Earnings beats confirm the bull thesis. Proceed."
    bear  = "Credit spreads widening. Concentration risk in cyclicals. VaR estimate elevated. Reduce position size immediately."
    devil = "Regime entropy approaching threshold. Correlation matrix suggests macro shock exposure. Monte Carlo p5 is alarming."
    score = compute_disagreement_score(bull, bear, devil)
    assert score > 0.40, f"Substantively different texts should score >0.40 disagreement, got {score:.4f}"


def test_returns_float_between_zero_and_one():
    from debate.disagreement_scorer import compute_disagreement_score
    score = compute_disagreement_score("alpha beta gamma", "delta epsilon zeta", "eta theta iota")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Score must be in [0, 1], got {score}"


def test_pairwise_computation_correct():
    from debate.disagreement_scorer import compute_disagreement_score, pairwise_similarities
    bull  = "momentum quality earnings growth"
    bear  = "risk drawdown concentration volatility"
    devil = "entropy correlation regime uncertainty"
    sims  = pairwise_similarities(bull, bear, devil)
    assert isinstance(sims, dict)
    for key in ("bull_bear", "bull_devil", "bear_devil"):
        assert key in sims, f"Missing key: {key}"
        assert 0.0 <= sims[key] <= 1.0, f"{key} similarity out of range: {sims[key]}"
    expected_score = 1.0 - (sims["bull_bear"] + sims["bull_devil"] + sims["bear_devil"]) / 3.0
    score = compute_disagreement_score(bull, bear, devil)
    assert abs(score - expected_score) < 1e-6, "Score must equal 1 - mean(pairwise similarities)"


def test_verdict_json_includes_disagreement_score(tmp_path):
    verdict_path = tmp_path / "verdict_2026-05-08.json"
    verdict_data = {
        "date": "2026-05-08",
        "verdict": {"recommendation": "proceed", "reasoning": "test"},
        "disagreement_score": 0.62,
        "pairwise_similarities": {"bull_bear": 0.30, "bull_devil": 0.45, "bear_devil": 0.38},
    }
    verdict_path.write_text(json.dumps(verdict_data))
    loaded = json.loads(verdict_path.read_text())
    assert "disagreement_score" in loaded, "verdict JSON must include disagreement_score"
    assert isinstance(loaded["disagreement_score"], float)
    assert "pairwise_similarities" in loaded


def test_judge_prompt_includes_score():
    from debate.disagreement_scorer import format_disagreement_for_judge
    prompt_fragment = format_disagreement_for_judge(disagreement_score=0.72)
    assert "0.28" in prompt_fragment or "0.72" in prompt_fragment, \
        "Judge note must include the similarity or disagreement value"
    assert "consensus" in prompt_fragment.lower() or "similar" in prompt_fragment.lower(), \
        "Judge note must mention consensus or similarity"
    # Must NOT instruct the judge to change verdict direction — informational only
    assert "reduce_size" not in prompt_fragment, \
        "Judge note must not prescribe a verdict — it is observational context only"
    assert "halt_and_review" not in prompt_fragment, \
        "Judge note must not prescribe a verdict — it is observational context only"


def test_empty_or_short_traces_handled_gracefully():
    from debate.disagreement_scorer import compute_disagreement_score
    # Should not raise — degenerate inputs return 0.0 (no information)
    score = compute_disagreement_score("", "", "")
    assert score == 0.0
    score = compute_disagreement_score("a", "b", "c")
    assert 0.0 <= score <= 1.0
