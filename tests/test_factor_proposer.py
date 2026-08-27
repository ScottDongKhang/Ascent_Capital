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

_BIG_UNIVERSE = [f"SYM{i}" for i in range(15)]


def _full_hypothesis(**overrides):
    """A minimally valid hypothesis under the widened schema."""
    h = {
        "signal_id": "quality_stress",
        "thesis": "In stressed regime, quality trumps momentum",
        "universe": _BIG_UNIVERSE,
        "expected_ic_sign": 1,
        "uncorrelation_rationale": "Fundamentals-driven, monthly rebalance -- orthogonal to "
                                    "meanrev's short-horizon price reversal and statarb's "
                                    "cross-sectional pairs mechanism.",
        "weight_biases": {"fundamental": 0.08, "trend": -0.08},
    }
    h.update(overrides)
    return h


def test_propose_hypotheses_returns_list(tmp_path):
    from ascent.research.factor_proposer import propose_hypotheses
    mock_response = json.dumps([
        _full_hypothesis(signal_id="quality_stress"),
        _full_hypothesis(
            signal_id="lowvol_credit",
            thesis="Credit stress means low-vol names outperform",
            weight_biases={"volatility": 0.06, "trend": -0.06},
        ),
    ])
    with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=2)
    assert isinstance(result, list)
    assert len(result) >= 1
    for h in result:
        assert "thesis" in h
        assert "weight_biases" in h
        assert "signal_id" in h
        assert "universe" in h
        assert "expected_ic_sign" in h
        assert "uncorrelation_rationale" in h


def test_hypothesis_missing_uncorrelation_rationale_is_rejected():
    from ascent.research.factor_proposer import propose_hypotheses
    h = _full_hypothesis()
    del h["uncorrelation_rationale"]
    mock_response = json.dumps([h])
    with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
    assert result == [], "Hypothesis missing uncorrelation_rationale must be rejected"


def test_hypothesis_missing_other_required_fields_is_rejected():
    from ascent.research.factor_proposer import propose_hypotheses
    for missing in ("signal_id", "universe", "expected_ic_sign"):
        h = _full_hypothesis()
        del h[missing]
        mock_response = json.dumps([h])
        with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
            result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
        assert result == [], f"Hypothesis missing {missing} must be rejected"


def test_hypothesis_with_small_universe_is_rejected():
    from ascent.research.factor_proposer import propose_hypotheses
    h = _full_hypothesis(universe=["AAPL", "MSFT", "GOOG"])  # only 3 symbols, floor is 10
    mock_response = json.dumps([h])
    with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
    assert result == [], "Hypothesis with fewer than 10 universe symbols must be rejected"


def test_hypothesis_with_named_universe_string_is_rejected():
    """
    Bug 1: a named-universe string (e.g. "sp500", or a narrow/fabricated name
    like "my_3_favorite_biotech_names") cannot actually be resolved to a symbol
    count anywhere in this codebase -- there is no name->universe lookup for
    arbitrary LLM-invented strings. Silently assuming it clears the floor let a
    concentrated or fabricated universe sail through unrejected. It must now be
    rejected; only an explicit symbol list can satisfy the floor.
    """
    from ascent.research.factor_proposer import propose_hypotheses
    for name in ("sp500", "my_3_favorite_biotech_names"):
        h = _full_hypothesis(universe=name)
        mock_response = json.dumps([h])
        with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
            result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
        assert result == [], f"Named universe string {name!r} must be rejected, not assumed broad enough"


def test_hypothesis_with_explicit_large_list_still_accepted():
    """The fix must not break the legitimate path: an explicit list >= the floor."""
    from ascent.research.factor_proposer import propose_hypotheses
    h = _full_hypothesis(universe=_BIG_UNIVERSE)
    mock_response = json.dumps([h])
    with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
    assert len(result) == 1, "An explicit symbol list meeting the floor must still be accepted"


def test_hypothesis_with_zero_ic_sign_is_rejected():
    """
    Bug 2: expected_ic_sign must be exactly +1 or -1. The old check
    (`h[field] in (None, "")`) only rejected missing/empty fields, so a
    meaningless 0 -- not a valid sign, not a valid "absent" sentinel either --
    passed validation.
    """
    from ascent.research.factor_proposer import propose_hypotheses
    for bad_sign in (0, 2, -2, 0.0, "1", None):
        h = _full_hypothesis(expected_ic_sign=bad_sign)
        mock_response = json.dumps([h])
        with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
            result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
        assert result == [], f"expected_ic_sign={bad_sign!r} must be rejected"


def test_hypothesis_with_valid_ic_sign_is_accepted():
    from ascent.research.factor_proposer import propose_hypotheses
    for good_sign in (1, -1, 1.0, -1.0):
        h = _full_hypothesis(expected_ic_sign=good_sign)
        mock_response = json.dumps([h])
        with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
            result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=1)
        assert len(result) == 1, f"expected_ic_sign={good_sign!r} must be accepted"


def test_duplicate_hypotheses_are_deduplicated():
    from ascent.research.factor_proposer import deduplicate_hypotheses
    h1 = {"thesis": "A", "weight_biases": {"fundamental": 0.10, "trend": -0.10}}
    h2 = {"thesis": "B", "weight_biases": {"fundamental": 0.10, "trend": -0.10}}  # identical biases
    h3 = {"thesis": "C", "weight_biases": {"volatility": 0.10, "trend": -0.10}}
    result = deduplicate_hypotheses([h1, h2, h3], similarity_threshold=0.85)
    assert len(result) < 3, "Identical bias vectors must be deduplicated"
    assert len(result) >= 2, "Genuinely different hypotheses must be kept"


def test_generate_guided_variants_sums_to_one():
    from ascent.research.factor_proposer import generate_guided_variants
    hypotheses = [
        {"thesis": "More quality", "weight_biases": {"fundamental": 0.10, "trend": -0.10}},
        {"thesis": "More vol",    "weight_biases": {"volatility": 0.08, "trend": -0.08}},
    ]
    variants = generate_guided_variants(_CURRENT_WEIGHTS, hypotheses, perturb_range=0.03)
    for v in variants:
        w = v["alpha_weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total}"


def test_generate_guided_variants_respects_floor():
    from ascent.research.factor_proposer import generate_guided_variants
    hypotheses = [
        {"thesis": "Zero out trend", "weight_biases": {"trend": -0.99, "fundamental": 0.50}},
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
        {"thesis": "Quality over momentum", "weight_biases": {"fundamental": 0.08, "trend": -0.08}},
        {"thesis": "Vol regime favors stability", "weight_biases": {"volatility": 0.07, "trend": -0.07}},
        {"thesis": "Mean reversion in stress",   "weight_biases": {"meanrev": 0.06, "trend": -0.06}},
        {"thesis": "ML signals outperform",      "weight_biases": {"ml": 0.06, "trend": -0.06}},
        {"thesis": "Earnings drive alpha",       "weight_biases": {"earnings": 0.06, "trend": -0.06}},
    ]
    with patch("ascent.research.self_improve.SELF_MODIFY_ENABLED", True):
        with patch("ascent.research.factor_proposer.propose_hypotheses", return_value=mock_hyp):
            with patch("ascent.research.factor_proposer.generate_guided_variants",
                       wraps=lambda w, h, **kw: [{"variant_id": f"guided_{i}", "alpha_weights": w} for i, _ in enumerate(h)]):
                variants = generate_variants({"alpha_weights": _CURRENT_WEIGHTS}, n=5, regime="stressed")
    assert len(variants) == 5
    assert any(v["variant_id"].startswith("guided_") for v in variants), \
        "generate_variants must use guided proposer when regime is provided"


# ── insider re-testable candidate (Part B) ──────────────────────────────────

def test_insider_candidate_flows_through_guided_variants():
    """
    insider is not part of the live 2-sleeve default set (meanrev/statarb), but is
    an explicit, already-identified re-testable candidate (see
    docs/target_architecture/27_trend_insider_reconciliation.md). A hypothesis that
    references it must not be silently dropped by the `if sleeve in weights` guard.
    """
    from ascent.research.factor_proposer import generate_guided_variants
    live_weights = {"meanrev": 0.50, "statarb": 0.50}  # matches self_improve.DEFAULT_ALPHA_WEIGHTS
    hypotheses = [
        {"thesis": "Insider buying has more data now, worth re-testing",
         "weight_biases": {"insider": 0.05, "meanrev": -0.025, "statarb": -0.025}},
    ]
    variants = generate_guided_variants(live_weights, hypotheses, perturb_range=0.0)
    assert len(variants) == 1
    w = variants[0]["alpha_weights"]
    assert "insider" in w, "insider must be admitted into the variant's weight set, not dropped"
    assert w["insider"] > 0.0, "insider's proposed upweight must actually take effect"
    assert abs(sum(w.values()) - 1.0) < 1e-4


def test_trend_is_not_a_revivable_candidate():
    """
    trend has a statistically significant NEGATIVE IC per the reconciliation doc and
    must not be treated as a re-testable candidate the way insider is.
    """
    from ascent.research.factor_proposer import _ELIGIBLE_CANDIDATE_SLEEVES
    assert "trend" not in _ELIGIBLE_CANDIDATE_SLEEVES
    assert "insider" in _ELIGIBLE_CANDIDATE_SLEEVES
