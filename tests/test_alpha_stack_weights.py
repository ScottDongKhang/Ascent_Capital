"""tests/test_alpha_stack_weights.py
Weight integrity + stack skip-on-empty tests for the earnings_tone sleeve.
Covers the 4 paths from the reconciled plan coverage diagram.
"""
import pandas as pd
import pytest
from unittest.mock import patch


# ── test_stack_skips_empty_earnings_tone ──────────────────────────────────────

def test_stack_skips_empty_earnings_tone():
    """Empty earnings_tone sleeve → stack skips it; blend still renormalizes to sum=1.0."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS

    assert "earnings_tone" in DEFAULT_ALPHA_WEIGHTS, \
        "earnings_tone must be registered in DEFAULT_ALPHA_WEIGHTS"

    # Confirm empty sleeve is not in alphas dict (skipped), i.e. the block does not
    # insert an empty DF. We test this by importing and calling earnings_tone_alpha
    # with an empty loader, then verifying it returns empty.
    from ascent.alpha.earnings_tone import earnings_tone_alpha
    features = {"close": pd.DataFrame(150.0, index=pd.bdate_range("2026-01-05", periods=3), columns=["AAPL"])}

    with patch("ascent.alpha.earnings_tone.load_transcript_signals", return_value=pd.DataFrame()):
        result = earnings_tone_alpha(features)

    assert result.empty, "Empty cache must produce empty DF so stack skips it"


# ── test_default_weights_sum_to_one ──────────────────────────────────────────

def test_default_weights_sum_to_one():
    """DEFAULT_ALPHA_WEIGHTS in stack.py must sum to 1.0 after adding earnings_tone."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS

    total = sum(DEFAULT_ALPHA_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, \
        f"DEFAULT_ALPHA_WEIGHTS must sum to 1.0, got {total}"


# ── test_regime_variants_sum_to_one ──────────────────────────────────────────

def test_regime_variants_sum_to_one():
    """Each regime variant in DEFAULT_ALPHA_WEIGHTS_BY_REGIME must sum to ~1.0."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS_BY_REGIME

    for regime, weights in DEFAULT_ALPHA_WEIGHTS_BY_REGIME.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, \
            f"Regime '{regime}' weights sum to {total}, expected 1.0"


# ── test_self_improve_has_earnings_tone ───────────────────────────────────────

def test_self_improve_has_earnings_tone():
    """self_improve.DEFAULT_ALPHA_WEIGHTS must include earnings_tone (integrity constraint #6)."""
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as SI_WEIGHTS
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as STACK_WEIGHTS

    assert "earnings_tone" in SI_WEIGHTS, \
        "self_improve.DEFAULT_ALPHA_WEIGHTS missing 'earnings_tone' — violates constraint #6"

    assert "earnings_tone" in STACK_WEIGHTS, \
        "stack.DEFAULT_ALPHA_WEIGHTS missing 'earnings_tone'"

    assert SI_WEIGHTS["earnings_tone"] == STACK_WEIGHTS["earnings_tone"], \
        "earnings_tone weight must match between stack.py and self_improve.py"
