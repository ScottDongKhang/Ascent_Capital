"""tests/test_alpha_stack_weights.py
Weight integrity + stack skip-on-empty tests.

Post proof-audit reduction: DEFAULT_ALPHA_WEIGHTS carries only meanrev/statarb
(the 2 of 15 sleeves that cleared the walk-forward significance bar). The
earnings_tone-specific skip-on-empty regression and the DEFAULT_ALPHA_WEIGHTS_BY_REGIME
sum check are gone with it: earnings_tone is no longer live-weighted, and there is
nothing left to regime-tilt between two equal-weighted sleeves. The earnings_tone
sleeve implementation itself is untouched and still importable.
"""
import pandas as pd
import pytest
from unittest.mock import patch


# ── test_earnings_tone_sleeve_still_returns_empty_on_empty_cache ──────────────
# (sleeve implementation code stays in the repo for future re-measurement;
# this regression guard is kept even though the sleeve is no longer weighted)

def test_earnings_tone_sleeve_still_returns_empty_on_empty_cache():
    """earnings_tone_alpha still returns an empty DF on an empty transcript cache."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha
    features = {"close": pd.DataFrame(150.0, index=pd.bdate_range("2026-01-05", periods=3), columns=["AAPL"])}

    with patch("ascent.alpha.earnings_tone.load_transcript_signals", return_value=pd.DataFrame()):
        result = earnings_tone_alpha(features)

    assert result.empty, "Empty cache must produce empty DF"


# ── test_default_weights_sum_to_one ──────────────────────────────────────────

def test_default_weights_sum_to_one():
    """DEFAULT_ALPHA_WEIGHTS in stack.py must sum to 1.0."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS

    total = sum(DEFAULT_ALPHA_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, \
        f"DEFAULT_ALPHA_WEIGHTS must sum to 1.0, got {total}"


# ── test_default_alpha_weights_reduced_to_meanrev_statarb ─────────────────────

def test_default_alpha_weights_reduced_to_meanrev_statarb():
    """Only meanrev/statarb cleared the proof audit's walk-forward significance bar."""
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as SI_WEIGHTS
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as STACK_WEIGHTS

    assert set(STACK_WEIGHTS) == set(SI_WEIGHTS) == {"meanrev", "statarb"}, \
        "DEFAULT_ALPHA_WEIGHTS key sets must match between stack.py and self_improve.py " \
        "(integrity constraint #6) and be reduced to {meanrev, statarb}"

    assert STACK_WEIGHTS == SI_WEIGHTS == {"meanrev": 0.50, "statarb": 0.50}
