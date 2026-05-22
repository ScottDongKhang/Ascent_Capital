"""
Test narrative alpha weight activation.
"""
import pytest


def test_narrative_weight_nonzero():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert DEFAULT_ALPHA_WEIGHTS["narrative"] == 0.03, (
        f"Expected narrative=0.03, got {DEFAULT_ALPHA_WEIGHTS['narrative']}"
    )


def test_weights_sum_to_one():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    total = sum(DEFAULT_ALPHA_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_self_improve_weights_match_stack():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as stack_w
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as si_w
    assert stack_w == si_w, (
        f"stack.py and self_improve.py DEFAULT_ALPHA_WEIGHTS differ: "
        f"{set(stack_w.items()) ^ set(si_w.items())}"
    )
