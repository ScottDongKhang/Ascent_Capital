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


def test_llm_fundamental_series_sleeve_does_not_crash_blend(monkeypatch):
    """llm_fundamental_alpha returns a cross-sectional (per-symbol) Series; the stack
    must broadcast it to a date×symbol panel before _cs_normalize (df.mean(axis=1)),
    otherwise the whole us_equities pipeline raises 'No axis named 1 for object type
    Series'. Regression guard for that crash."""
    import numpy as np
    import pandas as pd
    import ascent.alpha.llm_fundamental as llm_mod
    from ascent.alpha.stack import build_alpha_stack

    syms = ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = pd.DataFrame(
        np.random.RandomState(0).uniform(50, 150, size=(len(dates), len(syms))),
        index=dates, columns=syms,
    )

    # Cross-sectional snapshot, exactly what llm_fundamental_alpha returns.
    monkeypatch.setattr(
        llm_mod, "llm_fundamental_alpha",
        lambda *_a, **_k: pd.Series({"AAA": 1.2, "BBB": -0.5, "CCC": 0.3, "DDD": 0.0}),
    )

    composite = build_alpha_stack(
        {"close": close}, alpha_weights={"llm_fundamental": 1.0}
    )

    assert isinstance(composite, pd.DataFrame)
    assert not composite.empty
    assert list(composite.columns) == syms
