"""
Test narrative alpha weight activation.
"""
import pytest


def test_default_alpha_weights_key_sets_match():
    """Post-reduction: only meanrev/statarb survive, equal-weighted, in both files."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as stack_w
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as si_w

    assert set(stack_w) == set(si_w) == {"meanrev", "statarb"}
    assert stack_w == {"meanrev": 0.50, "statarb": 0.50}


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


def test_build_alpha_stack_ignores_regime_signal():
    """regime_overlay scored CUT (p=0.35) in the proof audit — build_alpha_stack()
    must produce an identical composite regardless of what regime_signal is passed.
    Regression guard for removing the regime-conditional weight-adjustment branch."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from enum import Enum
    from ascent.alpha.stack import build_alpha_stack

    syms = ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.RandomState(1)
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, size=(len(dates), len(syms))), axis=0),
        index=dates, columns=syms,
    )
    features = {
        "close": close,
        "returns_1d": close.pct_change().fillna(0),
        "mom_252d": close.pct_change(252).fillna(0),
        "vol_21d": close.pct_change().rolling(21).std().fillna(0.01),
    }

    class _Label(Enum):
        STRESSED = "stressed"

    fake_regime_signal = SimpleNamespace(label=_Label.STRESSED)

    composite_no_regime = build_alpha_stack(features, regime_signal=None)
    composite_with_regime = build_alpha_stack(features, regime_signal=fake_regime_signal)

    pd.testing.assert_frame_equal(composite_no_regime, composite_with_regime)


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
