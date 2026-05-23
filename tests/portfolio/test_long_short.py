import pytest
import pandas as pd
import numpy as np
from ascent.portfolio.long_short import build_long_short_weights


def _make_alpha(symbols: list[str], seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(0, 1, len(symbols)), index=symbols)


def test_weights_sum_to_one():
    alpha = _make_alpha(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    weights = build_long_short_weights(alpha, long_n=6, short_n=3)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_long_short_signs():
    """Top 3 should be positive (long), bottom 2 should be negative (short)."""
    alpha = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": -0.5, "E": -2.0})
    weights = build_long_short_weights(alpha, long_n=3, short_n=2,
                                       long_pct=1.30, short_pct=0.30)
    assert weights["A"] > 0
    assert weights["B"] > 0
    assert weights["C"] > 0
    assert weights["D"] < 0
    assert weights["E"] < 0


def test_long_exposure_is_130_pct():
    alpha = _make_alpha(["A", "B", "C", "D", "E", "F"])
    weights = build_long_short_weights(alpha, long_n=4, short_n=2,
                                       long_pct=1.30, short_pct=0.30)
    long_sum = sum(v for v in weights.values() if v > 0)
    assert abs(long_sum - 1.30) < 1e-6


def test_short_exposure_is_30_pct():
    alpha = _make_alpha(["A", "B", "C", "D", "E", "F"])
    weights = build_long_short_weights(alpha, long_n=4, short_n=2,
                                       long_pct=1.30, short_pct=0.30)
    short_sum = abs(sum(v for v in weights.values() if v < 0))
    assert abs(short_sum - 0.30) < 1e-6


def test_max_position_cap_respected():
    alpha = _make_alpha(["A", "B", "C", "D", "E"])
    weights = build_long_short_weights(alpha, long_n=3, short_n=2,
                                       long_pct=1.30, short_pct=0.30,
                                       max_long_weight=0.15)
    for sym, w in weights.items():
        if w > 0:
            assert w <= 0.15 + 1e-9


def test_not_enough_symbols():
    """Fewer symbols than long_n + short_n — should raise ValueError."""
    alpha = pd.Series({"A": 1.0, "B": -1.0})
    with pytest.raises(ValueError, match="Not enough symbols"):
        build_long_short_weights(alpha, long_n=3, short_n=2)
