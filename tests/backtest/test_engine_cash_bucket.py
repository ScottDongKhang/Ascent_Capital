# tests/backtest/test_engine_cash_bucket.py
"""
Backtest engine must preserve a partially-invested book.

Before this fix the daily drift step renormalized weights by the invested
sum alone (`drifted / drifted.sum()`), which snaps gross exposure back to
1.0 on every non-rebalance day. That silently erased the 200MA cut, vol
targeting, and any stop-loss-to-cash rule everywhere except the single
rebalance day itself.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.backtest.engine import BacktestEngine


def _synthetic_market(n_days: int = 60, n_syms: int = 2, seed: int = 0):
    """Deterministic price panel; returns (close, open_) frames."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_syms)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        {s: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days)) for s in syms},
        index=dates,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    return close, open_


class TestCashBucketPreserved:
    def test_half_invested_book_is_not_relevered(self):
        """A book targeting 0.50 gross must stay near 0.50, not snap to 1.0."""
        close, open_ = _synthetic_market()
        # 0.25 + 0.25 = 0.50 gross on every date.
        tw = pd.DataFrame(0.25, index=close.index, columns=close.columns)

        res = BacktestEngine(rebalance_freq_days=21, execution_delay=1).run(
            tw, close, open_
        )
        gross = res.held_weights.sum(axis=1)
        invested = gross[gross > 1e-9]  # skip the pre-first-signal cash period

        assert not invested.empty, "expected at least one invested day"
        # Drift around 0.50 is fine; snapping to 1.0 is the bug.
        assert invested.max() < 0.60, (
            f"gross exposure re-levered to {invested.max():.4f}; "
            "expected to stay near the 0.50 target"
        )

    def test_fully_invested_book_still_sums_to_one(self):
        """Regression: the common gross==1.0 case must be unchanged."""
        close, open_ = _synthetic_market()
        tw = pd.DataFrame(0.5, index=close.index, columns=close.columns)  # 1.0 gross

        res = BacktestEngine(rebalance_freq_days=21, execution_delay=1).run(
            tw, close, open_
        )
        gross = res.held_weights.sum(axis=1)
        invested = gross[gross > 1e-9]

        assert not invested.empty
        assert invested.max() == pytest.approx(1.0, abs=1e-6)
        assert invested.min() == pytest.approx(1.0, abs=1e-6)

    def test_derisked_book_earns_less_than_full_book(self):
        """The economic point: half the exposure means roughly half the return."""
        close, open_ = _synthetic_market(seed=3)
        half = pd.DataFrame(0.25, index=close.index, columns=close.columns)
        full = pd.DataFrame(0.50, index=close.index, columns=close.columns)

        eng = lambda: BacktestEngine(rebalance_freq_days=21, execution_delay=1)
        r_half = eng().run(half, close, open_).portfolio_returns
        r_full = eng().run(full, close, open_).portfolio_returns

        # Compare only days where the full book is actually invested.
        mask = r_full.abs() > 1e-12
        assert mask.any()
        ratio = r_half[mask].abs().sum() / r_full[mask].abs().sum()
        # Before the fix this ratio is ~1.0 (identical books after day 1).
        assert ratio < 0.75, (
            f"de-risked book moved {ratio:.3f}x as much as the full book; "
            "expected roughly half"
        )

    def test_all_cash_book_does_not_divide_by_zero(self):
        """A zero-weight book must stay flat rather than error or NaN."""
        close, open_ = _synthetic_market()
        tw = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        res = BacktestEngine(rebalance_freq_days=21, execution_delay=1).run(
            tw, close, open_
        )
        assert res.held_weights.sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)
        assert not res.portfolio_returns.isna().any()
