# tests/portfolio/test_stop_loss.py
"""
Position-level stop-loss — Han, Zhou & Zhu (2014).

Exit a name that has fallen more than `threshold` below its entry price;
block re-entry for a cooldown window. Stopped weight goes to cash by
default rather than being redistributed into the remaining momentum book.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.stop_loss import (
    STOP_THRESHOLD,
    compute_stop_breaches,
    apply_stop_loss,
)


class TestComputeStopBreaches:
    def test_name_below_threshold_is_breached(self):
        entry = pd.Series({"ALGM": 66.37, "TLT": 87.41})
        now = pd.Series({"ALGM": 55.49, "TLT": 87.00})  # ALGM -16.4%, TLT -0.5%
        out = compute_stop_breaches(entry, now, threshold=0.10)
        assert out["ALGM"] is np.True_ or out["ALGM"] == True  # noqa: E712
        assert not out["TLT"]

    def test_exactly_at_threshold_is_breached(self):
        """-10.0% with a 10% stop breaches (inclusive), no float-edge escape."""
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 90.0})
        assert compute_stop_breaches(entry, now, threshold=0.10)["A"]

    def test_just_inside_threshold_is_not_breached(self):
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 90.5})  # -9.5%
        assert not compute_stop_breaches(entry, now, threshold=0.10)["A"]

    def test_gain_is_never_breached(self):
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 156.6})
        assert not compute_stop_breaches(entry, now, threshold=0.10)["A"]

    def test_missing_current_price_is_not_breached(self, caplog):
        """Fail-open: unknown price must never trigger a forced exit."""
        entry = pd.Series({"A": 100.0, "B": 100.0})
        now = pd.Series({"A": 50.0})  # B absent
        with caplog.at_level("WARNING"):
            out = compute_stop_breaches(entry, now, threshold=0.10)
        assert out["A"]
        assert not out["B"]
        assert any("B" in rec.message for rec in caplog.records)

    def test_missing_entry_price_is_not_breached(self):
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 50.0, "B": 1.0})
        out = compute_stop_breaches(entry, now, threshold=0.10)
        assert out["A"]
        assert not out["B"]

    def test_non_positive_entry_price_is_not_breached(self):
        entry = pd.Series({"A": 0.0, "B": -5.0})
        now = pd.Series({"A": 1.0, "B": 1.0})
        out = compute_stop_breaches(entry, now, threshold=0.10)
        assert not out.any()

    def test_empty_input_returns_empty(self):
        out = compute_stop_breaches(pd.Series(dtype=float), pd.Series(dtype=float))
        assert out.empty
        assert out.dtype == bool


class TestApplyStopLoss:
    def test_breached_name_goes_to_zero_and_gross_falls(self):
        w = pd.Series({"ALGM": 0.04, "MRNA": 0.04, "TLT": 0.08})
        breached = pd.Series({"ALGM": True, "MRNA": False, "TLT": False})
        out = apply_stop_loss(w, breached)
        assert out["ALGM"] == 0.0
        assert out["MRNA"] == pytest.approx(0.04)
        assert out["TLT"] == pytest.approx(0.08)
        # Freed weight becomes cash, it does NOT refill the book.
        assert out.sum() == pytest.approx(0.12)

    def test_redistribute_true_preserves_gross(self):
        w = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10})
        breached = pd.Series({"A": True, "B": False, "C": False})
        out = apply_stop_loss(w, breached, redistribute=True)
        assert out["A"] == 0.0
        assert out.sum() == pytest.approx(w.sum())
        assert out["B"] == pytest.approx(0.15)
        assert out["C"] == pytest.approx(0.15)

    def test_no_breach_is_exact_noop(self):
        w = pd.Series({"A": 0.10, "B": 0.10})
        breached = pd.Series({"A": False, "B": False})
        pd.testing.assert_series_equal(apply_stop_loss(w, breached), w.astype(float))

    def test_all_breached_goes_fully_to_cash(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        breached = pd.Series({"A": True, "B": True})
        out = apply_stop_loss(w, breached)
        assert out.sum() == pytest.approx(0.0)

    def test_all_breached_with_redistribute_does_not_divide_by_zero(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        breached = pd.Series({"A": True, "B": True})
        out = apply_stop_loss(w, breached, redistribute=True)
        assert out.sum() == pytest.approx(0.0)
        assert not out.isna().any()

    def test_symbol_missing_from_breach_series_is_kept(self):
        w = pd.Series({"A": 0.10, "B": 0.10})
        breached = pd.Series({"A": True})  # B unknown
        out = apply_stop_loss(w, breached)
        assert out["A"] == 0.0
        assert out["B"] == pytest.approx(0.10)
