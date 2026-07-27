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


import json

from ascent.portfolio.stop_loss import (
    COOLDOWN_DAYS,
    load_stop_state,
    record_stops,
    blocked_symbols,
)


class TestCooldownState:
    def test_record_then_load_roundtrip(self, tmp_path):
        p = str(tmp_path / "state.json")
        record_stops(["ALGM", "MRNA"], "2026-07-02", path=p)
        state = load_stop_state(p)
        assert state == {"ALGM": "2026-07-02", "MRNA": "2026-07-02"}

    def test_record_overwrites_older_stop_for_same_symbol(self, tmp_path):
        p = str(tmp_path / "state.json")
        record_stops(["ALGM"], "2026-07-02", path=p)
        record_stops(["ALGM"], "2026-07-20", path=p)
        assert load_stop_state(p)["ALGM"] == "2026-07-20"

    def test_missing_state_file_is_empty_not_an_error(self, tmp_path):
        assert load_stop_state(str(tmp_path / "nope.json")) == {}

    def test_corrupt_state_file_is_empty_not_an_error(self, tmp_path, caplog):
        p = tmp_path / "state.json"
        p.write_text("{not valid json")
        with caplog.at_level("WARNING"):
            assert load_stop_state(str(p)) == {}
        assert caplog.records

    def test_symbol_is_blocked_inside_cooldown(self):
        state = {"ALGM": "2026-07-02"}
        assert "ALGM" in blocked_symbols(state, "2026-07-10", cooldown_days=30)

    def test_symbol_is_free_after_cooldown(self):
        state = {"ALGM": "2026-07-02"}
        assert "ALGM" not in blocked_symbols(state, "2026-08-02", cooldown_days=30)

    def test_boundary_day_is_free(self):
        """Exactly cooldown_days later the name is tradeable again."""
        state = {"A": "2026-07-01"}
        assert "A" not in blocked_symbols(state, "2026-07-31", cooldown_days=30)

    def test_unparseable_date_does_not_block(self, caplog):
        """Fail-open: bad state must not permanently freeze a symbol out."""
        with caplog.at_level("WARNING"):
            out = blocked_symbols({"A": "garbage"}, "2026-07-10")
        assert "A" not in out
        assert caplog.records

    def test_empty_state_blocks_nothing(self):
        assert blocked_symbols({}, "2026-07-10") == set()


from ascent.portfolio.stop_loss import apply_stop_loss_panel


class TestStopLossPanel:
    def test_position_is_stopped_and_stays_out_for_cooldown(self):
        dates = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        )
        # A crashes 20% on day 3; B is flat.
        close = pd.DataFrame(
            {"A": [100.0, 99.0, 80.0, 81.0], "B": [50.0, 50.0, 50.0, 50.0]},
            index=dates,
        )
        w = pd.DataFrame(0.5, index=dates, columns=["A", "B"])

        out, events = apply_stop_loss_panel(
            w, close, threshold=0.10, cooldown_days=30
        )

        assert out.loc[dates[0], "A"] == pytest.approx(0.5)   # entry day
        assert out.loc[dates[1], "A"] == pytest.approx(0.5)   # -1%, fine
        assert out.loc[dates[2], "A"] == 0.0                  # -20%, stopped
        assert out.loc[dates[3], "A"] == 0.0                  # cooldown
        # B untouched throughout — no redistribution. (Exact equality: B is
        # never touched by the rule, so there is no float drift to tolerate.
        # `Series == pytest.approx(scalar)` is broken in this pandas/pytest
        # combo — Series.__eq__ intercepts before ApproxScalar's reflected
        # comparison runs and silently returns all-False; confirmed via
        # `out["B"].values == pytest.approx(0.5)` which correctly gives True.)
        assert (out["B"] == 0.5).all()

        assert len(events) == 1
        assert events[0]["symbol"] == "A"
        assert events[0]["entry_price"] == pytest.approx(100.0)
        assert events[0]["pct_from_entry"] == pytest.approx(-0.20)

    def test_gross_falls_when_a_name_is_stopped(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        close = pd.DataFrame({"A": [100.0, 80.0], "B": [50.0, 50.0]}, index=dates)
        w = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
        out, _ = apply_stop_loss_panel(w, close, threshold=0.10)
        assert out.loc[dates[0]].sum() == pytest.approx(1.0)
        assert out.loc[dates[1]].sum() == pytest.approx(0.5)  # A -> cash

    def test_reentry_allowed_after_cooldown_resets_entry_price(self):
        dates = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-03-01", "2026-03-02"]
        )
        close = pd.DataFrame({"A": [100.0, 80.0, 40.0, 39.0]}, index=dates)
        w = pd.DataFrame(1.0, index=dates, columns=["A"])
        out, events = apply_stop_loss_panel(w, close, threshold=0.10,
                                            cooldown_days=30)
        assert out.loc[dates[1], "A"] == 0.0     # stopped
        assert out.loc[dates[2], "A"] == 1.0     # cooldown expired, re-entered
        # Re-entry price is 40.0, so -2.5% on the next day is NOT a breach.
        assert out.loc[dates[3], "A"] == 1.0
        assert len(events) == 1

    def test_name_that_exits_naturally_clears_its_entry(self):
        """Weight -> 0 by the strategy, then back in later at a new price."""
        dates = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03"]
        )
        close = pd.DataFrame({"A": [100.0, 100.0, 91.0]}, index=dates)
        w = pd.DataFrame({"A": [1.0, 0.0, 1.0]}, index=dates)
        out, events = apply_stop_loss_panel(w, close, threshold=0.10)
        # Re-entry on day 3 at 91.0 is a fresh entry, not -9% from 100.
        assert out.loc[dates[2], "A"] == 1.0
        assert events == []

    def test_disabled_threshold_zero_is_a_noop(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        close = pd.DataFrame({"A": [100.0, 1.0]}, index=dates)
        w = pd.DataFrame(1.0, index=dates, columns=["A"])
        out, events = apply_stop_loss_panel(w, close, threshold=0.0)
        pd.testing.assert_frame_equal(out, w.astype(float))
        assert events == []

    def test_empty_weights_returns_empty(self):
        out, events = apply_stop_loss_panel(pd.DataFrame(), pd.DataFrame())
        assert out.empty
        assert events == []


class TestAlgmMrnaRegression:
    """
    The 2026-06-29 -> 2026-07-24 episode that motivated this work.

    Real closes. Entry prices are the 2026-06-29 rebalance closes. Held to
    the end, ALGM returned -30.65% and MRNA -22.42%. A 10% stop should exit
    ALGM on 2026-07-02 and MRNA on 2026-07-17.
    """

    ALGM = [66.370003, 69.620003, 63.200001, 55.485001, 56.560001, 51.549999,
            51.465000, 57.380001, 54.869999, 50.855000, 52.320000, 50.029999,
            47.119999, 46.480000, 46.360001, 49.349998, 49.869999, 50.070000,
            46.029999]
    MRNA = [69.699997, 70.029999, 72.500000, 79.760002, 81.800003, 79.769997,
            73.800003, 76.559998, 68.269997, 67.010002, 67.440002, 68.279999,
            63.150002, 61.820000, 59.490002, 59.660000, 58.070000, 57.020000,
            54.070000]
    DATES = ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
             "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
             "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
             "2026-07-16", "2026-07-17", "2026-07-20", "2026-07-21",
             "2026-07-22", "2026-07-23", "2026-07-24"]

    def _panel(self):
        idx = pd.to_datetime(self.DATES)
        close = pd.DataFrame({"ALGM": self.ALGM, "MRNA": self.MRNA}, index=idx)
        w = pd.DataFrame(0.04086962656941537, index=idx,
                         columns=["ALGM", "MRNA"])
        return w, close

    def test_ten_percent_stop_exits_both_on_the_expected_dates(self):
        w, close = self._panel()
        out, events = apply_stop_loss_panel(w, close, threshold=0.10,
                                            cooldown_days=30)
        by_sym = {e["symbol"]: e for e in events}
        assert set(by_sym) == {"ALGM", "MRNA"}
        assert str(pd.Timestamp(by_sym["ALGM"]["date"]).date()) == "2026-07-02"
        assert str(pd.Timestamp(by_sym["MRNA"]["date"]).date()) == "2026-07-17"
        # Both stay out for the remainder (cooldown covers the window).
        assert out.iloc[-1].sum() == pytest.approx(0.0)

    def test_ten_percent_stop_saves_about_one_percentage_point(self):
        """
        Measured 2026-07-27: +1.037pp vs holding. Assert the magnitude, with
        tolerance for the exact fill convention.
        """
        w, close = self._panel()
        _, events = apply_stop_loss_panel(w, close, threshold=0.10)
        weight = 0.04086962656941537
        held = {"ALGM": close["ALGM"].iloc[-1] / close["ALGM"].iloc[0] - 1,
                "MRNA": close["MRNA"].iloc[-1] / close["MRNA"].iloc[0] - 1}
        saved = sum(
            (e["pct_from_entry"] - held[e["symbol"]]) * weight for e in events
        )
        assert saved == pytest.approx(0.01037, abs=0.002)

    def test_twenty_percent_stop_saves_much_less(self):
        """Threshold matters: a 20% stop recovers roughly a third as much."""
        w, close = self._panel()
        _, events = apply_stop_loss_panel(w, close, threshold=0.20)
        weight = 0.04086962656941537
        held = {"ALGM": close["ALGM"].iloc[-1] / close["ALGM"].iloc[0] - 1,
                "MRNA": close["MRNA"].iloc[-1] / close["MRNA"].iloc[0] - 1}
        saved = sum(
            (e["pct_from_entry"] - held[e["symbol"]]) * weight for e in events
        )
        assert saved == pytest.approx(0.0034, abs=0.002)
        assert saved < 0.01037


class TestFailOpenGuard:
    """
    Plan-mandated contract: "Risk overlays never raise. On any internal
    failure, log a warning and return the input unchanged (fail-open),
    matching enforce_cluster_cap."

    Each public function delegates to an `_impl` helper under a blanket
    try/except. These tests force the `_impl` to raise via monkeypatch —
    the only honest way to exercise the except branch, since none of the
    module's real code paths raise on this pandas version.
    """

    def test_compute_stop_breaches_guards_unexpected_error(self, monkeypatch, caplog):
        import ascent.portfolio.stop_loss as sl

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(sl, "_compute_stop_breaches_impl", boom)
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 50.0})
        with caplog.at_level("WARNING"):
            out = sl.compute_stop_breaches(entry, now)
        assert not out.any()
        assert set(out.index) == {"A"}
        assert any("compute_stop_breaches" in rec.message for rec in caplog.records)

    def test_apply_stop_loss_guards_unexpected_error(self, monkeypatch, caplog):
        import ascent.portfolio.stop_loss as sl

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(sl, "_apply_stop_loss_impl", boom)
        w = pd.Series({"A": 0.5, "B": 0.5})
        breached = pd.Series({"A": True, "B": False})
        with caplog.at_level("WARNING"):
            out = sl.apply_stop_loss(w, breached)
        assert out is w  # unchanged, not a copy
        assert any("apply_stop_loss" in rec.message for rec in caplog.records)

    def test_apply_stop_loss_panel_guards_unexpected_error(self, monkeypatch, caplog):
        import ascent.portfolio.stop_loss as sl

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(sl, "_apply_stop_loss_panel_impl", boom)
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        w = pd.DataFrame({"A": [1.0, 1.0]}, index=dates)
        close = pd.DataFrame({"A": [100.0, 80.0]}, index=dates)
        with caplog.at_level("WARNING"):
            out, events = sl.apply_stop_loss_panel(w, close)
        assert out is w  # unchanged, not a copy
        assert events == []
        assert any("apply_stop_loss_panel" in rec.message for rec in caplog.records)
