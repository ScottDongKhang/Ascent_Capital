"""The judge's position change must be applied on every path that runs a debate.

Audit, 2026-07-27: the apply-and-record block lived only in the scheduled
rebalance branch of run_all_agents.main(). The discovery mini-rebalance path
called run_debate(), wrote a complete verdict file with a sized, reasoned,
falsifiable position change — and then honoured only `halt_and_review`, never
applying `position_changes` and never calling record_intervention().

4 of 7 judge position changes in history died there (2026-06-15 PK, 06-22 BAX,
06-29 TLT, 07-27 VNQ — all off-calendar days). Confirmed against executed
weights: on 06-15 the judge said cut PK to 6.22% and PK executed at 0.072221,
untouched. Because record_intervention() never ran, those predictions were never
scored, so n_scored stayed 0 and the authority tier stayed frozen at the 1.0pp
floor — a self-sealing loop.
"""

import datetime as dt

import pytest

from run_all_agents import _apply_position_change_to_weights


def _change(symbol="BAX", new_weight=0.065, current_weight=0.075, **kw):
    c = {
        "symbol": symbol,
        "new_weight": new_weight,
        "current_weight": current_weight,
        "intervention_type": "adversarial_thesis",
        "reason": "worst momentum in book",
        "prediction": f"{symbol} underperforms SPY over next 10 trading days",
    }
    c.update(kw)
    return c


class TestReduction:
    def test_reduction_lands_on_the_requested_weight(self):
        w = {"BAX": 0.075, "AAPL": 0.5, "MSFT": 0.425}
        out, applied = _apply_position_change_to_weights(w, _change())
        assert applied is True
        assert out["BAX"] == pytest.approx(0.065, abs=1e-6)

    def test_book_stays_normalized(self):
        w = {"BAX": 0.075, "AAPL": 0.5, "MSFT": 0.425}
        out, _ = _apply_position_change_to_weights(w, _change())
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)

    def test_freed_weight_goes_to_the_others_proportionally(self):
        w = {"BAX": 0.10, "AAPL": 0.60, "MSFT": 0.30}
        out, _ = _apply_position_change_to_weights(w, _change(new_weight=0.05))
        assert out["AAPL"] > w["AAPL"] and out["MSFT"] > w["MSFT"]
        assert out["AAPL"] / out["MSFT"] == pytest.approx(w["AAPL"] / w["MSFT"], rel=1e-6)


class TestIncrease:
    def test_conviction_press_is_funded_from_the_others(self):
        w = {"BAX": 0.05, "AAPL": 0.60, "MSFT": 0.35}
        out, applied = _apply_position_change_to_weights(w, _change(new_weight=0.08))
        assert applied is True
        assert out["BAX"] > w["BAX"]
        assert out["AAPL"] < w["AAPL"]

    def test_increase_is_capped_at_max_weight(self):
        w = {"BAX": 0.05, "AAPL": 0.60, "MSFT": 0.35}
        out, _ = _apply_position_change_to_weights(w, _change(new_weight=0.40))
        assert out["BAX"] <= 0.10 + 1e-6


class TestRejections:
    def test_unknown_symbol_is_not_applied(self):
        w = {"AAPL": 0.6, "MSFT": 0.4}
        out, applied = _apply_position_change_to_weights(w, _change(symbol="NVDA"))
        assert applied is False
        assert out == w

    def test_below_one_percent_floor_is_not_applied(self):
        w = {"BAX": 0.075, "AAPL": 0.925}
        out, applied = _apply_position_change_to_weights(w, _change(new_weight=0.005))
        assert applied is False
        assert out == w

    def test_empty_or_malformed_change_is_not_applied(self):
        w = {"AAPL": 0.6, "MSFT": 0.4}
        for bad in ({}, None, {"symbol": ""}, {"symbol": "AAPL", "new_weight": "x"}):
            out, applied = _apply_position_change_to_weights(w, bad)
            assert applied is False
            assert out == w

    def test_input_is_not_mutated(self):
        w = {"BAX": 0.075, "AAPL": 0.925}
        before = dict(w)
        _apply_position_change_to_weights(w, _change())
        assert w == before, "caller's dict must not be mutated in place"


class TestDiscoveryPathIsWired:
    """The whole point: the discovery path must go through the shared helper."""

    def _src(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "run_all_agents.py")) as f:
            return f.read()

    def test_apply_helper_is_called_more_than_once(self):
        src = self._src()
        assert src.count("apply_judge_position_change(") >= 3, (
            "expected the definition plus at least two call sites "
            "(scheduled rebalance and discovery mini-rebalance)"
        )

    def test_discovery_path_no_longer_only_checks_halt(self):
        src = self._src()
        i = src.index("def _trigger_mini_rebalance")
        tail = src[i:i + 8000]
        assert "apply_judge_position_change(" in tail, (
            "_trigger_mini_rebalance runs a debate but never applies its verdict"
        )
