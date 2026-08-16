"""The judge's position change must NOT be applied on any path, anywhere.

History: `debate_judge_intervention` scored CUT (p=0.75, n=47 — underpowered,
but the same evidence bar applied consistently to every other component in
this rebuild) in the proof audit. Both call sites of
`apply_judge_position_change` in run_all_agents.py (the scheduled-rebalance
branch and the discovery mini-rebalance path) were removed 2026-08-14 as a
result, and `apply_judge_position_change` itself was deleted 2026-08-15
after a repo-wide confirm of zero remaining callers.

This file used to assert the opposite: that the write path reached both
branches (extracted 2026-07-28 after discovering the discovery path silently
dropped 4 of 7 judge position changes in history: 2026-06-15 PK, 06-22 BAX,
06-29 TLT, 07-27 VNQ). That defect is now moot -- neither path applies a
judge position change anymore, so there is nothing left to wire up.

`_apply_position_change_to_weights` is left in place (untouched, still
tested below for its own pure behaviour) in case the write path is
reinstated later; it is simply uncalled from run_all_agents.main() and
_trigger_mini_rebalance() now. `run_debate()` and verdict logging to
outputs/debate_log/ are unaffected -- the verdict, including any
position_changes, is still written in full; only the mutation of live
weights is gone.
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


class TestNeitherPathAppliesTheJudgeChange:
    """The whole point, inverted: neither path may call the write helper.
    apply_judge_position_change was deleted 2026-08-15 (zero callers), so
    it is enough to confirm it no longer exists anywhere in the module."""

    def _src(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "run_all_agents.py")) as f:
            return f.read()

    def test_apply_helper_is_gone_entirely(self):
        src = self._src()
        assert "apply_judge_position_change(" not in src, (
            "apply_judge_position_change was deleted as dead code -- neither "
            "its definition nor any call site should remain"
        )
