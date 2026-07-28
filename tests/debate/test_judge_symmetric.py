"""
tests/debate/test_judge_symmetric.py

Unit tests for Wave 4 judge fixes:
  - Parse failure defaults to proceed + degraded=True (Finding 9)
  - Symmetric validation: conviction_press increases + reductions (Finding 4)
"""

import json
import pytest
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

_PORTFOLIO_STATE = {
    "date": "2026-06-10",
    "us_regime": "calm_bull",
    "n_positions": 3,
    "weights": {
        "AAPL": 0.07,
        "MSFT": 0.08,
        "WMT":  0.05,
    },
}


def _make_valid_verdict(recommendation="proceed", changes=None):
    """Return JSON string that the judge's LLM call would produce."""
    return json.dumps({
        "confidence": 0.8,
        "recommendation": recommendation,
        "key_risks": ["risk A"],
        "reasoning": "test reasoning",
        "position_changes": changes or [],
    })


# ── Test 1: Parse failure returns proceed + degraded ─────────────────────────

class TestParseFailureDefault:
    @patch("debate.judge.extended_thinking_completion")
    def test_invalid_json_returns_proceed_degraded(self, mock_llm):
        """Parse failure must return proceed + degraded=True, never reduce_size."""
        mock_llm.return_value = "This is not JSON at all!!"

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["recommendation"] == "proceed", (
            f"Expected 'proceed' on parse failure, got '{result['recommendation']}'"
        )
        assert result.get("degraded") is True, "Expected degraded=True on parse failure"
        assert result["position_changes"] == []
        assert result["confidence"] == 0.0
        # key_risks should mention parse failure
        assert any("judge_parse_failure" in r for r in result["key_risks"])

    @patch("debate.judge.extended_thinking_completion")
    def test_missing_keys_returns_proceed_degraded(self, mock_llm):
        """JSON missing required keys (assertion error) also returns proceed+degraded."""
        mock_llm.return_value = json.dumps({"reasoning": "no confidence or recommendation"})

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["recommendation"] == "proceed"
        assert result.get("degraded") is True


# ── Test 2: Valid reduction still works ───────────────────────────────────────

class TestReductionStillWorks:
    @patch("debate.judge.extended_thinking_completion")
    def test_reduction_passes_validation(self, mock_llm):
        """A standard reduction (new_w < curr_w) must pass validation."""
        changes = [{
            "symbol": "AAPL",
            "current_weight": 0.07,
            "new_weight": 0.04,   # clear reduction, >0.5pp, above 1% floor
            "intervention_type": "adversarial_thesis",
            "reason": "adversarial score 0.8",
            "prediction": "AAPL underperforms SPY over next 10 days",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["recommendation"] == "proceed"
        assert len(result["position_changes"]) == 1
        pc = result["position_changes"][0]
        assert pc["symbol"] == "AAPL"
        assert pc["new_weight"] < pc["current_weight"]

    @patch("debate.judge.extended_thinking_completion")
    def test_reduction_not_a_reduction_is_rejected(self, mock_llm):
        """A non-conviction_press 'reduction' where new_w >= curr_w must be rejected."""
        changes = [{
            "symbol": "AAPL",
            "current_weight": 0.07,
            "new_weight": 0.09,   # increase, not a reduction
            "intervention_type": "adversarial_thesis",
            "reason": "test",
            "prediction": "test",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["position_changes"] == [], (
            "A non-conviction_press change that increases weight should be rejected"
        )


# ── Test 3: conviction_press above max_weight is clamped ─────────────────────

class TestConvictionPressMaxWeightCap:
    @patch("debate.judge.extended_thinking_completion")
    def test_conviction_press_clamped_to_max_weight(self, mock_llm):
        """conviction_press with new_w=0.15 must be clamped to 0.10."""
        changes = [{
            "symbol": "WMT",
            "current_weight": 0.05,
            "new_weight": 0.15,   # exceeds MAX_WEIGHT=0.10
            "intervention_type": "conviction_press",
            "reason": "top-quartile alpha rank, clean crowding, no bear concern",
            "prediction": "WMT outperforms SPY over next 10 days",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        # conviction_press should still be accepted but clamped to 0.10
        assert len(result["position_changes"]) == 1
        pc = result["position_changes"][0]
        assert pc["symbol"] == "WMT"
        assert pc["new_weight"] <= 0.10, (
            f"Expected new_weight <= 0.10 after cap, got {pc['new_weight']}"
        )
        assert pc["new_weight"] > pc["current_weight"], (
            "conviction_press must still increase the position after clamping"
        )


# ── Test 4: conviction_press that doesn't increase is rejected ────────────────

class TestConvictionPressNoIncrease:
    @patch("debate.judge.extended_thinking_completion")
    def test_conviction_press_equal_weight_rejected(self, mock_llm):
        """conviction_press with new_w == curr_w must be rejected."""
        changes = [{
            "symbol": "AAPL",
            "current_weight": 0.07,
            "new_weight": 0.07,   # no change
            "intervention_type": "conviction_press",
            "reason": "test",
            "prediction": "test",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["position_changes"] == [], (
            "conviction_press with new_w <= curr_w must be rejected"
        )

    @patch("debate.judge.extended_thinking_completion")
    def test_conviction_press_lower_weight_rejected(self, mock_llm):
        """conviction_press with new_w < curr_w must be rejected."""
        changes = [{
            "symbol": "AAPL",
            "current_weight": 0.07,
            "new_weight": 0.04,   # decrease — wrong direction for conviction_press
            "intervention_type": "conviction_press",
            "reason": "test",
            "prediction": "test",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["position_changes"] == [], (
            "conviction_press with new_w < curr_w must be rejected"
        )


# ── Test 5: Trivial change (< 0.5pp) is dropped ─────────────────────────────

class TestTrivialChangeDrop:
    @patch("debate.judge.extended_thinking_completion")
    def test_trivial_reduction_dropped(self, mock_llm):
        """A change < 0.5pp (0.005) must be dropped as not worth the cost."""
        changes = [{
            "symbol": "AAPL",
            "current_weight": 0.07,
            "new_weight": 0.0660,  # only 0.4pp change — below 0.5pp threshold
            "intervention_type": "adversarial_thesis",
            "reason": "tiny trim",
            "prediction": "test",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["position_changes"] == [], (
            f"Expected trivial change to be dropped, got {result['position_changes']}"
        )

    @patch("debate.judge.extended_thinking_completion")
    def test_trivial_conviction_press_dropped(self, mock_llm):
        """A conviction_press < 0.5pp must also be dropped."""
        changes = [{
            "symbol": "AAPL",
            "current_weight": 0.07,
            "new_weight": 0.0730,  # only 0.3pp — below threshold
            "intervention_type": "conviction_press",
            "reason": "tiny increase",
            "prediction": "test",
        }]
        mock_llm.return_value = _make_valid_verdict(changes=changes)

        from debate.judge import run_judge
        result = run_judge(
            bull_argument="bull",
            bear_argument="bear",
            devils_argument="devil",
            portfolio_state=_PORTFOLIO_STATE,
        )

        assert result["position_changes"] == [], (
            f"Expected trivial conviction_press to be dropped, got {result['position_changes']}"
        )


# ── Test 6: conviction_press in _ALL_TYPES ────────────────────────────────────

class TestConvictionPressInAllTypes:
    def test_conviction_press_in_all_types(self):
        """conviction_press must be in _ALL_TYPES so authority is tracked."""
        from debate.adversarial_authority import _ALL_TYPES
        assert "conviction_press" in _ALL_TYPES

    def test_get_authority_conviction_press(self, tmp_path, monkeypatch):
        """get_authority('conviction_press') must return valid authority dict."""
        monkeypatch.setattr("debate.adversarial_authority.AUTHORITY_PATH",
                            tmp_path / "auth.json")
        from debate.adversarial_authority import get_authority
        auth = get_authority("conviction_press")
        assert "allowed_change_pct" in auth
        assert auth["suspended"] is False
        assert auth["allowed_change_pct"] > 0


# ── Protected positions + reduction_pct (2026-07-28) ─────────────────────────

class TestProtectedPositionsContract:
    """The judge's protection used to exist only as prose in `reasoning`, which
    the execution layer never reads. On 2026-07-27 a verdict argued UUP/TLT must
    not be cut 48h before FOMC and the size-sorted fallback sold both. These
    fields are the machine-readable channel."""

    def _verdict(self, **extra):
        base = {
            "confidence": 0.6,
            "recommendation": "reduce_size",
            "key_risks": ["correlated drawdown"],
            "reasoning": "de-risk but keep the hedge leg",
            "position_changes": [],
        }
        base.update(extra)
        return json.dumps(base)

    @patch("debate.judge.extended_thinking_completion")
    def test_held_protected_symbols_survive(self, mock_llm):
        from debate.judge import run_judge
        mock_llm.return_value = self._verdict(
            protected_positions=[{"symbol": "AAPL", "reason": "hedge leg"}])
        v = run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)
        assert v["protected_positions"] == [{"symbol": "AAPL", "reason": "hedge leg"}]

    @patch("debate.judge.extended_thinking_completion")
    def test_unheld_symbols_are_dropped(self, mock_llm):
        """A hallucinated ticker would shield nothing while shrinking the sleeve
        available to fund the reduction."""
        from debate.judge import run_judge
        mock_llm.return_value = self._verdict(
            protected_positions=[{"symbol": "NVDA", "reason": "not held"}])
        v = run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)
        assert v["protected_positions"] == []

    @patch("debate.judge.extended_thinking_completion")
    def test_bare_strings_accepted_and_upper_cased(self, mock_llm):
        from debate.judge import run_judge
        mock_llm.return_value = self._verdict(protected_positions=[" aapl "])
        v = run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)
        assert v["protected_positions"] == [{"symbol": "AAPL", "reason": ""}]

    @patch("debate.judge.extended_thinking_completion")
    def test_field_always_present_even_when_omitted(self, mock_llm):
        from debate.judge import run_judge
        mock_llm.return_value = self._verdict()
        v = run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)
        assert v["protected_positions"] == []

    @patch("debate.judge.extended_thinking_completion")
    def test_reduction_pct_is_clamped(self, mock_llm):
        from debate.judge import run_judge
        mock_llm.return_value = self._verdict(reduction_pct=0.95)
        assert run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)["reduction_pct"] == 0.25

    @patch("debate.judge.extended_thinking_completion")
    def test_garbage_reduction_pct_is_removed(self, mock_llm):
        from debate.judge import run_judge
        mock_llm.return_value = self._verdict(reduction_pct="a lot")
        assert "reduction_pct" not in run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)

    @patch("debate.judge.extended_thinking_completion")
    def test_degraded_path_still_exposes_the_field(self, mock_llm):
        from debate.judge import run_judge
        mock_llm.return_value = "not json"
        v = run_judge(bull_argument='bull', bear_argument='bear', devils_argument='devil', portfolio_state=_PORTFOLIO_STATE)
        assert v["degraded"] is True
        assert v["protected_positions"] == []
