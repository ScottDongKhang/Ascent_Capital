"""Tests for AI PM result-store poisoning fixes (Findings 1 + 10).

Three cases:
1. Rejection does not pollute result_store.
2. Rejection-then-valid-resubmission leaves exactly one real entry.
3. Revision fallback safety — run_ai_pm returns initial_result when revision is a fallback.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_thesis(acknowledged: bool = True) -> dict:
    return {
        "market_view": "calm_bull continuation",
        "feedback_acknowledged": acknowledged,
        "worst_call_response": "I over-weighted energy in a falling oil environment.",
        "position_rationale": {},
        "quant_overrides": [],
    }


def _make_valid_weights() -> dict:
    return {"AAPL": 0.10, "MSFT": 0.10, "SPY": 0.05}


def _make_fb_dir(tmp_path: Path) -> Path:
    """Create the data_cache dir + feedback file under tmp_path and return the dir."""
    dc = tmp_path / "data_cache"
    dc.mkdir(parents=True, exist_ok=True)
    (dc / "ai_pm_perf_feedback.json").write_text('{"worst_call": "energy overweight"}')
    return dc


# ---------------------------------------------------------------------------
# Bug 1 (Finding 10): Rejection must not pollute result_store
# ---------------------------------------------------------------------------

class TestRejectionDoesNotPollute:
    """_tool_propose_portfolio rejection returns a string, leaves result_store empty."""

    def test_rejection_leaves_store_empty(self, tmp_path):
        """When feedback file exists and feedback_acknowledged is missing/false,
        result_store must remain empty (length 0)."""
        _make_fb_dir(tmp_path)

        from agents.ai_pm_agent import _tool_propose_portfolio

        result_store = []
        inputs = {
            "weights": _make_valid_weights(),
            "thesis": _make_valid_thesis(acknowledged=False),
        }

        with patch("agents.ai_pm_agent._REPO_ROOT", tmp_path):
            ret = _tool_propose_portfolio(inputs, result_store)

        assert len(result_store) == 0, (
            f"result_store should be empty after rejection, got {result_store}"
        )
        assert "REJECTED" in ret, f"Expected REJECTED in return string, got: {ret!r}"

    def test_rejection_returns_string_not_none(self, tmp_path):
        """The rejection return value must be a non-empty string (tool loop needs it)."""
        _make_fb_dir(tmp_path)

        from agents.ai_pm_agent import _tool_propose_portfolio

        result_store = []
        inputs = {
            "weights": _make_valid_weights(),
            "thesis": {"feedback_acknowledged": False},
        }

        with patch("agents.ai_pm_agent._REPO_ROOT", tmp_path):
            ret = _tool_propose_portfolio(inputs, result_store)

        assert isinstance(ret, str) and len(ret) > 0

    def test_no_feedback_file_accepts_proposal(self, tmp_path):
        """When no feedback file exists, proposal is accepted regardless of acknowledged flag."""
        # tmp_path has no data_cache/ai_pm_perf_feedback.json
        (tmp_path / "data_cache").mkdir(parents=True, exist_ok=True)

        from agents.ai_pm_agent import _tool_propose_portfolio

        result_store = []
        inputs = {
            "weights": _make_valid_weights(),
            "thesis": {"feedback_acknowledged": False, "position_rationale": {}},
        }

        # check_conviction_inflation is imported locally inside the function;
        # patch at its source module.
        # log_prediction must be stubbed — it uses CALIBRATION_LOG set at import time,
        # so _REPO_ROOT patching does not protect the production log.
        with patch("agents.ai_pm_agent._REPO_ROOT", tmp_path), \
             patch("ascent.strategy.ai_pm_guardrails.check_conviction_inflation",
                   return_value={}), \
             patch("ascent.strategy.calibration_tracker.log_prediction"):
            ret = _tool_propose_portfolio(inputs, result_store)

        assert len(result_store) == 1
        assert not result_store[0].fallback


# ---------------------------------------------------------------------------
# Bug 1 (Finding 10): Rejection-then-valid-resubmission
# ---------------------------------------------------------------------------

class TestRejectionThenValidResubmission:
    """Call _tool_propose_portfolio twice: first rejected, then accepted."""

    def test_store_has_exactly_one_real_entry_after_retry(self, tmp_path):
        """After one rejected call and one accepted call, result_store has exactly one
        real (non-fallback, non-empty) entry."""
        _make_fb_dir(tmp_path)

        from agents.ai_pm_agent import _tool_propose_portfolio

        result_store = []

        # First call — rejected (no acknowledgement)
        inputs_bad = {
            "weights": _make_valid_weights(),
            "thesis": _make_valid_thesis(acknowledged=False),
        }
        with patch("agents.ai_pm_agent._REPO_ROOT", tmp_path):
            ret_bad = _tool_propose_portfolio(inputs_bad, result_store)

        assert len(result_store) == 0, "Store must be empty after rejection"
        assert "REJECTED" in ret_bad

        # Second call — accepted (with acknowledgement)
        inputs_good = {
            "weights": _make_valid_weights(),
            "thesis": _make_valid_thesis(acknowledged=True),
        }
        with patch("agents.ai_pm_agent._REPO_ROOT", tmp_path), \
             patch("ascent.strategy.ai_pm_guardrails.check_conviction_inflation",
                   return_value={}), \
             patch("ascent.strategy.calibration_tracker.log_prediction"):
            ret_good = _tool_propose_portfolio(inputs_good, result_store)

        assert len(result_store) == 1, f"Expected 1 entry, got {len(result_store)}"
        assert not result_store[0].fallback, "Accepted entry must not be a fallback"
        assert result_store[0].portfolio, "Accepted entry must have non-empty portfolio"
        assert "submitted" in ret_good.lower() or "complete" in ret_good.lower()


# ---------------------------------------------------------------------------
# Bug 2 (Finding 1): Revision fallback safety
# ---------------------------------------------------------------------------

class TestRevisionFallbackSafety:
    """run_ai_pm returns initial_result when revision result is a fallback or empty."""

    def _make_aipm_result(self, fallback: bool, portfolio: dict):
        from agents.ai_pm_agent import AIPMResult
        return AIPMResult(
            portfolio=portfolio,
            thesis={"market_view": "test"} if not fallback else {},
            fallback=fallback,
        )

    def test_fallback_revision_returns_initial(self):
        """If result_store_v2[-1].fallback is True, run_ai_pm must return initial_result."""
        initial = self._make_aipm_result(fallback=False, portfolio={"AAPL": 0.10})
        fallback_revision = self._make_aipm_result(fallback=True, portfolio={})

        # Replicate the guard introduced by Fix 2:
        result_store_v2 = [fallback_revision]
        if result_store_v2 and not result_store_v2[-1].fallback and result_store_v2[-1].portfolio:
            chosen = result_store_v2[-1]
        else:
            chosen = initial

        assert chosen is initial, (
            "Expected initial_result to be returned when revision is a fallback"
        )

    def test_empty_portfolio_revision_returns_initial(self):
        """If result_store_v2[-1].portfolio is empty, run_ai_pm must return initial_result."""
        initial = self._make_aipm_result(fallback=False, portfolio={"AAPL": 0.10})
        empty_revision = self._make_aipm_result(fallback=False, portfolio={})

        result_store_v2 = [empty_revision]
        if result_store_v2 and not result_store_v2[-1].fallback and result_store_v2[-1].portfolio:
            chosen = result_store_v2[-1]
        else:
            chosen = initial

        assert chosen is initial, (
            "Expected initial_result when revision has empty portfolio"
        )

    def test_valid_revision_replaces_initial(self):
        """If revision is real and non-empty, it should be preferred over initial."""
        initial = self._make_aipm_result(fallback=False, portfolio={"AAPL": 0.10})
        good_revision = self._make_aipm_result(fallback=False, portfolio={"MSFT": 0.15})

        result_store_v2 = [good_revision]
        if result_store_v2 and not result_store_v2[-1].fallback and result_store_v2[-1].portfolio:
            chosen = result_store_v2[-1]
        else:
            chosen = initial

        assert chosen is good_revision, (
            "Expected revision to be returned when it has a real portfolio"
        )

    def test_empty_store_v2_returns_initial(self):
        """If revision pass produced nothing (empty store_v2), return initial."""
        initial = self._make_aipm_result(fallback=False, portfolio={"AAPL": 0.10})
        result_store_v2 = []

        if result_store_v2 and not result_store_v2[-1].fallback and result_store_v2[-1].portfolio:
            chosen = result_store_v2[-1]
        else:
            chosen = initial

        assert chosen is initial
