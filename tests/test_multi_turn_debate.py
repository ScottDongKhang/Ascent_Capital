# tests/test_multi_turn_debate.py
from unittest.mock import patch
from datetime import date
import pytest
import json


PORTFOLIO_STATE = {
    "date": "2026-04-12",
    "us_regime": "calm_bull",
    "macro_regime": "neutral",
    "n_positions": 5,
    "allocation": {"us_equities": 0.6},
    "weights": {"AAPL": 0.2, "MSFT": 0.2, "CAT": 0.2, "WMT": 0.2, "MRK": 0.2},
}

ROUND1_ARGS = {
    "bull": "Strong momentum in tech and industrials justifies full deployment.",
    "bear": "Valuations are stretched; a 10% drawdown is likely in Q2.",
    "devils_advocate": "The biggest risk is a surprise Fed hike that breaks momentum entirely.",
    "regime_specialist": "Calm bull regime supports full exposure.",
}


# ── Round 2 rebuttal agent tests ──────────────────────────────────────────────

def test_bull_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Bull rebuttal text") as mock_gen:
        from debate.agents import run_bull_rebuttal
        result = run_bull_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Bull rebuttal text"


def test_bear_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Bear rebuttal text") as mock_gen:
        from debate.agents import run_bear_rebuttal
        result = run_bear_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Bear rebuttal text"


def test_devils_advocate_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Devil rebuttal text") as mock_gen:
        from debate.agents import run_devils_advocate_rebuttal
        result = run_devils_advocate_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Devil rebuttal text"


def test_regime_specialist_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Regime rebuttal text") as mock_gen:
        from debate.agents import run_regime_specialist_rebuttal
        result = run_regime_specialist_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Regime rebuttal text"


def test_rebuttal_prompt_includes_all_round1_arguments():
    """The user prompt passed to the LLM must contain all Round 1 arguments."""
    captured_prompts = {}

    def mock_gen(system_prompt, user_prompt, **kwargs):
        captured_prompts["user"] = user_prompt
        return "rebuttal"

    with patch("debate.agents.generate_structured", side_effect=mock_gen):
        from debate.agents import run_bull_rebuttal
        run_bull_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)

    assert "Strong momentum" in captured_prompts["user"]        # bull's own round1
    assert "Valuations are stretched" in captured_prompts["user"]  # bear's round1
    assert "surprise Fed hike" in captured_prompts["user"]      # devil's round1


def test_rebuttal_failure_returns_fallback():
    """If LLM fails, rebuttal returns a non-crashing fallback string."""
    with patch("debate.agents.generate_structured", side_effect=Exception("API down")):
        from debate.agents import run_bull_rebuttal
        result = run_bull_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    assert "failed" in result.lower() or "error" in result.lower()


# ── Judge round2_args tests ───────────────────────────────────────────────────

def test_judge_accepts_round2_args():
    """Judge synthesizes Round 2 when provided."""
    round2_args = {
        "bull_rebuttal": "The bear is wrong — momentum is intact.",
        "bear_rebuttal": "The bull ignores valuation risk.",
        "devils_advocate_rebuttal": "Both sides miss the liquidity risk.",
        "regime_specialist_rebuttal": "Regime posture is fine.",
    }

    fake_verdict_json = json.dumps({
        "confidence": 0.75,
        "recommendation": "proceed",
        "key_risks": ["valuation"],
        "reasoning": "Bull case wins.",
    })

    with patch("debate.judge.generate_structured", return_value=fake_verdict_json):
        from debate.judge import run_judge
        verdict = run_judge(
            "bull round 1", "bear round 1", "devil round 1",
            PORTFOLIO_STATE,
            regime_arg="regime round 1",
            quant_check="QUANT SANITY CHECK:\n  ✓ Clean",
            round2_args=round2_args,
        )

    assert verdict["recommendation"] == "proceed"


def test_judge_round2_prompt_includes_rebuttals():
    """The judge's user prompt must include Round 2 rebuttal content."""
    captured = {}

    def mock_gen(system_prompt, user_prompt, **kwargs):
        captured["user"] = user_prompt
        return json.dumps({
            "confidence": 0.5,
            "recommendation": "reduce_size",
            "key_risks": [],
            "reasoning": "test",
        })

    round2_args = {
        "bull_rebuttal": "UniqueStringBullR2",
        "bear_rebuttal": "UniqueStringBearR2",
        "devils_advocate_rebuttal": "UniqueStringDevilR2",
        "regime_specialist_rebuttal": "UniqueStringRegimeR2",
    }

    with patch("debate.judge.generate_structured", side_effect=mock_gen):
        from debate.judge import run_judge
        run_judge(
            "bull r1", "bear r1", "devil r1",
            PORTFOLIO_STATE,
            round2_args=round2_args,
        )

    assert "UniqueStringBullR2" in captured["user"]
    assert "UniqueStringBearR2" in captured["user"]
    assert "UniqueStringDevilR2" in captured["user"]


def test_judge_works_without_round2_args():
    """run_judge still works when round2_args is not passed (backward compatible)."""
    fake_verdict_json = json.dumps({
        "confidence": 0.6,
        "recommendation": "proceed",
        "key_risks": [],
        "reasoning": "ok",
    })

    with patch("debate.judge.generate_structured", return_value=fake_verdict_json):
        from debate.judge import run_judge
        verdict = run_judge("bull", "bear", "devil", PORTFOLIO_STATE)

    assert verdict["recommendation"] == "proceed"


# ── debate_runner two-round integration test ──────────────────────────────────

def test_debate_runner_executes_two_rounds():
    """debate_runner calls both Round 1 and Round 2 agent functions."""
    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 2,
        "allocation": {},
        "weights": {"AAPL": 0.5, "MSFT": 0.5},
    }

    verdict_data = {
        "confidence": 0.7,
        "recommendation": "proceed",
        "key_risks": [],
        "reasoning": "ok",
    }

    import debate.debate_runner as dr
    with patch.object(dr, "score_pending_verdicts", return_value=0), \
         patch.object(dr, "run_pending_debriefs", return_value=0), \
         patch.object(dr, "detect_blind_spots"), \
         patch.object(dr, "load_blind_spot_context", return_value=""), \
         patch.object(dr, "run_all_scenarios", return_value=[]), \
         patch.object(dr, "scan_catalysts", return_value={"upcoming_events": [], "catalyst_text": ""}), \
         patch.object(dr, "run_bull_agent", return_value="bull_r1"), \
         patch.object(dr, "run_bear_agent", return_value="bear_r1"), \
         patch.object(dr, "run_devils_advocate", return_value="devil_r1"), \
         patch.object(dr, "run_regime_specialist", return_value="regime_r1"), \
         patch.object(dr, "run_quant_sanity_check", return_value="quant_r1"), \
         patch.object(dr, "run_bull_rebuttal", return_value="bull_r2") as mock_bull_r2, \
         patch.object(dr, "run_bear_rebuttal", return_value="bear_r2") as mock_bear_r2, \
         patch.object(dr, "run_devils_advocate_rebuttal", return_value="devil_r2") as mock_devil_r2, \
         patch.object(dr, "run_regime_specialist_rebuttal", return_value="regime_r2") as mock_regime_r2, \
         patch.object(dr, "run_judge", return_value=verdict_data) as mock_judge:
        dr.run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    # All Round 2 functions must have been called
    mock_bull_r2.assert_called_once()
    mock_bear_r2.assert_called_once()
    mock_devil_r2.assert_called_once()
    mock_regime_r2.assert_called_once()

    # Judge must have received round2_args
    _, kwargs = mock_judge.call_args
    round2 = kwargs.get("round2_args", {})
    assert round2.get("bull_rebuttal") == "bull_r2"
    assert round2.get("bear_rebuttal") == "bear_r2"
