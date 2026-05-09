# tests/test_regime_conditional_personas.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def _write_credibility(path: Path, bull_stressed=0.35, bear_stressed=0.72, n=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    cred = {
        "by_regime": {
            "stressed": {"bull": bull_stressed, "bear": bear_stressed,
                         "devil": 0.60, "regime_specialist": 0.55},
            "calm_bull": {"bull": 0.68, "bear": 0.45},
        },
        "sample_counts": {
            "stressed":  {"bull": n, "bear": n, "devil": n, "regime_specialist": n},
            "calm_bull": {"bull": n, "bear": n},
        },
    }
    path.write_text(json.dumps(cred))


def test_get_agent_regime_accuracy_returns_float(tmp_path):
    from debate.outcome_tracker import get_agent_regime_accuracy
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path)
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        acc = get_agent_regime_accuracy("bull", "stressed")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_get_agent_regime_accuracy_none_for_missing(tmp_path):
    from debate.outcome_tracker import get_agent_regime_accuracy
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, n=5)  # below min sample count of 10
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        acc = get_agent_regime_accuracy("bull", "stressed")  # n=5 < min_samples=10
    assert acc is None


def test_agent_track_record_injected_into_bull_prompt(tmp_path):
    from debate.agents import _get_agent_track_record
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, bull_stressed=0.35)
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        record = _get_agent_track_record("bull", "stressed")
    assert "35%" in record or "0.35" in record or "35" in record, \
        "Track record must include the accuracy percentage"
    assert "stressed" in record.lower() or "STRESSED" in record


def test_track_record_warns_when_accuracy_below_50(tmp_path):
    from debate.agents import _get_agent_track_record
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, bull_stressed=0.35)
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        record = _get_agent_track_record("bull", "stressed")
    assert "below" in record.lower() or "calibrate" in record.lower() or \
           "50%" in record or "down" in record.lower(), \
        "Should warn when accuracy < 50%"


def test_track_record_empty_string_when_no_data(tmp_path):
    from debate.agents import _get_agent_track_record
    cred_path = tmp_path / "nonexistent_cred.json"
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        record = _get_agent_track_record("bull", "stressed")
    assert record == "", "Must return empty string (not crash) when no credibility data"


def test_run_bull_agent_system_prompt_includes_track_record(tmp_path):
    """The system prompt sent to the LLM must include track record text."""
    import debate.agents as agents_mod
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, bull_stressed=0.35)

    captured_prompts = []
    def mock_generate(system_prompt, user_prompt, **kwargs):
        captured_prompts.append(system_prompt)
        return '{"verdict": "proceed", "confidence": 0.7, "reasoning": "test"}'

    portfolio_state = {
        "date": "2026-05-03", "us_regime": "stressed", "macro_regime": "stressed",
        "n_positions": 5, "allocation": {}, "weights": {"AAPL": 0.20, "MSFT": 0.20},
    }
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        with patch("debate.agents.generate_structured", side_effect=mock_generate):
            agents_mod.run_bull_agent(portfolio_state)

    assert len(captured_prompts) > 0
    combined = " ".join(captured_prompts)
    assert "35" in combined or "track" in combined.lower() or "accuracy" in combined.lower(), \
        "System prompt must include agent track record"
