# tests/test_reflection_agent.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def _write_scored_verdict(path: Path, regime="stressed", recommendation="proceed",
                          nav_change=-0.032, bull_text="strong momentum", bear_text="credit spreads widening"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "date": "2026-04-10",
        "outcome_scored": True,
        "outcome_nav_change": nav_change,
        "outcome_score": 0.0 if nav_change < -0.01 else 1.0,
        "verdict": {"recommendation": recommendation, "reasoning": "base case holds"},
        "portfolio_state": {"us_regime": regime, "weights": {"AAPL": 0.15, "MSFT": 0.10}},
        "arguments": {"bull": bull_text, "bear": bear_text, "devils_advocate": "liquidity gap"},
    }
    path.write_text(json.dumps(data))
    return data


def test_reflect_returns_dict(tmp_path):
    from memory.reflection_agent import reflect_on_verdict
    vpath = tmp_path / "verdict_2026-04-10.json"
    _write_scored_verdict(vpath)

    def mock_generate(system_prompt, user_prompt, **kwargs):
        return json.dumps({
            "lesson": "Ignored credit spreads in stressed regime",
            "key_error": "Bull underweighted macro headwinds",
            "confidence_calibration": "DOWN",
            "regime": "stressed",
        })

    with patch("memory.reflection_agent._call_llm", side_effect=mock_generate):
        result = reflect_on_verdict(vpath)

    assert isinstance(result, dict)
    for key in ["lesson", "key_error", "confidence_calibration", "regime", "date"]:
        assert key in result, f"Missing key: {key}"


def test_reflect_skips_unscored_verdict(tmp_path):
    from memory.reflection_agent import reflect_on_verdict
    vpath = tmp_path / "verdict_2026-04-10.json"
    data = {"date": "2026-04-10", "outcome_scored": False, "verdict": {}, "portfolio_state": {}, "arguments": {}}
    vpath.write_text(json.dumps(data))
    result = reflect_on_verdict(vpath)
    assert result is None, "Must return None for unscored verdicts"


def test_reflect_on_new_outcomes_processes_new_files(tmp_path):
    from memory.reflection_agent import reflect_on_new_outcomes
    debate_dir = tmp_path / "debate_log"
    refl_path  = tmp_path / "reflections.jsonl"
    vpath = debate_dir / "verdict_2026-04-10.json"
    _write_scored_verdict(vpath)

    def mock_generate(system_prompt, user_prompt, **kwargs):
        return json.dumps({"lesson": "Test", "key_error": "None", "confidence_calibration": "HOLD", "regime": "stressed"})

    with patch("memory.reflection_agent.DEBATE_LOG_DIR", debate_dir):
        with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
            with patch("memory.reflection_agent._call_llm", side_effect=mock_generate):
                count = reflect_on_new_outcomes()

    assert count >= 1
    assert refl_path.exists()
    lines = [json.loads(l) for l in refl_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    assert "lesson" in lines[0]


def test_reflect_not_processed_twice(tmp_path):
    from memory.reflection_agent import reflect_on_new_outcomes
    debate_dir = tmp_path / "debate_log"
    refl_path  = tmp_path / "reflections.jsonl"
    vpath = debate_dir / "verdict_2026-04-10.json"
    _write_scored_verdict(vpath)
    call_count = [0]

    def mock_generate(system_prompt, user_prompt, **kwargs):
        call_count[0] += 1
        return json.dumps({"lesson": "T", "key_error": "N", "confidence_calibration": "HOLD", "regime": "stressed"})

    with patch("memory.reflection_agent.DEBATE_LOG_DIR", debate_dir):
        with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
            with patch("memory.reflection_agent._call_llm", side_effect=mock_generate):
                reflect_on_new_outcomes()
                first_count = call_count[0]
                reflect_on_new_outcomes()
                second_count = call_count[0]

    assert second_count == first_count, "Second run must not re-process already-reflected verdicts"


def test_load_recent_reflections_filters_by_regime(tmp_path):
    from memory.reflection_agent import load_recent_reflections
    refl_path = tmp_path / "reflections.jsonl"
    rows = [
        {"date": "2026-04-01", "regime": "stressed",  "lesson": "A", "key_error": "", "confidence_calibration": "DOWN"},
        {"date": "2026-04-02", "regime": "calm_bull", "lesson": "B", "key_error": "", "confidence_calibration": "UP"},
        {"date": "2026-04-03", "regime": "stressed",  "lesson": "C", "key_error": "", "confidence_calibration": "HOLD"},
    ]
    refl_path.write_text("\n".join(json.dumps(r) for r in rows))

    with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
        results = load_recent_reflections(regime="stressed", n=5)

    assert len(results) == 2
    assert all(r["regime"] == "stressed" for r in results)


def test_format_reflections_for_context_returns_string(tmp_path):
    from memory.reflection_agent import load_recent_reflections, format_reflections_for_context
    refl_path = tmp_path / "reflections.jsonl"
    refl_path.write_text(json.dumps({
        "date": "2026-04-01", "regime": "stressed",
        "lesson": "Ignored credit spreads", "key_error": "Bull wrong",
        "confidence_calibration": "DOWN",
    }))
    with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
        refs = load_recent_reflections(regime="stressed", n=3)
    ctx = format_reflections_for_context(refs)
    assert isinstance(ctx, str)
    assert len(ctx) > 0
    assert "stressed" in ctx.lower() or "lesson" in ctx.lower()
