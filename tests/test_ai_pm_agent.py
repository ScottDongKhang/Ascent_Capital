# tests/test_ai_pm_agent.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── pm_risk_validator ──────────────────────────────────────────────────────────

def test_validator_accepts_clean_portfolio():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.08, "MSFT": 0.07, "GOOG": 0.06, "AMZN": 0.06,
        "META": 0.07, "NVDA": 0.08, "TSLA": 0.05, "JPM": 0.07,
        "V": 0.06, "UNH": 0.06,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is True
    assert violations == []


def test_validator_rejects_concentrated_position():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.50, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("AAPL" in v for v in violations)


def test_validator_rejects_too_few_positions():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {"AAPL": 0.50, "MSFT": 0.50}
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("positions" in v.lower() for v in violations)


def test_validator_rejects_empty_portfolio():
    from ascent.risk.pm_risk_validator import validate
    ok, violations = validate({})
    assert ok is False


def test_validator_rejects_distressed_name():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10, "BAD": 0.10, "JPM": 0.08,
        "V": 0.07, "UNH": 0.07, "GS": 0.08,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=["BAD"]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("BAD" in v for v in violations)


def test_validator_rejects_sector_overweight():
    from ascent.risk.pm_risk_validator import validate
    sector_map = {
        "AAPL": "tech", "MSFT": "tech", "GOOG": "tech", "AMZN": "tech",
        "META": "tech", "NVDA": "tech",
        "JPM": "finance", "V": "finance", "GS": "finance", "WFC": "finance",
    }
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10,
        "JPM": 0.10, "V": 0.10, "GS": 0.10, "WFC": 0.10,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value=sector_map):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("tech" in v.lower() for v in violations)


def test_validator_rejects_negative_weight():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "BAD": -0.05,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    # Only the negative-weight violation — no spurious position-limit violations
    assert any("BAD" in v for v in violations)
    assert not any("exceeds max" in v for v in violations)


# ── earned_authority ──────────────────────────────────────────────────────────

def _make_state(phase=0, ai_weight=0.0, ai_returns=None, qt_returns=None, reverts=0):
    return {
        "ai_weight": ai_weight,
        "phase": phase,
        "phase_start_date": "2026-05-15",
        "ai_returns_21d": ai_returns or [],
        "quant_returns_21d": qt_returns or [],
        "auto_revert_count": reverts,
        "last_updated": "2026-05-15",  # Yesterday, so today's update will process
    }


def test_shadow_phase_blend_returns_pure_quant():
    """At ai_weight=0, blend returns quant portfolio unchanged (only quant names survive)."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=0, ai_weight=0.0)))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                quant = {"AAPL": 0.50, "MSFT": 0.50}
                ai = {"GOOG": 0.60, "AMZN": 0.40}
                result = ea.blend(ai, quant)
    assert "AAPL" in result
    assert "MSFT" in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_union_of_positions():
    """At ai_weight=0.5, both portfolios contribute."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=2, ai_weight=0.5)))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                ai = {"VICR": 0.10, "AMKR": 0.08}
                quant = {"VICR": 0.06, "FIX": 0.07}
                result = ea.blend(ai, quant)
    assert "VICR" in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_min_weight_filter():
    """Positions below 0.02 after blending are dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=0, ai_weight=0.1)))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                ai = {"TINY": 0.05}   # 0.1 * 0.05 = 0.005 < 0.02
                quant = {"AAPL": 0.50, "MSFT": 0.50}
                result = ea.blend(ai, quant)
    assert "TINY" not in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_renormalizes_to_1():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=1, ai_weight=0.25)))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                ai = {f"A{i}": 0.1 for i in range(5)}
                quant = {f"Q{i}": 0.1 for i in range(5)}
                result = ea.blend(ai, quant)
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_authority_advances_after_edge():
    """After 21 days with AI Sharpe > quant+0.05, phase advances 0→1."""
    ai_returns = [0.002, 0.0015, 0.0025] * 7  # Variance present, AI > QT
    qt_returns  = [0.001, 0.0005, 0.0015] * 7
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state = _make_state(phase=0, ai_weight=0.0, ai_returns=ai_returns[:20], qt_returns=qt_returns[:20])
        state_path.write_text(json.dumps(state))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                result = ea.update_authority(ai_returns[-1], qt_returns[-1])
    assert result["phase"] == 1
    assert result["ai_weight"] == 0.25


def test_authority_stays_if_no_edge():
    """With equal Sharpe, phase stays at 0."""
    ai_returns = [0.001] * 21
    qt_returns  = [0.001] * 21
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state = _make_state(phase=0, ai_weight=0.0, ai_returns=ai_returns[:20], qt_returns=qt_returns[:20])
        state_path.write_text(json.dumps(state))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                result = ea.update_authority(ai_returns[-1], qt_returns[-1])
    assert result["phase"] == 0
    assert result["ai_weight"] == 0.0


def test_auto_revert_on_drawdown():
    """AI drawdown > quant+5% at phase>0 reverts to phase 0."""
    # 9 bad returns in buffer, trigger on the 10th (matches ADVANCE_WINDOW=10)
    ai_returns = [-0.03] * 9
    qt_returns  = [0.001] * 9
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state = _make_state(phase=2, ai_weight=0.5, ai_returns=ai_returns, qt_returns=qt_returns)
        state_path.write_text(json.dumps(state))
        shadow_path = Path(tmp) / "shadow.jsonl"
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", shadow_path):
                import ascent.strategy.earned_authority as ea
                result = ea.update_authority(-0.03, 0.001)
    assert result["phase"] == 0
    assert result["ai_weight"] == 0.0
    assert result["auto_revert_count"] == 1


def test_hard_cap_at_0_80():
    """PHASE_WEIGHTS never exceeds HARD_CAP."""
    import ascent.strategy.earned_authority as ea
    assert ea.HARD_CAP == 0.80
    assert all(w <= ea.HARD_CAP for w in ea.PHASE_WEIGHTS)
    assert max(ea.PHASE_WEIGHTS) == 0.75  # phase schedule tops at 75%


# ── thesis_formatter ──────────────────────────────────────────────────────────

def test_format_thesis_fills_missing_fields():
    """Missing keys get filled with schema defaults; as_of_date is injected."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ai_pm_theses"
        with patch("ascent.strategy.thesis_formatter.OUTPUT_DIR", out_dir):
            from ascent.strategy.thesis_formatter import format_thesis
            result = format_thesis({"market_view": "Calm bull, credit spreads tight."})
    assert result["market_view"] == "Calm bull, credit spreads tight."
    assert result["quant_overrides"] == []     # default filled
    assert result["quant_agreement"] == []     # default filled
    assert result["key_risks"] == []           # default filled
    assert "as_of_date" in result


def test_thesis_to_plaintext_returns_non_empty_string():
    from ascent.strategy.thesis_formatter import thesis_to_plaintext
    thesis = {
        "market_view": "Credit spreads are widening.",
        "regime_assessment": "calm_bull, confidence 0.73",
        "ai_pm_portfolio": {"VICR": 0.06, "AMKR": 0.05, "FIX": 0.07},
        "quant_agreement": ["VICR", "FIX"],
        "quant_overrides": [{"symbol": "VAL", "ai_action": "exclude"}],
        "key_risks": ["Macro: Fed surprise", "Idio: VICR miss"],
    }
    result = thesis_to_plaintext(thesis)
    assert isinstance(result, str)
    assert len(result) > 20


# ── ai_pm_agent ───────────────────────────────────────────────────────────────

def test_fallback_on_no_propose_portfolio_call():
    """If the tool loop exits without calling propose_portfolio, return fallback."""
    from unittest.mock import patch

    def fake_tool_completion(system_prompt, user_prompt, tools, tool_executor,
                             model, max_tokens, max_tool_calls):
        return "I forgot to submit."

    with patch("agents.ai_pm_agent.tool_completion", fake_tool_completion):
        from agents.ai_pm_agent import run_ai_pm
        result = run_ai_pm()

    assert result.fallback is True
    assert result.portfolio == {}


def test_tool_executor_never_raises():
    """All tools with bad inputs return strings, never raise."""
    from agents.ai_pm_agent import _make_executor
    result_store = []
    executor = _make_executor(result_store)

    bad_inputs = [
        ("get_regime_state", {}),
        ("get_macro_data", {}),
        ("run_quant_agent", {"agent_id": "nonexistent_agent"}),
        ("get_sec_signal", {"symbol": "FAKESYM"}),
        ("get_transcript_signal", {"symbol": "FAKESYM"}),
        ("get_attribution_history", {"symbol": "FAKESYM"}),
        ("get_earnings_signal", {"symbol": "FAKESYM"}),
        ("get_past_verdicts", {"regime": "nonexistent_regime"}),
        ("get_factor_exposures", {"weights": {"FAKE": 1.0}}),
        ("get_var_estimate", {"weights": {"FAKE": 1.0}}),
        ("get_sector_concentration", {"weights": {"FAKE": 1.0}}),
        ("get_position_momentum", {"symbols": ["FAKESYM"]}),
        ("propose_portfolio", {"weights": {}, "thesis": {}}),
        ("completely_unknown_tool", {}),
    ]
    for tool_name, inputs in bad_inputs:
        result = executor(tool_name, inputs)
        assert isinstance(result, str), f"Tool {tool_name} returned {type(result)}, expected str"
