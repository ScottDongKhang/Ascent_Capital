# tests/agents/test_ai_pm_prethesis_fixes.py
"""
Tests for Findings 6, 7, 8 fixes:
  - Finding 6: _prethesis_universe defined from portfolio + alpha scores
  - Finding 7: AIPreThesis has conviction_reasons + sector_thesis + Phase 1→2 handoff works
  - Finding 8: directional_stance required in schema, rendered first, enforced in _tool_propose_portfolio
"""
from __future__ import annotations

import json
import sys
import types
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_agent():
    """Import ai_pm_agent, stubbing heavy optional deps if needed."""
    # Stub heavy deps that may not be installed in test env
    for mod_name in ["yfinance", "pandas", "sklearn", "anthropic"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
    # Stub ascent.llm.client so the import itself doesn't fail
    llm_stub = types.ModuleType("ascent.llm.client")
    llm_stub.tool_completion = MagicMock()
    llm_stub.DEFAULT_MODEL = "claude-opus-4-6"
    llm_stub.SONNET_MODEL = "claude-sonnet-4-6"
    llm_stub.HAIKU_MODEL = "claude-haiku-4-5-20251001"
    sys.modules.setdefault("ascent.llm.client", llm_stub)
    sys.modules.setdefault("ascent.llm", types.ModuleType("ascent.llm"))
    import importlib
    import agents.ai_pm_agent as m
    importlib.reload(m)
    return m


# ── Test 1: _build_data_grounding returns news even with empty symbols ────────

def test_build_data_grounding_returns_news_when_no_symbols():
    """Fix 1 (Finding 6): When symbols=[] but news_context is provided, return the news block."""
    m = _import_agent()
    result = m._build_data_grounding(
        symbols=[],
        news_context={"AAPL": ["Apple beats estimates by 12%", "New iPhone cycle strong"]},
    )
    assert "AAPL" in result, "News context should appear even with empty symbols list"
    assert "Apple beats estimates" in result


def test_build_data_grounding_empty_symbols_no_news_returns_empty():
    """When both symbols and news_context are empty, return empty string."""
    m = _import_agent()
    result = m._build_data_grounding(symbols=[], news_context=None)
    assert result == ""


# ── Test 2: AIPreThesis has conviction_reasons, sector_thesis, directional_stance ──

def test_aiprethesis_has_conviction_reasons_field():
    """Fix 2 (Finding 7): AIPreThesis dataclass has conviction_reasons defaulting to []."""
    m = _import_agent()
    pt = m.AIPreThesis(
        macro_view="test macro",
        regime_interpretation="calm_bull",
        high_conviction_names=[],
        names_to_avoid=[],
        sector_tilts=[],
    )
    assert hasattr(pt, "conviction_reasons"), "AIPreThesis must have conviction_reasons field"
    assert pt.conviction_reasons == []


def test_aiprethesis_has_sector_thesis_field():
    """Fix 2 (Finding 7): AIPreThesis dataclass has sector_thesis defaulting to []."""
    m = _import_agent()
    pt = m.AIPreThesis(
        macro_view="test macro",
        regime_interpretation="calm_bull",
        high_conviction_names=[],
        names_to_avoid=[],
        sector_tilts=[],
    )
    assert hasattr(pt, "sector_thesis"), "AIPreThesis must have sector_thesis field"
    assert pt.sector_thesis == []


def test_aiprethesis_has_directional_stance_field():
    """Fix 3 (Finding 8): AIPreThesis dataclass has directional_stance defaulting to {}."""
    m = _import_agent()
    pt = m.AIPreThesis(
        macro_view="test macro",
        regime_interpretation="calm_bull",
        high_conviction_names=[],
        names_to_avoid=[],
        sector_tilts=[],
    )
    assert hasattr(pt, "directional_stance"), "AIPreThesis must have directional_stance field"
    assert pt.directional_stance == {}


# ── Test 3: directional_stance required in schema ─────────────────────────────

def test_directional_stance_in_prethesis_schema_required():
    """Fix 3 (Finding 8): directional_stance must be in the required list of _PROPOSE_PRETHESIS_TOOL."""
    m = _import_agent()
    required = m._PROPOSE_PRETHESIS_TOOL["input_schema"]["required"]
    assert "directional_stance" in required, (
        f"directional_stance must be required in _PROPOSE_PRETHESIS_TOOL schema, got: {required}"
    )


def test_directional_stance_in_prethesis_schema_properties():
    """directional_stance must be in properties with correct sub-fields."""
    m = _import_agent()
    props = m._PROPOSE_PRETHESIS_TOOL["input_schema"]["properties"]
    assert "directional_stance" in props
    ds_required = props["directional_stance"].get("required", [])
    for expected in ["direction", "thesis", "upside_case_pct", "downside_case_pct", "falsifier", "horizon_days"]:
        assert expected in ds_required, f"directional_stance.required must include '{expected}'"


# ── Test 4: _format_prethesis_for_prompt renders stance first ─────────────────

def test_format_prethesis_renders_directional_stance_first():
    """Fix 3 (Finding 8): _format_prethesis_for_prompt should start with DIRECTIONAL STANCE."""
    m = _import_agent()
    stance = {
        "direction": "risk_on",
        "thesis": "Fed pivot drives equity re-rating",
        "upside_case_pct": 5.0,
        "downside_case_pct": 3.0,
        "falsifier": "SPY closes below 480",
        "horizon_days": 21,
    }
    pt = m.AIPreThesis(
        macro_view="Equities bullish",
        regime_interpretation="calm_bull",
        high_conviction_names=[],
        names_to_avoid=[],
        sector_tilts=[],
        directional_stance=stance,
    )
    rendered = m._format_prethesis_for_prompt(pt)
    first_line = rendered.strip().splitlines()[0]
    assert first_line.startswith("DIRECTIONAL STANCE: RISK_ON"), (
        f"First line should start with 'DIRECTIONAL STANCE: RISK_ON', got: {first_line!r}"
    )
    assert "SPY closes below 480" in rendered
    assert "Fed pivot" in rendered


# ── Test 5: _tool_propose_portfolio rejects missing prethesis_disposition ──────

def test_tool_propose_portfolio_rejects_missing_disposition(tmp_path):
    """Fix 3 (Finding 8): When this session sealed a pre-thesis (prethesis_active=True),
    submission without prethesis_disposition must be rejected. The gate is
    session-scoped — a stale ai_prethesis_latest.json from a previous run must
    NOT trigger it (that caused spurious rejections)."""
    m = _import_agent()

    # Temporarily monkey-patch _REPO_ROOT to our tmp dir
    orig_root = m._REPO_ROOT
    m._REPO_ROOT = tmp_path
    (tmp_path / "data_cache").mkdir(parents=True, exist_ok=True)

    try:
        result_store: list = []
        # No prethesis_disposition in thesis
        result = m._tool_propose_portfolio(
            inputs={"weights": {"AAPL": 0.1}, "thesis": {"feedback_acknowledged": True}},
            result_store=result_store,
            prethesis_active=True,
        )
        assert isinstance(result, str) and "REJECTED" in result, (
            f"Should be rejected for missing prethesis_disposition, got: {result!r}"
        )
        assert len(result_store) == 0, "result_store should remain empty on rejection"
    finally:
        m._REPO_ROOT = orig_root


def test_tool_propose_portfolio_accepts_valid_disposition(tmp_path):
    """_tool_propose_portfolio should proceed when prethesis_disposition is FOLLOWED."""
    m = _import_agent()

    orig_root = m._REPO_ROOT
    m._REPO_ROOT = tmp_path
    (tmp_path / "data_cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data_cache" / "ai_prethesis_latest.json").write_text(
        json.dumps({"macro_view": "test", "directional_stance": {}})
    )
    # No feedback file → feedback gate won't trigger
    # No guardrails import needed — stub it
    import ascent.strategy.ai_pm_guardrails as _guard
    orig_check = _guard.check_conviction_inflation
    _guard.check_conviction_inflation = lambda x: x

    # Stub calibration tracker
    import ascent.strategy.calibration_tracker as _cal
    orig_log = getattr(_cal, "log_prediction", None)
    _cal.log_prediction = lambda *a, **kw: None

    try:
        result_store: list = []
        result = m._tool_propose_portfolio(
            inputs={
                "weights": {"AAPL": 0.1},
                "thesis": {
                    "feedback_acknowledged": True,
                    "prethesis_disposition": "FOLLOWED",
                },
            },
            result_store=result_store,
        )
        assert "Portfolio submitted" in result or len(result_store) > 0, (
            f"Should accept FOLLOWED disposition, got: {result!r}"
        )
    finally:
        m._REPO_ROOT = orig_root
        _guard.check_conviction_inflation = orig_check
        if orig_log is not None:
            _cal.log_prediction = orig_log
