# tests/test_narrative_alpha.py
"""
Tests for ascent/alpha/narrative_alpha.py
"""
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_fund_cache(symbols=None, n_quarters=2):
    """
    Build a minimal llm_fundamental_cache.json-like dict.
    Each entry: {symbol}_{date} → {direction, confidence, key_trend, uncertainty}.
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG"]
    cache = {}
    base_dates = ["2026-01-15", "2025-10-15", "2025-07-15"]
    directions = ["UP", "DOWN", "NEUTRAL"]
    for sym in symbols:
        for i in range(min(n_quarters, len(base_dates))):
            key = f"{sym}_{base_dates[i]}"
            cache[key] = {
                "direction": directions[i % len(directions)],
                "confidence": 0.70 - i * 0.05,
                "key_trend": f"Strong revenue growth Q{i}" if i == 0 else f"Margin pressure Q{i}",
                "uncertainty": "Competition risk",
            }
    return cache


# ── test_build_narrative_alpha_empty_cache ─────────────────────────────────────

def test_build_narrative_alpha_empty_cache(tmp_path):
    """No cache file → returns all-zeros Series for given symbols."""
    from ascent.alpha.narrative_alpha import build_narrative_alpha

    non_existent = tmp_path / "llm_fundamental_cache.json"
    narr_cache   = tmp_path / "narrative_shift_cache.json"

    with patch("ascent.alpha.narrative_alpha.LLM_FUND_CACHE_PATH", non_existent):
        with patch("ascent.alpha.narrative_alpha.NARRATIVE_CACHE_PATH", narr_cache):
            result = build_narrative_alpha(["AAPL", "MSFT", "GOOG"])

    assert isinstance(result, pd.Series)
    assert set(result.index) == {"AAPL", "MSFT", "GOOG"}
    assert (result == 0.0).all(), "No cache → all scores must be zero"


# ── test_build_narrative_alpha_single_entry ────────────────────────────────────

def test_build_narrative_alpha_single_entry(tmp_path):
    """One entry per symbol → no prior quarter → all zeros."""
    from ascent.alpha.narrative_alpha import build_narrative_alpha

    cache = {
        "AAPL_2026-01-15": {"direction": "UP", "confidence": 0.80,
                             "key_trend": "Strong revenue", "uncertainty": "Macro"},
        "MSFT_2026-01-15": {"direction": "DOWN", "confidence": 0.75,
                             "key_trend": "Margin pressure", "uncertainty": "Competition"},
    }
    fund_cache_file = tmp_path / "llm_fundamental_cache.json"
    fund_cache_file.write_text(json.dumps(cache))
    narr_cache_file = tmp_path / "narrative_shift_cache.json"

    with patch("ascent.alpha.narrative_alpha.LLM_FUND_CACHE_PATH", fund_cache_file):
        with patch("ascent.alpha.narrative_alpha.NARRATIVE_CACHE_PATH", narr_cache_file):
            result = build_narrative_alpha(["AAPL", "MSFT"])

    assert isinstance(result, pd.Series)
    assert (result == 0.0).all(), "Single entry → no prior → all zeros"


# ── test_build_narrative_alpha_two_entries ─────────────────────────────────────

def test_build_narrative_alpha_two_entries(tmp_path):
    """
    Two entries per symbol + mocked _compute_shift → z-scored result returned.
    """
    from ascent.alpha.narrative_alpha import build_narrative_alpha

    symbols = ["AAPL", "MSFT", "GOOG"]
    cache = _make_fund_cache(symbols=symbols, n_quarters=2)
    fund_cache_file = tmp_path / "llm_fundamental_cache.json"
    fund_cache_file.write_text(json.dumps(cache))
    narr_cache_file = tmp_path / "narrative_shift_cache.json"

    # Mock _compute_shift to return deterministic values
    raw_shifts = {"AAPL": 0.8, "MSFT": -0.4, "GOOG": 0.0}

    def mock_shift(symbol, current, prior):
        return raw_shifts.get(symbol, 0.0)

    with patch("ascent.alpha.narrative_alpha.LLM_FUND_CACHE_PATH", fund_cache_file):
        with patch("ascent.alpha.narrative_alpha.NARRATIVE_CACHE_PATH", narr_cache_file):
            with patch("ascent.alpha.narrative_alpha._compute_shift", side_effect=mock_shift):
                result = build_narrative_alpha(symbols)

    assert isinstance(result, pd.Series)
    assert set(result.index) == set(symbols)
    # Z-scored: mean ~0, std ~1 (within small tolerance for n=3)
    assert abs(result.mean()) < 0.5
    # AAPL should be highest, MSFT lowest
    assert result["AAPL"] > result["GOOG"] > result["MSFT"]
    # Values should be clipped to [-3, 3]
    assert (result.abs() <= 3.0).all()


# ── test_compute_shift_returns_float_on_failure ────────────────────────────────

def test_compute_shift_returns_float_on_failure(tmp_path):
    """If generate_structured raises, _compute_shift returns 0.0."""
    from ascent.alpha.narrative_alpha import _compute_shift

    narr_cache_file = tmp_path / "narrative_shift_cache.json"

    current = {"direction": "UP",   "confidence": 0.80, "key_trend": "Strong growth", "uncertainty": "Macro"}
    prior   = {"direction": "DOWN", "confidence": 0.60, "key_trend": "Margin squeeze", "uncertainty": "Costs"}

    with patch("ascent.alpha.narrative_alpha.NARRATIVE_CACHE_PATH", narr_cache_file):
        with patch("ascent.alpha.narrative_alpha.generate_structured",
                   side_effect=RuntimeError("API unavailable")):
            result = _compute_shift("AAPL", current, prior)

    assert isinstance(result, float)
    assert result == 0.0


# ── test_get_symbol_analyses_sorts_newest_first ────────────────────────────────

def test_get_symbol_analyses_sorts_newest_first(tmp_path):
    """Two entries for the same symbol — verify newest date appears first."""
    from ascent.alpha.narrative_alpha import _get_symbol_analyses

    fund_cache = {
        "AAPL_2025-10-15": {"direction": "DOWN", "confidence": 0.70,
                             "key_trend": "Margin pressure", "uncertainty": "Costs"},
        "AAPL_2026-01-15": {"direction": "UP",   "confidence": 0.80,
                             "key_trend": "Strong revenue",  "uncertainty": "Macro"},
        "MSFT_2026-01-15": {"direction": "UP",   "confidence": 0.75,
                             "key_trend": "Cloud growth",    "uncertainty": "Competition"},
    }

    analyses = _get_symbol_analyses("AAPL", fund_cache)

    assert len(analyses) == 2
    # Newest first
    assert analyses[0]["date"] == "2026-01-15"
    assert analyses[1]["date"] == "2025-10-15"
    # Correct analysis attached
    assert analyses[0]["analysis"]["direction"] == "UP"
    assert analyses[1]["analysis"]["direction"] == "DOWN"
    # MSFT not included
    for entry in analyses:
        assert "MSFT" not in str(entry)


# ── test_narrative_cache_prevents_duplicate_llm_calls ─────────────────────────

def test_narrative_cache_prevents_duplicate_llm_calls(tmp_path):
    """Second call for the same symbol pair reads from narrative cache, not LLM."""
    from ascent.alpha.narrative_alpha import _compute_shift

    narr_cache_file = tmp_path / "narrative_shift_cache.json"
    current = {"direction": "UP",   "confidence": 0.80, "key_trend": "Strong growth", "uncertainty": "Macro"}
    prior   = {"direction": "DOWN", "confidence": 0.60, "key_trend": "Margin squeeze", "uncertainty": "Costs"}

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        return '{"shift": 0.65, "reason": "Improved from DOWN to UP."}'

    with patch("ascent.alpha.narrative_alpha.NARRATIVE_CACHE_PATH", narr_cache_file):
        with patch("ascent.alpha.narrative_alpha.generate_structured", side_effect=mock_generate):
            shift1 = _compute_shift("AAPL", current, prior)
            shift2 = _compute_shift("AAPL", current, prior)

    assert call_count[0] == 1, "Second call must use cache, not re-invoke LLM"
    assert abs(shift1 - 0.65) < 1e-6
    assert abs(shift2 - 0.65) < 1e-6


# ── test_narrative_in_stack_weights ───────────────────────────────────────────

def test_narrative_in_stack_default_weights():
    """narrative sleeve must be present in DEFAULT_ALPHA_WEIGHTS at 0.03."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert "narrative" in DEFAULT_ALPHA_WEIGHTS, \
        "DEFAULT_ALPHA_WEIGHTS must include 'narrative' sleeve"
    assert abs(DEFAULT_ALPHA_WEIGHTS["narrative"] - 0.03) < 0.001, \
        "narrative sleeve activated at 3% (cache seeder built in Task #13)"


def test_system_prompt_contains_grounding_instruction():
    """System prompt must instruct model to reason only from provided summaries."""
    from ascent.alpha.narrative_alpha import _SYSTEM_PROMPT
    lowered = _SYSTEM_PROMPT.lower()
    assert ("only" in lowered or "provided" in lowered), \
        "System prompt must restrict model to the provided data"
    assert ("training" in lowered or "outside" in lowered or "do not" in lowered), \
        "System prompt must forbid using external knowledge"


def test_compute_shift_uses_json_schema():
    """_compute_shift must pass json_schema to generate_structured."""
    import ascent.alpha.narrative_alpha as mod
    from unittest.mock import patch

    calls = []

    def mock_generate(system_prompt, user_prompt, **kwargs):
        calls.append(kwargs)
        return '{"shift": 0.5, "reason": "direction improved"}'

    current = {"direction": "UP",   "confidence": 0.8, "key_trend": "improving margins"}
    prior   = {"direction": "DOWN", "confidence": 0.6, "key_trend": "declining margins"}

    with patch.object(mod, "generate_structured", mock_generate):
        with patch.object(mod, "_load_narrative_cache", return_value={}):
            with patch.object(mod, "_save_narrative_cache", return_value=None):
                mod._compute_shift("AAPL", current, prior)

    assert any("json_schema" in c for c in calls), \
        "_compute_shift must pass json_schema= to generate_structured"
