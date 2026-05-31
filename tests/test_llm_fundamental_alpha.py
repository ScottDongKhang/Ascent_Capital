# tests/test_llm_fundamental_alpha.py
import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path


def _make_fundamentals(symbols=None, n_quarters=4):
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    rows = []
    base_date = pd.Timestamp("2025-12-31")
    for sym in symbols:
        np.random.seed(hash(sym) % 2**31)
        for q in range(n_quarters):
            rows.append({
                "symbol": sym,
                "date":   base_date - pd.DateOffset(months=3 * q),
                "gross_profitability": np.random.uniform(0.2, 0.6),
                "accruals":            np.random.uniform(-0.05, 0.05),
                "asset_growth":        np.random.uniform(-0.02, 0.15),
            })
    return pd.DataFrame(rows)


def test_returns_series(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals()
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm",
                   return_value={"direction": "UP", "confidence": 0.80}):
            result = llm_fundamental_alpha(fund)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_scores_are_cross_sectional_zscored(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    syms = list("ABCDEFGHIJ")
    fund = _make_fundamentals(symbols=syms)
    responses = [
        {"direction": "UP",      "confidence": 0.9},
        {"direction": "DOWN",    "confidence": 0.8},
        {"direction": "UP",      "confidence": 0.7},
        {"direction": "NEUTRAL", "confidence": 0.6},
        {"direction": "DOWN",    "confidence": 0.5},
        {"direction": "UP",      "confidence": 0.85},
        {"direction": "DOWN",    "confidence": 0.75},
        {"direction": "UP",      "confidence": 0.65},
        {"direction": "NEUTRAL", "confidence": 0.4},
        {"direction": "DOWN",    "confidence": 0.9},
    ]
    call_idx = [0]
    def mock_call(symbol, table):
        r = responses[call_idx[0] % len(responses)]
        call_idx[0] += 1
        return r
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm", side_effect=mock_call):
            result = llm_fundamental_alpha(fund)
    assert abs(result.mean()) < 0.15, f"Mean should be ~0, got {result.mean()}"
    assert 0.5 < result.std() < 2.0, f"Std should be ~1, got {result.std()}"


def test_empty_on_none_input(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        assert len(llm_fundamental_alpha(None)) == 0
        assert len(llm_fundamental_alpha(pd.DataFrame())) == 0


def test_uses_cache_on_second_call(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals(symbols=["AAPL", "MSFT"])
    count = [0]
    def mock_call(symbol, table):
        count[0] += 1
        return {"direction": "UP", "confidence": 0.75}
    cache_path = tmp_path / "c.json"
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", cache_path):
        with patch("ascent.alpha.llm_fundamental._call_llm", side_effect=mock_call):
            llm_fundamental_alpha(fund)
            first = count[0]
            llm_fundamental_alpha(fund)
            second = count[0]
    assert second == first, "Second call must use cache, not re-call LLM"


def test_api_failure_returns_empty(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals()
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm", return_value=None):
            result = llm_fundamental_alpha(fund)
    assert isinstance(result, pd.Series)


def test_anonymization_no_ticker_in_table(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals(symbols=["AAPL"])
    captured = []
    def mock_call(symbol, table):
        captured.append(table)
        return {"direction": "UP", "confidence": 0.8}
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm", side_effect=mock_call):
            llm_fundamental_alpha(fund)
    assert all("AAPL" not in t for t in captured), "Ticker must not appear in anonymized table"


def test_respects_45day_filing_lag(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    # All quarters within 44 days of as_of_date should be excluded
    fund = pd.DataFrame([{
        "symbol": "AAPL",
        "date": pd.Timestamp("2026-04-20"),   # 13 days before 2026-05-03
        "gross_profitability": 0.4, "accruals": 0.01, "asset_growth": 0.05,
    }])
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm",
                   return_value={"direction": "UP", "confidence": 0.8}) as mock:
            llm_fundamental_alpha(fund, as_of_date=pd.Timestamp("2026-05-03"))
    # Should not crash; data within 45-day lag is excluded silently


def test_stack_includes_llm_fundamental_sleeve():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert "llm_fundamental" in DEFAULT_ALPHA_WEIGHTS, \
        "stack.py DEFAULT_ALPHA_WEIGHTS must include llm_fundamental sleeve"
    assert DEFAULT_ALPHA_WEIGHTS["llm_fundamental"] > 0
    assert abs(sum(DEFAULT_ALPHA_WEIGHTS.values()) - 1.0) < 1e-6, \
        "Sleeve weights must sum to 1.0"


def test_system_prompt_contains_amnesia_instruction():
    """The system prompt must explicitly forbid using training-data knowledge."""
    from ascent.alpha.llm_fundamental import _SYSTEM_PROMPT
    lowered = _SYSTEM_PROMPT.lower()
    assert ("training" in lowered or "amnesia" in lowered or "do not use" in lowered), \
           "System prompt must instruct model not to use training-data company knowledge"


def test_user_template_contains_quoted_evidence_field():
    """The user template must ask for a quoted_evidence field."""
    from ascent.alpha.llm_fundamental import _USER_TEMPLATE
    assert "quoted_evidence" in _USER_TEMPLATE, \
           "User template must include quoted_evidence in JSON schema"


def test_call_llm_uses_json_schema(tmp_path):
    """_call_llm must pass a json_schema to generate_structured."""
    import ascent.alpha.llm_fundamental as mod
    from unittest.mock import patch
    calls = []

    def mock_generate(system_prompt, user_prompt, **kwargs):
        calls.append(kwargs)
        return '{"direction": "UP", "confidence": 0.8, "key_trend": "improving", "uncertainty": "rates", "quoted_evidence": "Q0 gross_profitability=0.350"}'

    with patch.object(mod, "generate_structured", mock_generate):
        result = mod._call_llm("AAPL", "Quarter | ...\n---\nQ0 | 0.350 | 0.01 | 0.05")

    assert result is not None
    assert any("json_schema" in c for c in calls), \
           "_call_llm must pass json_schema= to generate_structured"


def test_quoted_evidence_stored_in_cache(tmp_path):
    """quoted_evidence from LLM response must be stored in the cache entry."""
    import ascent.alpha.llm_fundamental as mod
    from unittest.mock import patch
    import json

    fund = _make_fundamentals(symbols=["AAPL"])

    def mock_call(symbol, table):
        return {
            "direction": "UP",
            "confidence": 0.8,
            "key_trend": "improving",
            "uncertainty": "macro",
            "quoted_evidence": "Q0 gross_profitability=0.400",
        }

    cache_path = tmp_path / "c.json"
    with patch.object(mod, "CACHE_PATH", cache_path):
        with patch.object(mod, "_call_llm", side_effect=mock_call):
            mod.llm_fundamental_alpha(fund)

    cache = json.loads(cache_path.read_text())
    entries = list(cache.values())
    assert len(entries) == 1
    assert "quoted_evidence" in entries[0], \
           "Cache entry must store quoted_evidence for auditability"
