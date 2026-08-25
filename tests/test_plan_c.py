"""
tests/test_plan_c.py
Tests for Plan C: verdict scoring, live Sharpe exposure, deterministic self-improve evaluator.
"""

import json
from datetime import date, timedelta
from pathlib import Path
import pytest


def test_get_current_sharpe_reads_skill_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    scores = {
        "us_equities": {"sharpe": 0.642, "status": "active"},
        "macro": {"sharpe": 0.381, "status": "active"},
        "alternatives": {"sharpe": None, "status": "warming_up"},
    }
    (tmp_path / "dashboard" / "agent_skill_scores.json").write_text(json.dumps(scores))
    from ascent.monitoring.skill_tracker import get_current_sharpe
    assert abs(get_current_sharpe("us_equities") - 0.642) < 0.001
    assert abs(get_current_sharpe("macro") - 0.381) < 0.001
    assert get_current_sharpe("alternatives") is None
    assert get_current_sharpe("nonexistent") is None


def test_get_current_sharpe_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    from ascent.monitoring.skill_tracker import get_current_sharpe
    assert get_current_sharpe("us_equities") is None


def test_evaluate_variant_is_deterministic(tmp_path, monkeypatch):
    """Same variant must produce identical scores on repeated calls.

    No price cache exists in tmp_path, so run_lightweight_oos produces zero
    folds and _evaluate_variant_full's score/calmar come back None (no real
    Calmar computable). Bug fix (2026-08-23 review, round 2): evaluate_variant
    used to silently fall back to the Sharpe-scale baseline in this case --
    a unit-mismatch bug. It now fails closed to float('-inf') instead (same
    convention run_self_improve already used for a None score), so this test
    exercises exactly that branch. abs(s1 - s2) doesn't work for -inf (that's
    nan, which always fails a `< 0.001` comparison) -- assert equality
    directly instead.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "dashboard").mkdir()
    # run_lightweight_oos loads prices via the registered parquet store
    # (package-root data_cache), which chdir-ing into an empty tmp_path does
    # NOT isolate from -- it still finds this repo's real price cache. Force
    # the zero-folds path directly instead of relying on cwd-isolation that
    # doesn't actually apply here, so this test genuinely exercises the
    # fail-closed branch: score/calmar come back None, so evaluate_variant
    # fails closed to float('-inf').
    import ascent.research.walk_forward_lightweight as wfl
    monkeypatch.setattr(wfl, "run_lightweight_oos", lambda *a, **kw: {"n_folds": 0})
    from ascent.research.self_improve import evaluate_variant
    variant = {"alpha_weights": {"trend": 0.70, "statarb": 0.15, "meanrev": 0.05, "ml": 0.10, "volatility": 0.0}}
    s1 = evaluate_variant(variant)
    s2 = evaluate_variant(variant)
    assert s1 == s2 == float("-inf"), f"evaluate_variant not deterministic: {s1} vs {s2}"
    assert isinstance(s1, float)


def test_get_baseline_sharpe_returns_none_gracefully(tmp_path, monkeypatch):
    """get_baseline_sharpe returns None when no data exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    from ascent.research.self_improve import get_baseline_sharpe
    result = get_baseline_sharpe()
    assert result is None
