"""
tests/test_plan_c.py
Tests for Plan C: verdict scoring, live Sharpe exposure, deterministic self-improve evaluator.
"""

import json
from datetime import date, timedelta
from pathlib import Path
import pytest


def test_nav_series_reads_holdings_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    entries = [
        {"date": "2026-04-01", "equity": 100000.0},
        {"date": "2026-04-02", "equity": 101200.0},
        {"date": "2026-04-15", "equity": 103500.0},
    ]
    with open(tmp_path / "logs" / "holdings_log.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    from debate.outcome_tracker import _load_nav_series
    nav = _load_nav_series()
    assert "2026-04-01" in nav
    assert abs(nav["2026-04-01"] - 100000.0) < 0.01
    assert abs(nav["2026-04-15"] - 103500.0) < 0.01


def test_nav_series_falls_back_to_eod_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    # No holdings_log — only eod_log
    with open(tmp_path / "logs" / "eod_log.jsonl", "w") as f:
        f.write(json.dumps({"date": "2026-04-01", "nav": 98000.0}) + "\n")
    from debate.outcome_tracker import _load_nav_series
    nav = _load_nav_series()
    assert "2026-04-01" in nav
    assert abs(nav["2026-04-01"] - 98000.0) < 0.01


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
    """Same variant must produce identical scores on repeated calls."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "dashboard").mkdir()
    # No PnL data — falls back to baseline + diversity_adj
    from ascent.research.self_improve import evaluate_variant
    variant = {"alpha_weights": {"trend": 0.70, "statarb": 0.15, "meanrev": 0.05, "ml": 0.10, "volatility": 0.0}}
    s1 = evaluate_variant(variant)
    s2 = evaluate_variant(variant)
    assert abs(s1 - s2) < 0.001, f"evaluate_variant not deterministic: {s1} vs {s2}"
    assert isinstance(s1, float)


def test_get_baseline_sharpe_returns_none_gracefully(tmp_path, monkeypatch):
    """get_baseline_sharpe returns None when no data exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    from ascent.research.self_improve import get_baseline_sharpe
    result = get_baseline_sharpe()
    assert result is None
