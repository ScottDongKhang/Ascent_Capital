"""
tests/test_regime_memory.py
Tests for memory/regime_memory.py — regime-aware episodic memory.
"""
import json
from datetime import date, timedelta
from pathlib import Path
import pytest

import memory.regime_memory as rm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _patch_log(tmp_path, monkeypatch):
    """Redirect EPISODE_LOG to a temp path and return it."""
    ep_log = tmp_path / "regime_episodes.jsonl"
    monkeypatch.setattr(rm, "EPISODE_LOG", ep_log)
    return ep_log


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_log_episode_creates_file(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    rm.log_episode(
        run_date="2026-05-17",
        regime="calm_bull",
        quant_weights={"AAPL": 0.05, "MSFT": 0.08},
        ai_weights=None,
    )

    assert ep_log.exists(), "EPISODE_LOG should be created"
    rows = [json.loads(l) for l in ep_log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["date"] == "2026-05-17"
    assert row["regime"] == "calm_bull"
    assert row["quant_weights"] == {"AAPL": 0.05, "MSFT": 0.08}
    assert row["ai_weights"] is None
    assert row["realized_return_21d"] is None


def test_log_episode_deduplicates(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    rm.log_episode("2026-05-17", "calm_bull", {"AAPL": 0.05})
    rm.log_episode("2026-05-17", "stressed", {"TLT": 0.10})  # same date, different regime

    rows = [json.loads(l) for l in ep_log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1, "Second log_episode for same date should be a no-op"
    assert rows[0]["regime"] == "calm_bull"  # first write wins


def test_query_episodes_empty(tmp_path, monkeypatch):
    _patch_log(tmp_path, monkeypatch)
    # No file exists
    result = rm.query_episodes("calm_bull", n=5)
    assert "No episodes found" in result
    assert "calm_bull" in result


def test_query_episodes_regime_match(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    # 3 calm_bull, 1 stressed
    entries = [
        {"date": "2026-04-01", "regime": "calm_bull", "quant_weights": {"AAPL": 0.05}, "ai_weights": None, "realized_return_21d": 0.012},
        {"date": "2026-04-05", "regime": "calm_bull", "quant_weights": {"MSFT": 0.06}, "ai_weights": None, "realized_return_21d": None},
        {"date": "2026-04-10", "regime": "stressed",  "quant_weights": {"TLT": 0.10},  "ai_weights": None, "realized_return_21d": -0.03},
        {"date": "2026-04-15", "regime": "calm_bull", "quant_weights": {"GOOG": 0.07}, "ai_weights": None, "realized_return_21d": 0.02},
    ]
    with open(ep_log, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")

    calm_result = rm.query_episodes("calm_bull", n=5)
    assert "stressed" not in calm_result
    # Should contain 3 calm_bull episodes
    assert calm_result.count("Episode") == 3

    stressed_result = rm.query_episodes("stressed", n=5)
    assert "stressed" in stressed_result
    assert stressed_result.count("Episode") == 1


def test_query_episodes_prefix_match(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    entry = {
        "date": "2026-04-20",
        "regime": "calm_bull",
        "quant_weights": {"AAPL": 0.05},
        "ai_weights": None,
        "realized_return_21d": 0.015,
    }
    with open(ep_log, "w") as fh:
        fh.write(json.dumps(entry) + "\n")

    result = rm.query_episodes("calm", n=5)
    assert "No episodes found" not in result
    assert "calm_bull" in result
    assert "Episode" in result


def test_update_outcomes_fills_return(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    # Episode 30 days ago — old enough
    old_date = (date.today() - timedelta(days=30)).isoformat()
    entry = {
        "date": old_date,
        "regime": "calm_bull",
        "quant_weights": {"AAPL": 0.50, "MSFT": 0.50},
        "ai_weights": None,
        "realized_return_21d": None,
    }
    with open(ep_log, "w") as fh:
        fh.write(json.dumps(entry) + "\n")

    price_returns = {"AAPL": 0.08, "MSFT": 0.04}
    n = rm.update_outcomes(price_returns)

    assert n == 1, "Should have updated exactly 1 episode"

    rows = [json.loads(l) for l in ep_log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    realized = rows[0]["realized_return_21d"]
    assert realized is not None
    expected = 0.50 * 0.08 + 0.50 * 0.04  # = 0.06
    assert abs(realized - expected) < 1e-9, f"Expected {expected}, got {realized}"


def test_update_outcomes_skips_recent(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    # Episode only 5 days ago — too recent
    recent_date = (date.today() - timedelta(days=5)).isoformat()
    entry = {
        "date": recent_date,
        "regime": "calm_bull",
        "quant_weights": {"AAPL": 1.0},
        "ai_weights": None,
        "realized_return_21d": None,
    }
    with open(ep_log, "w") as fh:
        fh.write(json.dumps(entry) + "\n")

    n = rm.update_outcomes({"AAPL": 0.10})
    assert n == 0, "Recent episode should not be updated"
    rows = [json.loads(l) for l in ep_log.read_text().splitlines() if l.strip()]
    assert rows[0]["realized_return_21d"] is None


def test_update_outcomes_already_filled(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    old_date = (date.today() - timedelta(days=30)).isoformat()
    entry = {
        "date": old_date,
        "regime": "calm_bull",
        "quant_weights": {"AAPL": 1.0},
        "ai_weights": None,
        "realized_return_21d": 0.05,  # already filled
    }
    with open(ep_log, "w") as fh:
        fh.write(json.dumps(entry) + "\n")

    n = rm.update_outcomes({"AAPL": 0.99})
    assert n == 0, "Already-filled episode should not be updated again"
    rows = [json.loads(l) for l in ep_log.read_text().splitlines() if l.strip()]
    assert rows[0]["realized_return_21d"] == 0.05  # unchanged


def test_query_episodes_n_limit(tmp_path, monkeypatch):
    ep_log = _patch_log(tmp_path, monkeypatch)

    # Log 8 calm_bull episodes
    for i in range(8):
        entry = {
            "date": f"2026-0{i+1}-01" if i < 9 else f"2026-10-01",
            "regime": "calm_bull",
            "quant_weights": {"AAPL": 0.05},
            "ai_weights": None,
            "realized_return_21d": None,
        }
        with open(ep_log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    result = rm.query_episodes("calm_bull", n=3)
    assert result.count("Episode") == 3, "Should return at most n episodes"
