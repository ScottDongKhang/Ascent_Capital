# tests/test_calibration_tracker.py
"""
Tests for ascent/strategy/calibration_tracker.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_thesis(
    overrides=None,
    rationale=None,
    agreement=None,
) -> dict:
    """Build a minimal thesis dict matching the propose_portfolio schema."""
    return {
        "market_view": "test",
        "quant_overrides": overrides or [],
        "position_rationale": rationale or {},
        "quant_agreement": agreement or [],
    }


def _write_entry(log_path: Path, run_date: str, positions: dict) -> None:
    """Write a raw calibration entry directly to the log."""
    entry = {"date": run_date, "positions": positions}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_log_prediction_creates_file(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"
    portfolio = {"AAPL": 0.05, "MSFT": 0.04}
    thesis = _make_thesis()

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction("2026-05-17", portfolio, thesis)

    assert log_path.exists()
    entries = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["date"] == "2026-05-17"
    assert "AAPL" in entry["positions"]
    assert "MSFT" in entry["positions"]
    # Check position structure
    pos = entry["positions"]["AAPL"]
    assert "weight" in pos
    assert "conviction" in pos
    assert "realized_21d" in pos
    assert pos["realized_21d"] is None


def test_log_prediction_deduplicates(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"
    portfolio = {"AAPL": 0.05}
    thesis = _make_thesis()

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction("2026-05-17", portfolio, thesis)
        log_prediction("2026-05-17", portfolio, thesis)

    entries = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(entries) == 1, "Duplicate date should produce only one entry"


def test_log_prediction_conviction_levels(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"

    # AAPL: in quant_overrides → high
    # MSFT: in position_rationale only → medium
    # GOOG: in quant_agreement only → quant_agreed
    # AMZN: no mention → quant_agreed
    portfolio = {
        "AAPL": 0.10,
        "MSFT": 0.08,
        "GOOG": 0.06,
        "AMZN": 0.05,
    }
    thesis = _make_thesis(
        overrides=[{"symbol": "AAPL", "ai_action": "increase", "reason": "strong earnings"}],
        rationale={"MSFT": "Cloud growth accelerating"},
        agreement=["GOOG"],
    )

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction("2026-05-17", portfolio, thesis)

    entries = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    positions = entries[0]["positions"]

    assert positions["AAPL"]["conviction"] == "high"
    assert positions["MSFT"]["conviction"] == "medium"
    assert positions["GOOG"]["conviction"] == "quant_agreed"
    assert positions["AMZN"]["conviction"] == "quant_agreed"


def test_get_calibration_report_no_data(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"  # does not exist yet

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import get_calibration_report
        report = get_calibration_report(n_rebalances=10)

    assert report == "No calibration data yet."


def test_get_calibration_report_with_outcomes(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"

    # Inject 3 entries with realized_21d filled in
    entries = [
        {
            "date": "2026-04-01",
            "positions": {
                "AAPL": {"weight": 0.10, "conviction": "high",        "rationale": "strong", "realized_21d": 0.05},
                "MSFT": {"weight": 0.08, "conviction": "medium",      "rationale": "cloud",  "realized_21d": 0.03},
                "GOOG": {"weight": 0.06, "conviction": "quant_agreed","rationale": None,     "realized_21d": 0.01},
            },
        },
        {
            "date": "2026-04-15",
            "positions": {
                "NVDA": {"weight": 0.12, "conviction": "high",        "rationale": "AI",     "realized_21d": 0.08},
                "AMD":  {"weight": 0.07, "conviction": "quant_agreed","rationale": None,     "realized_21d": -0.01},
            },
        },
        {
            "date": "2026-05-01",
            "positions": {
                "V":   {"weight": 0.09, "conviction": "medium",      "rationale": "payments","realized_21d": 0.02},
                "JPM": {"weight": 0.07, "conviction": "quant_agreed","rationale": None,      "realized_21d": 0.00},
            },
        },
    ]
    with open(log_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import get_calibration_report
        report = get_calibration_report(n_rebalances=10)

    # Should contain IC line and by-conviction breakdown
    assert "IC (conviction → 21d return):" in report
    assert "high" in report
    assert "medium" in report
    assert "quant_agreed" in report
    # high picks (AAPL 5%, NVDA 8%) should show positive avg
    assert "Calibration Report" in report


def test_update_outcomes_fills_returns(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"

    # Entry 25 days old — old enough for 21-day lookback
    old_date = (date.today() - timedelta(days=25)).isoformat()
    entry = {
        "date": old_date,
        "positions": {
            "AAPL": {"weight": 0.10, "conviction": "high", "rationale": "test", "realized_21d": None},
            "MSFT": {"weight": 0.08, "conviction": "quant_agreed", "rationale": None, "realized_21d": None},
        },
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    price_returns = {"AAPL": 0.04, "MSFT": 0.02}

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import update_outcomes
        n = update_outcomes(price_returns, date.today().isoformat(), lookback_days=21)

    assert n == 1

    # Re-read and verify
    updated = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    aapl_ret = updated[0]["positions"]["AAPL"]["realized_21d"]
    msft_ret = updated[0]["positions"]["MSFT"]["realized_21d"]
    assert aapl_ret == pytest.approx(0.04)
    assert msft_ret == pytest.approx(0.02)


def test_update_outcomes_skips_recent(tmp_path):
    log_path = tmp_path / "ai_pm_calibration.jsonl"

    # Entry only 10 days old — too recent for 21-day lookback
    recent_date = (date.today() - timedelta(days=10)).isoformat()
    entry = {
        "date": recent_date,
        "positions": {
            "AAPL": {"weight": 0.10, "conviction": "high", "rationale": "test", "realized_21d": None},
        },
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    price_returns = {"AAPL": 0.05}

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import update_outcomes
        n = update_outcomes(price_returns, date.today().isoformat(), lookback_days=21)

    assert n == 0

    # realized_21d should still be None
    unchanged = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert unchanged[0]["positions"]["AAPL"]["realized_21d"] is None


def test_compute_calibration_returns_feeds_update_outcomes(tmp_path):
    """
    _compute_calibration_returns must return real symbol returns (not {}) for
    entries >= 21 days old so that update_outcomes can fill realized_21d and
    get_calibration_report produces a real IC report.
    """
    import importlib
    import run_all_agents
    importlib.reload(run_all_agents)

    # Verify the function exists (fails RED before implementation)
    assert hasattr(run_all_agents, "_compute_calibration_returns"), (
        "_compute_calibration_returns missing from run_all_agents"
    )

    tz = "America/New_York"
    entry_date = date.today() - timedelta(days=22)
    buy_ts = pd.Timestamp(entry_date.isoformat()).tz_localize(tz)
    today_ts = pd.Timestamp(date.today().isoformat()).tz_localize(tz)

    log_path = tmp_path / "ai_pm_calibration.jsonl"
    entry = {
        "date": entry_date.isoformat(),
        "positions": {
            "FAKE1": {"weight": 0.10, "conviction": "high", "rationale": "test", "realized_21d": None},
            "FAKE2": {"weight": 0.08, "conviction": "medium", "rationale": "test", "realized_21d": None},
        },
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    mock_prices = pd.DataFrame({
        "symbol": ["FAKE1", "FAKE1", "FAKE2", "FAKE2"],
        "date":   [buy_ts, today_ts, buy_ts, today_ts],
        "close":  [100.0, 105.0, 200.0, 206.0],
    })

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path), \
         patch("ascent.data.store.parquet.load_parquet", return_value=mock_prices):
        importlib.reload(run_all_agents)
        result = run_all_agents._compute_calibration_returns(date.today())

    assert "FAKE1" in result, "FAKE1 missing from returns dict"
    assert "FAKE2" in result, "FAKE2 missing from returns dict"
    assert result["FAKE1"] == pytest.approx(0.05, abs=1e-6)
    assert result["FAKE2"] == pytest.approx(0.03, abs=1e-6)

    # Verify the result integrates with update_outcomes → get_calibration_report
    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import update_outcomes, get_calibration_report
        n = update_outcomes(result, date.today().isoformat())
        assert n == 1, f"Expected 1 entry updated, got {n}"
        report = get_calibration_report()
        assert report != "No calibration data yet.", "get_calibration_report still shows no data"
