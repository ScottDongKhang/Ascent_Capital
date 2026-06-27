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
    thesis = _make_thesis(rationale={"AAPL": "real decision rationale"})

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
    thesis = _make_thesis(rationale={"AAPL": "real decision rationale"})

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


# ── A: Stub-pollution guard (non-decision entries must not be logged) ────────────
#
# Off-calendar / daily-view days were writing degenerate entries into the
# production calibration log (e.g. AAPL/MSFT/NVDA at default weights, empty
# thesis, all conviction "quant_agreed", null rationale). Those are NOT AI PM
# decisions — a real Phase-2 decision always carries AI reasoning in the thesis
# (position_rationale and/or quant_overrides). log_prediction must refuse an
# entry whose thesis contributes zero AI reasoning, so the calibration record
# stays a clean ledger of real decisions.

def test_log_prediction_skips_empty_thesis_stub(tmp_path):
    """A degenerate book with an empty thesis (no rationale, no overrides) is a
    non-decision and must NOT be written to the calibration log."""
    log_path = tmp_path / "ai_pm_calibration.jsonl"
    portfolio = {"AAPL": 0.10, "MSFT": 0.10, "NVDA": 0.08}  # the observed stub signature
    thesis = _make_thesis()  # empty: no overrides, no rationale, no agreement

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction("2026-06-23", portfolio, thesis)

    assert not log_path.exists() or log_path.read_text().strip() == "", (
        "Empty-thesis stub must not be logged"
    )


def test_log_prediction_logs_real_decision_with_rationale(tmp_path):
    """A genuine AI PM decision (thesis carries rationale) must still be logged."""
    log_path = tmp_path / "ai_pm_calibration.jsonl"
    portfolio = {"STLD": 0.09, "HAE": 0.09, "MGM": 0.07}
    thesis = _make_thesis(rationale={"STLD": "AMPLIFY — steel infra thesis, 93% earnings growth"})

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction("2026-06-10", portfolio, thesis)

    assert log_path.exists()
    entries = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(entries) == 1
    assert "STLD" in entries[0]["positions"]


def test_log_prediction_logs_override_only_decision(tmp_path):
    """A decision whose AI reasoning lives in quant_overrides (not position_rationale)
    is still a real decision and must be logged."""
    log_path = tmp_path / "ai_pm_calibration.jsonl"
    portfolio = {"EWY": 0.08}
    thesis = _make_thesis(overrides=[{"symbol": "EWY", "ai_action": "increase", "reason": "AI/semi cycle"}])

    with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction("2026-06-11", portfolio, thesis)

    assert log_path.exists()
    entries = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
    assert len(entries) == 1


# ── Task 5: Per-rebalance cross-sectional IC ───────────────────────────────────


def test_get_per_rebalance_ic_returns_empty_when_all_same_conviction():
    """If every position has the same conviction level, IC is undefined — returns []."""
    from ascent.strategy.calibration_tracker import get_per_rebalance_ic, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": 0.02},
            "MSFT": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": 0.01},
            "GOOG": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": -0.01},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            result = get_per_rebalance_ic()
    assert result == []


def test_get_per_rebalance_ic_positive_when_high_conviction_wins():
    """High conviction position outperforms → positive IC for that rebalance."""
    from ascent.strategy.calibration_tracker import get_per_rebalance_ic, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.2, "conviction": "high",          "realized_21d": 0.05},
            "MSFT": {"weight": 0.1, "conviction": "medium",        "realized_21d": 0.02},
            "GOOG": {"weight": 0.1, "conviction": "quant_agreed",  "realized_21d": 0.01},
            "AMZN": {"weight": 0.1, "conviction": "quant_agreed",  "realized_21d": -0.01},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            result = get_per_rebalance_ic()
    assert len(result) == 1
    assert result[0]["date"] == "2026-06-10"
    assert result[0]["ic"] > 0
    assert result[0]["n_conviction_levels"] >= 2


def test_get_per_rebalance_ic_skips_entry_without_realized_outcomes():
    """Entry where no position has realized_21d is excluded entirely."""
    from ascent.strategy.calibration_tracker import get_per_rebalance_ic, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.2, "conviction": "high",         "realized_21d": None},
            "MSFT": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": None},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            result = get_per_rebalance_ic()
    assert result == []


def test_calibration_report_includes_per_rebalance_section():
    """get_calibration_report() contains per-rebalance IC section when data exists."""
    from ascent.strategy.calibration_tracker import get_calibration_report, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.2, "conviction": "high",          "realized_21d": 0.05},
            "MSFT": {"weight": 0.1, "conviction": "medium",        "realized_21d": 0.02},
            "GOOG": {"weight": 0.1, "conviction": "quant_agreed",  "realized_21d": -0.01},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            report = get_calibration_report()
    assert "per-rebalance" in report.lower() or "Per-rebalance" in report
