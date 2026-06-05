"""
tests/test_ai_pm_counterfactual.py
Tests for ascent/monitoring/ai_pm_counterfactual.py
"""
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch


def test_snapshot_quant_star_is_idempotent():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_quant_star, QUANT_STAR_LOG
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_star.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_STAR_LOG", log_path):
            snapshot_quant_star(date(2026, 6, 4), {"AAPL": 0.5, "MSFT": 0.5})
            snapshot_quant_star(date(2026, 6, 4), {"AAPL": 0.6, "MSFT": 0.4})  # re-run same date
            lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1  # idempotent — second write skipped
    assert json.loads(lines[0])["weights"]["AAPL"] == 0.5  # first write preserved


def test_snapshot_ai_pm_normalizes_weights():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_ai.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 4), {"AAPL": 0.6, "MSFT": 0.6})  # sums to 1.2
            entry = json.loads(log_path.read_text().strip())
    total = sum(entry["weights"].values())
    assert abs(total - 1.0) < 0.001


def test_score_daily_appends_all_tracks():
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 5),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.012,
                spy_return=0.008,
                prices={
                    "AAPL": {"prev": 100.0, "curr": 101.5},
                    "MSFT": {"prev": 200.0, "curr": 201.0},
                },
            )
        entry = json.loads(log_path.read_text().strip())
    assert "track_astar_return" in entry
    assert "track_a_return" in entry
    assert "track_b_return" in entry
    assert "track_c_return" in entry
    assert "track_d_return" in entry
    assert abs(entry["track_b_return"] - 0.012) < 0.0001
    assert abs(entry["track_c_return"] - 0.008) < 0.0001


def test_score_daily_computes_track_d_correctly():
    """Track D = pure AI PM weighted returns. AAPL 60%, +1.5%; MSFT 40%, +0.5% → 1.1%"""
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 5),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.012,
                spy_return=0.008,
                prices={
                    "AAPL": {"prev": 100.0, "curr": 101.5},  # +1.5%
                    "MSFT": {"prev": 200.0, "curr": 201.0},  # +0.5%
                },
            )
        entry = json.loads(log_path.read_text().strip())
    expected_d = 0.6 * 0.015 + 0.4 * 0.005  # 0.011
    assert abs(entry["track_d_return"] - expected_d) < 0.0001


def test_load_snapshots_returns_empty_on_missing_files():
    from ascent.monitoring.ai_pm_counterfactual import load_snapshots
    with tempfile.TemporaryDirectory() as tmp:
        star = Path(tmp) / "star.jsonl"
        quant = Path(tmp) / "quant.jsonl"
        ai = Path(tmp) / "ai.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_STAR_LOG", star):
            with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_LOG", quant):
                with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", ai):
                    star_w, quant_w, ai_w = load_snapshots()
    assert star_w == {}
    assert quant_w == {}
    assert ai_w == {}
