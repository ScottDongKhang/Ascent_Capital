# tests/monitoring/test_alpha_wedge_tracker.py
import json
import pytest
from pathlib import Path
from datetime import date, timedelta


def _write_wedge_log(tmp_path, entries):
    p = tmp_path / "logs" / "alpha_wedge.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def test_record_rebalance_writes_entry(tmp_path):
    from ascent.monitoring.alpha_wedge_tracker import record_rebalance
    log_path = tmp_path / "logs" / "alpha_wedge.jsonl"
    record_rebalance(
        "2026-05-19",
        ai_pm_weights={"AAPL": 0.10, "MSFT": 0.08},
        quant_weights={"AAPL": 0.12, "MSFT": 0.08, "SATS": 0.10},
        override_types={"SATS": "data_quality"},
        log_path=log_path,
    )
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip())
    assert entry["rebalance_date"] == "2026-05-19"
    assert "ai_pm_weights" in entry
    assert entry["wedge_21d"] is None  # not filled yet


def test_record_rebalance_deduplicates(tmp_path):
    from ascent.monitoring.alpha_wedge_tracker import record_rebalance
    log_path = tmp_path / "logs" / "alpha_wedge.jsonl"
    record_rebalance("2026-05-19", {"A": 0.5}, {"A": 0.5}, log_path=log_path)
    record_rebalance("2026-05-19", {"B": 0.5}, {"B": 0.5}, log_path=log_path)
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1


def test_update_outcomes_fills_wedge(tmp_path):
    from ascent.monitoring.alpha_wedge_tracker import update_outcomes
    old_date = (date.today() - timedelta(days=35)).isoformat()
    log_path = _write_wedge_log(tmp_path, [{
        "rebalance_date": old_date,
        "ai_pm_weights": {"AAPL": 1.0},
        "quant_weights": {"MSFT": 1.0},
        "override_types": {},
        "ai_pm_return_21d": None,
        "quant_return_21d": None,
        "wedge_21d": None,
    }])
    price_returns = {"AAPL": 0.05, "MSFT": 0.02}
    update_outcomes(price_returns, date.today().isoformat(), log_path=log_path)
    entry = json.loads(log_path.read_text().strip())
    assert abs(entry["ai_pm_return_21d"] - 0.05) < 1e-6
    assert abs(entry["quant_return_21d"] - 0.02) < 1e-6
    assert abs(entry["wedge_21d"] - 0.03) < 1e-6


def test_get_wedge_summary_no_data(tmp_path):
    from ascent.monitoring.alpha_wedge_tracker import get_wedge_summary
    log_path = tmp_path / "logs" / "alpha_wedge.jsonl"
    result = get_wedge_summary(log_path=log_path)
    assert "No alpha wedge data" in result


def test_get_wedge_summary_with_matured(tmp_path):
    from ascent.monitoring.alpha_wedge_tracker import get_wedge_summary
    log_path = _write_wedge_log(tmp_path, [
        {"rebalance_date": "2026-04-15", "ai_pm_weights": {}, "quant_weights": {},
         "override_types": {}, "ai_pm_return_21d": 0.05, "quant_return_21d": 0.03, "wedge_21d": 0.02},
        {"rebalance_date": "2026-05-05", "ai_pm_weights": {}, "quant_weights": {},
         "override_types": {}, "ai_pm_return_21d": 0.01, "quant_return_21d": 0.04, "wedge_21d": -0.03},
    ])
    result = get_wedge_summary(log_path=log_path)
    assert "2026-04-15" in result
    assert "2026-05-05" in result
    assert "Mean wedge" in result
