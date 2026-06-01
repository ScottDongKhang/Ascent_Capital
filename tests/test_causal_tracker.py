# tests/test_causal_tracker.py
import json
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch


def _write_prediction(log_path: Path, symbol: str, timing: str,
                      rebalance_date: str, horizon_days: int,
                      velocity: float = 0.5) -> None:
    record = {
        "symbol": symbol,
        "mechanism": f"Test mechanism for {symbol}",
        "intervention": "IF x THEN y",
        "falsification_condition": "IF price < -8% thesis broken",
        "horizon_days": horizon_days,
        "rebalance_date": rebalance_date,
        "timing": timing,
        "velocity": velocity,
        "regime_compatible": True,
        "outcome": "pending",
        "early_exit": False,
        "checked_date": None,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def test_write_predictions_creates_jsonl(tmp_path):
    """write_predictions must create the log file with correct records."""
    from ascent.config.types import CausalMechanism
    from ascent.causal.tracker import write_predictions

    log_path = tmp_path / "causal_predictions.jsonl"
    mechanisms = [
        CausalMechanism(
            symbol="WDC", mechanism="NAND recovery", intervention="IF nand +15%",
            falsification_condition="IF margin < 38%", horizon_days=63,
            timing="catalyst_imminent", velocity=0.72,
            mechanism_type="supply_demand_inflection", regime_compatible=True,
        )
    ]
    write_predictions(mechanisms, rebalance_date="2026-06-15", log_path=log_path)

    assert log_path.exists()
    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert len(records) == 1
    r = records[0]
    assert r["symbol"] == "WDC"
    assert r["timing"] == "catalyst_imminent"
    assert r["outcome"] == "pending"
    assert r["early_exit"] is False
    assert r["rebalance_date"] == "2026-06-15"


def test_check_early_exits_flags_catalyst_imminent_on_large_drawdown(tmp_path):
    """catalyst_imminent with price -10% must be flagged early_exit=True."""
    from ascent.causal.tracker import check_early_exits

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=5)).isoformat()
    _write_prediction(log_path, "WDC", "catalyst_imminent", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=-0.10):
        early_exits = check_early_exits(log_path=log_path)

    assert "WDC" in early_exits, "WDC should be flagged for early exit"


def test_check_early_exits_no_flag_for_small_drawdown(tmp_path):
    """catalyst_imminent with price -3% must NOT be flagged."""
    from ascent.causal.tracker import check_early_exits

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=5)).isoformat()
    _write_prediction(log_path, "AAPL", "catalyst_imminent", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=-0.03):
        early_exits = check_early_exits(log_path=log_path)

    assert "AAPL" not in early_exits, "AAPL should NOT be flagged at -3% drawdown"


def test_check_outcomes_marks_confirmed_after_horizon(tmp_path):
    """Prediction past horizon with +7% return should be marked 'confirmed'."""
    from ascent.causal.tracker import check_outcomes

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=70)).isoformat()
    _write_prediction(log_path, "WDC", "catalyst_imminent", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=0.07):
        check_outcomes(log_path=log_path)

    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert records[0]["outcome"] == "confirmed"
    assert records[0]["checked_date"] is not None


def test_check_outcomes_marks_falsified_after_horizon(tmp_path):
    """Prediction past horizon with -7% return should be marked 'falsified'."""
    from ascent.causal.tracker import check_outcomes

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=70)).isoformat()
    _write_prediction(log_path, "AMD", "not_yet_priced", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=-0.07):
        check_outcomes(log_path=log_path)

    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert records[0]["outcome"] == "falsified"


def test_get_track_record_counts_outcomes(tmp_path):
    """get_track_record must return counts of total/confirmed/falsified and accuracy_pct."""
    from ascent.causal.tracker import get_track_record

    log_path = tmp_path / "causal_predictions.jsonl"
    _write_prediction(log_path, "A", "catalyst_imminent", "2026-01-01", 63)
    _write_prediction(log_path, "B", "not_yet_priced",   "2026-01-01", 63)

    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    records[0]["outcome"] = "confirmed"
    records[1]["outcome"] = "falsified"
    log_path.write_text("\n".join(json.dumps(r) for r in records))

    tr = get_track_record(log_path=log_path)
    assert tr["total"] == 2
    assert tr["confirmed"] == 1
    assert tr["falsified"] == 1
    assert tr["accuracy_pct"] == 50.0
