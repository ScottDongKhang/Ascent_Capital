# tests/test_conviction_gate.py
import json
import pytest
from pathlib import Path


def _write_memory_log(tmp_path, records):
    p = tmp_path / "logs" / "decision_memory.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _make_record(i, override_type, regime, wedge):
    return {
        "entry_id": f"2026-04-{i:02d}_{override_type[:3]}",
        "rebalance_date": f"2026-04-{i:02d}",
        "symbol": "X",
        "override_type": override_type,
        "regime": regime,
        "ai_action": "REDUCED",
        "ai_weight": 0.05,
        "quant_weight": 0.10,
        "weight_delta": -0.05,
        "momentum_252d": None,
        "wedge_21d": wedge,
    }


def test_data_quality_always_proceeds(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("data_quality", "calm_bull", log_path=log_path)
    assert result.proceed is True
    assert result.size_multiplier == 1.0
    assert result.confidence == "proceed"


def test_news_event_always_proceeds(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("news_event", "stressed", log_path=log_path)
    assert result.proceed is True
    assert result.size_multiplier == 1.0


def test_correlation_risk_always_proceeds(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("correlation_risk", "crisis", log_path=log_path)
    assert result.proceed is True
    assert result.size_multiplier == 1.0


def test_valuation_blocked_by_low_calibration_ic(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("valuation", "calm_bull", calibration_ic=0.05, log_path=log_path)
    assert result.proceed is False
    assert result.size_multiplier == 0.0
    assert result.confidence == "block"
    assert "IC=" in result.reason


def test_valuation_proceeds_with_good_ic(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("valuation", "calm_bull", calibration_ic=0.25, log_path=log_path)
    assert result.proceed is True


def test_valuation_blocked_by_poor_win_rate(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    records = [_make_record(i, "valuation", "calm_bull", -0.02) for i in range(1, 11)]
    log_path = _write_memory_log(tmp_path, records)
    result = evaluate("valuation", "calm_bull", log_path=log_path)
    assert result.proceed is False
    assert result.confidence == "block"


def test_insufficient_data_caution(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("regime_macro", "calm_bull", log_path=log_path)
    assert result.proceed is True
    assert result.size_multiplier == 0.85
    assert result.confidence == "caution"


def test_strong_track_record_full_size(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    records = [_make_record(i, "regime_macro", "calm_bull", 0.02) for i in range(1, 11)]
    log_path = _write_memory_log(tmp_path, records)
    result = evaluate("regime_macro", "calm_bull", log_path=log_path)
    assert result.proceed is True
    assert result.size_multiplier == 1.0
    assert result.confidence == "strong"


def test_mixed_track_record_reduced_size(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    # 5 wins, 5 losses → 50% win rate → "proceed" at 0.75
    records = (
        [_make_record(i, "regime_macro", "calm_bull", 0.02) for i in range(1, 6)] +
        [_make_record(i, "regime_macro", "calm_bull", -0.01) for i in range(6, 11)]
    )
    log_path = _write_memory_log(tmp_path, records)
    result = evaluate("regime_macro", "calm_bull", log_path=log_path)
    assert result.proceed is True
    assert abs(result.size_multiplier - 0.75) < 0.01
    assert result.confidence == "proceed"


def test_poor_track_record_blocked(tmp_path):
    from ascent.strategy.conviction_gate import evaluate
    # 9 losses, 1 win → 10% → block if n >= MIN_BLOCK_CASES (8)
    records = (
        [_make_record(i, "regime_macro", "calm_bull", -0.02) for i in range(1, 10)] +
        [_make_record(10, "regime_macro", "calm_bull", 0.01)]
    )
    log_path = _write_memory_log(tmp_path, records)
    result = evaluate("regime_macro", "calm_bull", log_path=log_path)
    assert result.proceed is False
    assert result.size_multiplier == 0.0
    assert result.confidence == "block"


def test_format_gate_result_proceed(tmp_path):
    from ascent.strategy.conviction_gate import evaluate, format_gate_result
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("data_quality", "calm_bull", log_path=log_path)
    text = format_gate_result(result)
    assert "PROCEED" in text
    assert "Size multiplier" in text


def test_format_gate_result_blocked(tmp_path):
    from ascent.strategy.conviction_gate import evaluate, format_gate_result
    records = [_make_record(i, "regime_macro", "calm_bull", -0.02) for i in range(1, 11)]
    log_path = _write_memory_log(tmp_path, records)
    result = evaluate("regime_macro", "calm_bull", log_path=log_path)
    text = format_gate_result(result)
    assert "BLOCKED" in text
    assert "Size multiplier" not in text
