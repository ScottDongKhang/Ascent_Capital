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


def _make_record(i, override_type, regime, wedge, sec_tone=None, trends_direction=None):
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
        "sec_tone": sec_tone,
        "transcript_sentiment": None,
        "reddit_buzz": None,
        "trends_direction": trends_direction,
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


def test_all_records_unmatured_no_crash(tmp_path):
    """n >= 5 records but all wedge_21d=None (unmatured) — wr is None, must not raise TypeError."""
    from ascent.strategy.conviction_gate import evaluate
    records = [
        {**_make_record(i, "regime_macro", "calm_bull", None), "wedge_21d": None}
        for i in range(1, 8)
    ]
    log_path = _write_memory_log(tmp_path, records)
    result = evaluate("regime_macro", "calm_bull", log_path=log_path)
    assert result.confidence in ("strong", "proceed", "caution", "block")


def test_evaluate_accepts_symbol_param(tmp_path):
    """symbol kwarg accepted without error — backward compat with no symbol."""
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("data_quality", "calm_bull", log_path=log_path, symbol="AAPL")
    assert result.proceed is True


def test_ml_model_not_triggered_below_30_cases(tmp_path):
    """25 matured valuation records → _get_ml_model returns None → rules path."""
    from unittest.mock import patch
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.01 if i % 2 == 0 else -0.01)
               for i in range(1, 26)]
    log_path = _write_memory_log(tmp_path, records)

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=None) as mock_ml:
        result = evaluate("valuation", "calm_bull", log_path=log_path)
        mock_ml.assert_called_once()
    assert result.confidence in ("strong", "proceed", "caution", "block")


def test_ml_model_activates_at_30_cases_high_confidence(tmp_path):
    """30 matured valuation records + mock ML returning 0.80 prob → STRONG gate."""
    from unittest.mock import patch, MagicMock
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.02) for i in range(1, 31)]
    log_path = _write_memory_log(tmp_path, records)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.20, 0.80]]

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=mock_model):
        result = evaluate("valuation", "calm_bull", log_path=log_path)

    assert result.proceed is True
    assert result.confidence == "strong"
    assert result.size_multiplier == 1.0


def test_ml_model_falls_back_when_not_confident(tmp_path):
    """Model returns prob=0.53 (|0.53−0.5|=0.03 < 0.05) → falls back to rules."""
    from unittest.mock import patch, MagicMock
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.01 if i <= 20 else -0.01)
               for i in range(1, 31)]
    log_path = _write_memory_log(tmp_path, records)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.47, 0.53]]

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=mock_model):
        result = evaluate("valuation", "calm_bull", log_path=log_path)

    # Model confidence too low — fell back to rules-based path.
    # Rules path can return any valid confidence level; verify result is rules-driven
    # (reason will mention "track record" or "calibration", not "ML model").
    assert "ML model" not in result.reason
    assert result.confidence in ("strong", "proceed", "caution", "block")


def test_ml_model_blocks_on_low_probability(tmp_path):
    """Model returns prob=0.25 → BLOCKED."""
    from unittest.mock import patch, MagicMock
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.02) for i in range(1, 31)]
    log_path = _write_memory_log(tmp_path, records)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.75, 0.25]]

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=mock_model):
        result = evaluate("valuation", "calm_bull", log_path=log_path)

    assert result.proceed is False
    assert result.confidence == "block"
