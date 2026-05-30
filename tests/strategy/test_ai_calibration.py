# tests/strategy/test_ai_calibration.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def _patch_log(tmp_path):
    from ascent.strategy import ai_calibration as _mod
    log_path = Path(tmp_path) / "ai_thesis_outcomes.jsonl"
    return patch.object(_mod, "OUTCOMES_LOG", log_path)


def test_log_thesis_writes_entry():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation",
                           {"trend": 0.004, "statarb": -0.002})
        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["thesis_date"] == "2026-06-09"
        assert entry["market_character"] == "momentum_continuation"
        assert entry["prediction_correct"] is None


def test_update_outcome_fills_most_recent_pending():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            cal.log_thesis("2026-06-23", "calm_bull", "sector_rotation")
            cal.update_outcome({"trend": 0.020, "statarb": 0.005, "meanrev": -0.003})

        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        # First entry still pending
        assert entries[0]["prediction_correct"] is None
        # Second entry (most recent) filled
        assert entries[1]["prediction_correct"] is not None
        assert entries[1]["realized_ic_leaders"] is not None


def test_momentum_continuation_correct_when_trend_leads():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            # trend has highest positive IC → correct for momentum_continuation
            cal.update_outcome({"trend": 0.020, "statarb": 0.005})

        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["prediction_correct"] is True


def test_momentum_continuation_wrong_when_trend_does_not_lead():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            # statarb leads, trend is negative → wrong
            cal.update_outcome({"statarb": 0.015, "meanrev": 0.012, "trend": -0.003})

        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["prediction_correct"] is False


def test_get_context_returns_empty_when_insufficient_data():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            # Only 2 entries — get_context needs >= 3
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            cal.log_thesis("2026-06-23", "calm_bull", "momentum_continuation")
        context = cal.get_context("calm_bull")
        assert context == ""


def test_get_context_returns_string_with_enough_data():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        # Write 5 completed entries directly
        entries = [
            {"thesis_date": f"2026-0{i+1}-01", "regime": "calm_bull",
             "market_character": "momentum_continuation",
             "sleeve_weight_prior": {},
             "realized_ic_leaders": ["trend"],
             "prediction_correct": True}
            for i in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))
        from ascent.strategy import ai_calibration as cal
        with patch.object(cal, "OUTCOMES_LOG", log_path):
            context = cal.get_context("calm_bull")
        assert "calm_bull" in context
        assert "momentum_continuation" in context
        assert "5/5" in context or "100%" in context


def test_get_context_is_empty_for_different_regime():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entries = [
            {"thesis_date": f"2026-0{i+1}-01", "regime": "calm_bull",
             "market_character": "momentum_continuation", "sleeve_weight_prior": {},
             "realized_ic_leaders": ["trend"], "prediction_correct": True}
            for i in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))
        from ascent.strategy import ai_calibration as cal
        with patch.object(cal, "OUTCOMES_LOG", log_path):
            context = cal.get_context("stressed")
        assert context == ""
