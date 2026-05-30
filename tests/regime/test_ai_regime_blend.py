# tests/regime/test_ai_regime_blend.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def _make_engine_with_cache(labels, risk_mults=None):
    """Return a RegimeEngine whose _signal_cache has the given labels."""
    from ascent.regime.engine import RegimeEngine
    engine = RegimeEngine()
    idx = pd.date_range("2026-01-01", periods=len(labels), freq="B")
    if risk_mults is None:
        risk_mults = [1.0] * len(labels)
    engine._signal_cache = pd.DataFrame({
        "label": labels,
        "risk_multiplier": risk_mults,
        "prob": [0.9] * len(labels),
    }, index=idx)
    engine._fitted = True
    return engine


def test_blend_with_ai_noop_when_agreeing():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"])
                engine.blend_with_ai(
                    {"label": "calm_bull", "confidence": 0.9, "reasoning": "Agree"},
                    as_of_date="2026-06-09",
                )
                last_label = engine._signal_cache.iloc[-1]["label"]
        assert last_label == "calm_bull"


def test_blend_with_ai_changes_label_on_strong_disagreement():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        # alpha=0.55 with confidence=1.0 → pull_weight=0.55 > 0.50 → label override.
        # Patch MAX_ALPHA to 1.0 so the production cap (0.30) doesn't clamp alpha in tests,
        # allowing us to exercise the label-override branch without changing production limits.
        state_p.write_text(json.dumps({"alpha": 0.55, "history": []}))
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                with patch.object(_eng, "AI_BLEND_MAX_ALPHA", 1.0):
                    engine = _make_engine_with_cache(["calm_bull"], risk_mults=[1.0])
                    engine._cfg["regime_risk_multiplier"] = {"stressed": 0.8}
                    engine.blend_with_ai(
                        {"label": "stressed", "confidence": 1.0, "reasoning": "Seeing stress"},
                        as_of_date="2026-06-09",
                    )
                    last_label = engine._signal_cache.iloc[-1]["label"]
        assert last_label == "stressed"


def test_blend_with_ai_invalid_label_does_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"])
                engine.blend_with_ai(
                    {"label": "alien_regime", "confidence": 0.9, "reasoning": "Bad"},
                    as_of_date="2026-06-09",
                )
                last_label = engine._signal_cache.iloc[-1]["label"]
        assert last_label == "calm_bull"


def test_blend_with_ai_skipped_when_not_fitted():
    from ascent.regime.engine import RegimeEngine
    engine = RegimeEngine()
    # _fitted is False — should return without error
    engine.blend_with_ai({"label": "calm_bull", "confidence": 0.9, "reasoning": "x"}, "2026-06-09")
    assert engine._signal_cache is None


def test_blend_logs_to_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"])
                engine.blend_with_ai(
                    {"label": "calm_bull", "confidence": 0.8, "reasoning": "test"},
                    as_of_date="2026-06-09",
                )
        assert log_p.exists()
        entry = json.loads(log_p.read_text().strip())
        assert entry["as_of_date"] == "2026-06-09"
        assert "hmm_label" in entry
        assert "ai_label" in entry
        assert "alpha" in entry
