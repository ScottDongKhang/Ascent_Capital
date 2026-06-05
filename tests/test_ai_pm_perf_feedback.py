"""
tests/test_ai_pm_perf_feedback.py
Tests for ascent/strategy/ai_pm_perf_feedback.py
"""
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch


def _make_decision(sym="STRL", ai_w=0.09, quant_w=0.07, days_ago=12, ov_type="amplify"):
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return {
        "date": d, "level": 1, "ai_weight": 0.05,
        "overrides_applied": [{"symbol": sym, "type": ov_type, "ai_w": ai_w, "quant_w": quant_w}],
    }


def test_compute_feedback_writes_file():
    import ascent.strategy.ai_pm_perf_feedback as fb_mod
    with tempfile.TemporaryDirectory() as tmp:
        fb_path  = Path(tmp) / "feedback.json"
        dec_path = Path(tmp) / "decisions.jsonl"
        cf_path  = Path(tmp) / "cf_daily.jsonl"
        dec_path.write_text(json.dumps(_make_decision()) + "\n")
        with patch.object(fb_mod, "FEEDBACK_PATH", fb_path):
            with patch.object(fb_mod, "DECISION_LOG", dec_path):
                with patch.object(fb_mod, "DAILY_LOG", cf_path):
                    with patch.object(fb_mod, "_fetch_price_return", return_value=0.03):
                        fb = fb_mod.compute_feedback()
        assert fb_path.exists()
        data = json.loads(fb_path.read_text())
        assert "level" in data
        assert "promotion_gates" in data
        assert "n_decisions_evaluated" in data


def test_incremental_alpha_uses_delta_weight():
    from ascent.strategy.ai_pm_perf_feedback import _incremental_alpha
    # AI PM 9%, quant 7%, stock +10% → incremental = (0.09-0.07)*0.10 = 0.002
    result = _incremental_alpha(ai_w=0.09, quant_w=0.07, stock_return=0.10)
    assert abs(result - 0.002) < 0.0001


def test_incremental_alpha_short_position():
    from ascent.strategy.ai_pm_perf_feedback import _incremental_alpha
    # AI PM shorts -4%, quant short -2%, stock -8% → incremental = (-0.04-(-0.02))*(-0.08) = +0.0016
    result = _incremental_alpha(ai_w=-0.04, quant_w=-0.02, stock_return=-0.08)
    assert result > 0  # short thesis correct → positive incremental alpha


def test_fade_detection():
    from ascent.strategy.ai_pm_perf_feedback import _is_fade
    assert _is_fade(outcome_10d=0.02, outcome_21d=-0.01) is True
    assert _is_fade(outcome_10d=0.02, outcome_21d=0.03)  is False
    assert _is_fade(outcome_10d=-0.01, outcome_21d=-0.02) is False  # not a fade if negative at 10d
    assert _is_fade(outcome_10d=None, outcome_21d=-0.01) is False   # no verdict yet


def test_confidence_label():
    from ascent.strategy.ai_pm_perf_feedback import _confidence
    assert _confidence(n=2)  == "low"
    assert _confidence(n=8)  == "medium"
    assert _confidence(n=20) == "high"


def test_stuck_alert_in_feedback():
    import ascent.strategy.ai_pm_perf_feedback as fb_mod
    fake_state = {
        "level": 1, "title": "Analyst", "ai_weight": 0.05,
        "days_at_level": 70, "days_stuck": 70,
        "in_cooldown": False, "cooldown_until": None,
        "track_d_returns": [], "track_astar_returns": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        fb_path  = Path(tmp) / "feedback.json"
        dec_path = Path(tmp) / "decisions.jsonl"
        cf_path  = Path(tmp) / "cf_daily.jsonl"
        dec_path.write_text("")
        with patch.object(fb_mod, "FEEDBACK_PATH", fb_path):
            with patch.object(fb_mod, "DECISION_LOG", dec_path):
                with patch.object(fb_mod, "DAILY_LOG", cf_path):
                    with patch.object(fb_mod, "_load_authority_state", return_value=fake_state):
                        fb = fb_mod.compute_feedback()
        assert fb["stuck_alert"] is True
        assert fb["days_stuck"] == 70
