import json
from pathlib import Path
from unittest.mock import patch
from ascent.monitoring.daily_intelligence import run_daily_intelligence


def test_writes_daily_entry(tmp_path):
    merged_weights = {"AAPL": 0.07, "MSFT": 0.07}
    agent_outputs  = []

    with patch("ascent.monitoring.daily_intelligence.compute_conviction_decay", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.compute_signal_health", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.compute_regime_trajectory", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.find_historical_analogues", return_value=[]), \
         patch("ascent.monitoring.daily_intelligence.update_position_theses", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.generate_adversarial_challenge", return_value="test challenge"), \
         patch("ascent.monitoring.daily_intelligence.build_event_calendar", return_value=[]):

        run_daily_intelligence(
            "2026-05-20", merged_weights, agent_outputs,
            output_dir=str(tmp_path)
        )

    out = tmp_path / "2026-05-20.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["date"] == "2026-05-20"
    assert "conviction_decay" in data
    assert "adversarial_challenge" in data
    assert data["adversarial_challenge"] == "test challenge"


def test_module_failure_does_not_block(tmp_path):
    with patch("ascent.monitoring.daily_intelligence.compute_conviction_decay", side_effect=RuntimeError("boom")), \
         patch("ascent.monitoring.daily_intelligence.compute_signal_health", return_value={"trend": {}}), \
         patch("ascent.monitoring.daily_intelligence.compute_regime_trajectory", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.find_historical_analogues", return_value=[]), \
         patch("ascent.monitoring.daily_intelligence.update_position_theses", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.generate_adversarial_challenge", return_value=""), \
         patch("ascent.monitoring.daily_intelligence.build_event_calendar", return_value=[]):

        run_daily_intelligence("2026-05-20", {"AAPL": 0.07}, [], output_dir=str(tmp_path))

    out = tmp_path / "2026-05-20.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["signal_health"] == {"trend": {}}
    assert data["conviction_decay"] == {}
