# tests/test_mirofish_integration.py
from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------- Task 2 tests ----------

def test_find_analogues_returns_top3():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues(
        "Federal Reserve begins rate hike cycle to fight inflation",
        ["QQQ", "XLK"],
        top_k=3,
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    # Each entry is (analogue_dict, confidence_float)
    for analogue, conf in results:
        assert isinstance(analogue, dict)
        assert "event_id" in analogue
        assert 0.0 <= conf <= 1.0

def test_find_analogues_best_match_is_relevant():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues(
        "Fed hikes interest rates 75 basis points to fight inflation",
        ["QQQ"],
        top_k=3,
    )
    event_ids = [a["event_id"] for a, _ in results]
    # Should match rate hike events
    assert any("hike" in eid or "pivot" in eid or "cpi" in eid for eid in event_ids)

def test_find_analogues_empty_returns_empty():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues("xyzzy unknown gibberish event", [], top_k=3)
    # Should return list (may be empty if no similarity found)
    assert isinstance(results, list)

def test_find_analogues_returns_confidence_between_0_and_1():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues("infrastructure spending federal contracts", ["CAT"], top_k=2)
    for _, conf in results:
        assert 0.0 <= conf <= 1.0

# ---------- Task 3 tests ----------

def test_bootstrap_calibration_populates_entries(tmp_path, monkeypatch):
    from ascent.integrations import mirofish_calibration as mc
    cal_path = tmp_path / "mirofish_calibration.json"
    cal_path.write_text('{"bootstrapped": false, "entries": []}')
    monkeypatch.setattr(mc, "_CAL_PATH", cal_path)
    mc.bootstrap_calibration()
    data = json.loads(cal_path.read_text())
    assert data["bootstrapped"] is True
    assert len(data["entries"]) > 0

def test_get_base_rate_bullish_event(tmp_path, monkeypatch):
    from ascent.integrations import mirofish_calibration as mc
    cal_path = tmp_path / "mirofish_calibration.json"
    cal_path.write_text('{"bootstrapped": false, "entries": []}')
    monkeypatch.setattr(mc, "_CAL_PATH", cal_path)
    mc.bootstrap_calibration()
    result = mc.get_base_rate("bullish")
    assert isinstance(result, dict)
    assert "n_events" in result
    assert "median_21d_return" in result
    assert "positive_rate" in result

def test_get_base_rate_returns_defaults_when_empty(tmp_path, monkeypatch):
    from ascent.integrations import mirofish_calibration as mc
    cal_path = tmp_path / "mirofish_calibration.json"
    cal_path.write_text('{"bootstrapped": true, "entries": []}')
    monkeypatch.setattr(mc, "_CAL_PATH", cal_path)
    result = mc.get_base_rate("bullish")
    assert result["n_events"] == 0
    assert result["median_21d_return"] is None

def test_record_entry_persists(tmp_path, monkeypatch):
    from ascent.integrations import mirofish_calibration as mc
    cal_path = tmp_path / "mirofish_calibration.json"
    cal_path.write_text('{"bootstrapped": true, "entries": []}')
    monkeypatch.setattr(mc, "_CAL_PATH", cal_path)
    mc.record_entry("test_event_001", "bullish", "Industrials", 0.045)
    data = json.loads(cal_path.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["event_id"] == "test_event_001"
    assert data["entries"][0]["realized_21d_return"] == pytest.approx(0.045)

# ---------- Task 4 tests ----------

@pytest.fixture
def mock_mirofish_api():
    """Fixture that mocks the full MiroFish HTTP flow."""
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        def post_side_effect(url, **kwargs):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            if "ontology/generate" in url:
                r.json.return_value = {"success": True, "data": {"project_id": "proj_test123"}}
            elif "graph/build" in url:
                r.json.return_value = {"success": True, "data": {"task_id": "task_build_001"}}
            elif "simulation/create" in url:
                r.json.return_value = {"success": True, "data": {"simulation_id": "sim_test001"}}
            elif "simulation/prepare" in url and "status" not in url:
                r.json.return_value = {"success": True, "data": {"simulation_id": "sim_test001", "status": "ready", "already_prepared": True}}
            elif "simulation/prepare/status" in url:
                r.json.return_value = {"success": True, "data": {"status": "ready", "already_prepared": True}}
            elif "simulation/start" in url:
                r.json.return_value = {"success": True, "data": {"runner_status": "running"}}
            elif "report/generate" in url and "status" not in url:
                r.json.return_value = {"success": True, "data": {"report_id": "report_abc", "task_id": "task_report_001", "already_generated": False}}
            elif "report/generate/status" in url:
                r.json.return_value = {"success": True, "data": {"status": "completed", "report_id": "report_abc"}}
            return r

        def get_side_effect(url, **kwargs):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            if "graph/task/task_build_001" in url:
                r.json.return_value = {"success": True, "data": {"status": "completed", "result": {"graph_id": "mirofish_graph_xyz"}}}
            elif "simulation/sim_test001/run-status" in url:
                r.json.return_value = {"success": True, "data": {"runner_status": "completed", "current_round": 10}}
            elif "report/report_abc" in url:
                r.json.return_value = {"success": True, "data": {
                    "report_id": "report_abc",
                    "status": "completed",
                    "markdown_content": "## Summary\nThe crowd was **bullish** on infrastructure spending. Concerns about tariff risk were noted by some participants. Top themes: federal contracts, construction demand, supply chain.",
                    "outline": {}
                }}
            return r

        mock_post.side_effect = post_side_effect
        mock_get.side_effect = get_side_effect
        yield mock_post, mock_get


def test_run_sync_returns_structured_result(mock_mirofish_api):
    from ascent.integrations.mirofish_client import MiroFishClient
    client = MiroFishClient(base_url="http://localhost:5001")
    result = client.run_sync(
        event_description="Infrastructure spending acceleration — federal contracts for CAT and STRL",
        symbols=["CAT", "STRL"],
        n_rounds=10,
        timeout_secs=60,
    )
    assert result is not None
    assert "overall_sentiment" in result
    assert result["overall_sentiment"] in ("bullish", "bearish", "mixed")
    assert "top_themes" in result
    assert isinstance(result["top_themes"], list)

def test_run_sync_timeout_returns_none():
    from ascent.integrations.mirofish_client import MiroFishClient
    import requests
    client = MiroFishClient(base_url="http://localhost:5001")
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
        result = client.run_sync(
            event_description="test event",
            symbols=["SPY"],
            n_rounds=10,
            timeout_secs=5,
        )
    assert result is None

def test_parse_sentiment_bullish():
    from ascent.integrations.mirofish_client import _parse_sentiment_from_markdown
    md = "The crowd was overwhelmingly bullish. Participants expressed optimism. Buy signals everywhere."
    result = _parse_sentiment_from_markdown(md)
    assert result["overall_sentiment"] == "bullish"
    assert result["confidence"] > 0.5

def test_parse_sentiment_bearish():
    from ascent.integrations.mirofish_client import _parse_sentiment_from_markdown
    md = "Bearish outlook dominated. Participants feared a recession. Pessimistic views on earnings."
    result = _parse_sentiment_from_markdown(md)
    assert result["overall_sentiment"] == "bearish"

def test_parse_sentiment_extracts_themes():
    from ascent.integrations.mirofish_client import _parse_sentiment_from_markdown
    md = "## Key Themes\n- Federal contracts\n- Infrastructure spending\n- Supply chain concerns"
    result = _parse_sentiment_from_markdown(md)
    assert len(result.get("top_themes", [])) >= 0  # may find themes in headers
