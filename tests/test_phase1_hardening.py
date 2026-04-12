# tests/test_phase1_hardening.py
import json
import pytest
from datetime import date, timedelta
from unittest.mock import patch


def test_skill_tracker_writes_as_of_date(tmp_path):
    """export_skill_scores() must write skill_score_as_of to the JSON payload."""
    from ascent.monitoring import skill_tracker

    with patch.object(skill_tracker, "SKILL_OUTPUT_PATH", tmp_path / "agent_skill_scores.json"), \
         patch.object(skill_tracker, "SKILL_LOG_PATH", tmp_path / "skill_scores_log.jsonl"), \
         patch.object(skill_tracker, "compute_all_skill_scores", return_value={}):
        skill_tracker.export_skill_scores()

    payload = json.loads((tmp_path / "agent_skill_scores.json").read_text())
    assert "skill_score_as_of" in payload, "skill_score_as_of key missing from output"
    assert payload["skill_score_as_of"] == date.today().isoformat()


def test_orchestrator_rejects_stale_skill_scores(tmp_path):
    """_load_skill_scores() returns {} when skill_score_as_of is more than 1 day old."""
    import json
    from orchestrator import central_intelligence as ci

    stale_date = (date.today() - timedelta(days=3)).isoformat()
    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-09T14:00:00",
        "skill_score_as_of": stale_date,
        "agents": {
            "us_equities": {"skill_score": 1.23, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {}, f"Expected empty dict for stale scores, got {result}"


def test_orchestrator_accepts_fresh_skill_scores(tmp_path):
    """_load_skill_scores() returns scores when skill_score_as_of is today."""
    import json
    from orchestrator import central_intelligence as ci

    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-12T14:00:00",
        "skill_score_as_of": date.today().isoformat(),
        "agents": {
            "us_equities": {"skill_score": 1.23, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {"us_equities": 1.23}


def test_orchestrator_accepts_yesterday_scores_on_monday(tmp_path):
    """_load_skill_scores() returns scores when as_of is 1 day ago (weekend buffer)."""
    import json
    from orchestrator import central_intelligence as ci

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-12T14:00:00",
        "skill_score_as_of": yesterday,
        "agents": {
            "us_equities": {"skill_score": 0.80, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {"us_equities": 0.80}
