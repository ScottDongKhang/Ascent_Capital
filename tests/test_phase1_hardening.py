# tests/test_phase1_hardening.py
import json
import pytest
from datetime import date
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
