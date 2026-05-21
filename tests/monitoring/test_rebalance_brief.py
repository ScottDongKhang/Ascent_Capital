import json
from pathlib import Path
from unittest.mock import patch
from ascent.monitoring.rebalance_brief import generate_rebalance_brief


def _write_intel_entries(tmp_path, n=3):
    d = tmp_path / "daily_intelligence"
    d.mkdir()
    for i in range(n):
        date_str = f"2026-05-{17+i:02d}"
        entry = {
            "date": date_str,
            "conviction_decay": {"AAPL": {"rank_at_rebalance": 1, "rank_today": 3}},
            "signal_health": {"trend": {"status": "healthy", "change_pct": -5.0}},
            "regime_trajectory": {"current_label": "calm_bull", "stability_10d": 0.9},
            "historical_analogues": [{"date": "2024-03-01", "outcome_21d": 0.03}],
            "position_theses": {"AAPL": "Thesis intact."},
            "adversarial_challenge": f"Risk #{i+1}: crowding.",
            "macro_events": [{"event": "FOMC", "date": "2026-05-27", "sensitivity": 4}],
        }
        (d / f"{date_str}.json").write_text(json.dumps(entry))
    return str(d)


def test_generates_brief_from_entries(tmp_path):
    intel_dir  = _write_intel_entries(tmp_path)
    brief_path = str(tmp_path / "rebalance_brief.json")

    with patch("ascent.monitoring.rebalance_brief.generate_structured") as mock_llm:
        mock_llm.return_value = "The portfolio enters rebalance in a stable calm_bull regime."
        generate_rebalance_brief(
            "2026-05-20",
            intel_dir=intel_dir,
            brief_path=brief_path,
        )

    assert Path(brief_path).exists()
    data = json.loads(Path(brief_path).read_text())
    assert data["date"] == "2026-05-20"
    assert "synthesis" in data
    assert "stale_positions" in data
    assert "weakening_sleeves" in data
    assert mock_llm.called


def test_returns_empty_brief_without_entries(tmp_path):
    empty_dir  = str(tmp_path / "daily_intelligence")
    Path(empty_dir).mkdir()
    brief_path = str(tmp_path / "brief.json")

    result = generate_rebalance_brief("2026-05-20", intel_dir=empty_dir, brief_path=brief_path)
    assert result["synthesis"] == ""
    assert result["stale_positions"] == []
