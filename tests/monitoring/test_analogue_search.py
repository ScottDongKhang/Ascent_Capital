import json
import pytest
from pathlib import Path
from ascent.monitoring.analogue_search import find_historical_analogues


def test_finds_analogues_with_matching_regime(tmp_path):
    episodes = [
        {"date": f"2024-0{i+1}-10", "regime": "calm_bull",
         "quant_weights": {"AAPL": 0.1}, "ai_weights": None,
         "realized_return_21d": 0.03 + i * 0.01}
        for i in range(5)
    ]
    p = tmp_path / "regime_episodes.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in episodes))

    trajectory = {
        "current_label": "calm_bull", "stability_10d": 0.9,
        "rs_trend": "flat", "days_in_regime": 8,
    }
    signal_health = {"trend": {"ic_5d_avg": 0.013, "status": "healthy"}}

    result = find_historical_analogues(
        "2026-05-20", trajectory, signal_health,
        episodes_path=str(p)
    )

    assert isinstance(result, list)
    assert len(result) <= 3
    for analogue in result:
        assert "date" in analogue
        assert "regime" in analogue
        assert "outcome_21d" in analogue


def test_returns_empty_without_episodes(tmp_path):
    result = find_historical_analogues(
        "2026-05-20", {}, {},
        episodes_path=str(tmp_path / "missing.jsonl")
    )
    assert result == []


def test_excludes_episodes_without_outcomes(tmp_path):
    episodes = [
        {"date": "2026-05-10", "regime": "calm_bull",
         "quant_weights": {}, "ai_weights": None,
         "realized_return_21d": None}
    ]
    p = tmp_path / "regime_episodes.jsonl"
    p.write_text(json.dumps(episodes[0]))

    result = find_historical_analogues(
        "2026-05-20", {"current_label": "calm_bull"}, {},
        episodes_path=str(p)
    )
    assert result == []
