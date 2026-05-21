import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ascent.monitoring.conviction_tracker import compute_conviction_decay


def _make_agent_output(scores: dict):
    ao = MagicMock()
    ao.agent_id = "us_equities"
    ao.alpha_scores = pd.DataFrame(
        [scores],
        index=pd.to_datetime(["2026-05-20"])
    )
    return ao


def test_conviction_decay_detects_rank_drop(tmp_path, monkeypatch):
    last_state = {
        "date": "2026-05-19",
        "weights": {"AAPL": 0.07, "MSFT": 0.07},
        "alpha_ranks": {"AAPL": 1, "MSFT": 2},
        "alpha_scores": {"AAPL": 0.9, "MSFT": 0.8},
        "sleeve_ics": {"trend": 0.014},
        "regime": "calm_bull",
        "regime_stability_10d": 0.9,
    }
    import json
    state_path = tmp_path / "last_rebalance_state.json"
    state_path.write_text(json.dumps(last_state))

    # AAPL dropped from rank 1 to rank 5, MSFT stayed near top (rank 3)
    scores = {"AAPL": 0.6, "MSFT": 0.88, "GOOG": 0.95, "META": 0.92, "AMZN": 0.85}
    agent_outputs = [_make_agent_output(scores)]
    merged_weights = {"AAPL": 0.07, "MSFT": 0.07}

    result = compute_conviction_decay(
        "2026-05-20", merged_weights, agent_outputs,
        state_path=str(state_path)
    )

    assert "AAPL" in result
    assert result["AAPL"]["rank_today"] > result["AAPL"]["rank_at_rebalance"]
    assert result["AAPL"]["rank_at_rebalance"] == 1
    assert result["MSFT"]["rank_today"] <= 3


def test_conviction_decay_returns_empty_without_state(tmp_path):
    result = compute_conviction_decay(
        "2026-05-20", {"AAPL": 0.07}, [],
        state_path=str(tmp_path / "missing.json")
    )
    assert result == {}
