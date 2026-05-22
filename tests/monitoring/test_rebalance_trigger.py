import json
import pytest
from pathlib import Path
from unittest.mock import patch
import pandas as pd


def _make_state(date: str, ics: dict) -> dict:
    return {
        "date": date,
        "sleeve_ics": ics,
        "composite_ic": sum(ics.values()) / len(ics),
    }


def test_trigger_fires_on_ic_decay(tmp_path):
    from ascent.monitoring.rebalance_trigger import check_ic_decay_trigger

    state_path = tmp_path / "last_rebalance_state.json"
    trigger_path = tmp_path / "rebalance_trigger.json"

    # Baseline: healthy ICs
    state = _make_state("2026-05-10", {"trend": 0.10, "ml": 0.08, "statarb": 0.05})
    state_path.write_text(json.dumps(state))

    # Current: ICs have dropped 50%
    current_ics = {"trend": 0.05, "ml": 0.04, "statarb": 0.025}

    result = check_ic_decay_trigger(
        date="2026-05-19",          # 7 business days after 2026-05-10
        current_ics=current_ics,
        state_path=state_path,
        trigger_path=trigger_path,
        decay_threshold=0.30,
        min_days=5,
    )

    assert result is True
    assert trigger_path.exists()
    data = json.loads(trigger_path.read_text())
    assert "triggered_date" in data
    assert data["ic_decay_pct"] > 0.30


def test_trigger_does_not_fire_when_healthy(tmp_path):
    from ascent.monitoring.rebalance_trigger import check_ic_decay_trigger

    state_path = tmp_path / "last_rebalance_state.json"
    trigger_path = tmp_path / "rebalance_trigger.json"

    state = _make_state("2026-05-10", {"trend": 0.10, "ml": 0.08})
    state_path.write_text(json.dumps(state))

    # ICs only dropped 10% — below threshold
    current_ics = {"trend": 0.09, "ml": 0.073}

    result = check_ic_decay_trigger(
        date="2026-05-19",
        current_ics=current_ics,
        state_path=state_path,
        trigger_path=trigger_path,
        decay_threshold=0.30,
        min_days=5,
    )

    assert result is False
    assert not trigger_path.exists()


def test_trigger_does_not_fire_too_soon(tmp_path):
    from ascent.monitoring.rebalance_trigger import check_ic_decay_trigger

    state_path = tmp_path / "last_rebalance_state.json"
    trigger_path = tmp_path / "rebalance_trigger.json"

    state = _make_state("2026-05-19", {"trend": 0.10})   # rebalanced TODAY
    state_path.write_text(json.dumps(state))

    # ICs collapsed
    current_ics = {"trend": 0.02}

    result = check_ic_decay_trigger(
        date="2026-05-20",          # only 1 business day later
        current_ics=current_ics,
        state_path=state_path,
        trigger_path=trigger_path,
        decay_threshold=0.30,
        min_days=5,
    )

    assert result is False


def test_consume_trigger_deletes_file(tmp_path):
    from ascent.monitoring.rebalance_trigger import consume_trigger

    trigger_path = tmp_path / "rebalance_trigger.json"
    trigger_path.write_text(json.dumps({"triggered_date": "2026-05-19"}))

    assert consume_trigger(trigger_path=trigger_path) is True
    assert not trigger_path.exists()

    assert consume_trigger(trigger_path=trigger_path) is False
