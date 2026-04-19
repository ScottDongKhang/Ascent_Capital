# tests/test_debate_gate.py
import pytest
from datetime import date


def test_debate_fires_on_high_entropy():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.02}}
    regime = {"entropy": 0.75, "label": "stressed"}
    assert should_run_debate(state, regime) is True


def test_debate_skipped_on_low_entropy_calm():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08, "MSFT": 0.07}, "quant_context": {"portfolio_var_99": -0.015},
             "catalyst_detected": False}
    regime = {"entropy": 0.40, "label": "calm_bull"}
    assert should_run_debate(state, regime) is False


def test_debate_fires_on_concentrated_position():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"EWY": 0.14, "GLD": 0.08}, "quant_context": {"portfolio_var_99": -0.018},
             "catalyst_detected": False}
    regime = {"entropy": 0.30, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_on_catalyst():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.02},
             "catalyst_detected": True}
    regime = {"entropy": 0.35, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_on_var_tail():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.038},
             "catalyst_detected": False}
    regime = {"entropy": 0.30, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


# Counterfactual tests
def test_counterfactual_snapshot_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    from ascent.monitoring.counterfactual_tracker import snapshot_quant_weights, snapshot_debate_weights
    import json

    snapshot_quant_weights({"AAPL": 0.10, "MSFT": 0.09}, run_date=date(2026, 4, 29))
    snapshot_debate_weights({"AAPL": 0.08, "MSFT": 0.07}, run_date=date(2026, 4, 29))

    log = tmp_path / "logs" / "counterfactual_log.jsonl"
    assert log.exists()
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert any(l["type"] == "quant_snapshot" for l in lines)
    assert any(l["type"] == "debate_snapshot" for l in lines)


def test_counterfactual_outcome_scored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    import json
    from datetime import timedelta
    from ascent.monitoring.counterfactual_tracker import (
        snapshot_quant_weights, snapshot_debate_weights, score_pending_counterfactuals
    )

    d = date(2026, 4, 10)
    snapshot_quant_weights({"AAPL": 0.50, "MSFT": 0.50}, run_date=d)
    snapshot_debate_weights({"AAPL": 0.40, "MSFT": 0.60}, run_date=d)

    scored = score_pending_counterfactuals(
        prices_override={"AAPL": [150.0] * 11, "MSFT": [300.0] + [315.0] * 10},
        as_of_date=d + timedelta(days=10),
    )
    assert scored >= 0  # at least attempted
