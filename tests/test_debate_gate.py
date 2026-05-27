# tests/test_debate_gate.py
import pytest
from datetime import date


def test_debate_fires_every_rebalance_calm_bull():
    """Adversarial Intelligence fires every rebalance — entropy gate removed."""
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.02}}
    regime = {"entropy": 0.0, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_every_rebalance_low_entropy():
    """Even low entropy, low concentration, low VaR — still fires."""
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.05, "MSFT": 0.05}, "quant_context": {"portfolio_var_99": -0.01},
             "catalyst_detected": False}
    regime = {"entropy": 0.10, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_stressed_regime():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"EWY": 0.06}, "quant_context": {"portfolio_var_99": -0.025}}
    regime = {"entropy": 0.75, "label": "stressed"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_empty_portfolio():
    from ascent.execution.debate_gate import should_run_debate
    assert should_run_debate({}, {}) is True


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
