# tests/monitoring/test_position_health.py
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import date, timedelta


def _make_agent_output(agent_id, alpha_scores):
    ao = MagicMock()
    ao.agent_id = agent_id
    ao.alpha_scores = alpha_scores
    return ao


def _make_prices_df(symbols, n_days=300, base_price=100.0):
    """Return a wide price DataFrame with n_days of business-day data."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n = len(idx)
    np.random.seed(42)
    data = {}
    for sym in symbols:
        returns = np.random.normal(0.0005, 0.01, n)
        prices = base_price * np.cumprod(1 + returns)
        data[sym] = prices
    return pd.DataFrame(data, index=idx)


def test_compute_returns_since_rebalance(tmp_path):
    from ascent.monitoring.position_health import compute_position_health

    symbols = ["AAPL", "MSFT"]
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    prices = _make_prices_df(symbols, n_days=260)

    # Write a fake wedge log so last_rebalance_date() returns a date 10 days ago
    wedge_log = tmp_path / "logs" / "alpha_wedge.jsonl"
    wedge_log.parent.mkdir(parents=True, exist_ok=True)
    rb_date = (date.today() - timedelta(days=10)).isoformat()
    wedge_log.write_text(json.dumps({
        "rebalance_date": rb_date,
        "ai_pm_weights": {}, "quant_weights": {},
        "override_types": {}, "ai_pm_return_21d": None,
        "quant_return_21d": None, "wedge_21d": None,
    }) + "\n")

    agent_outputs = [_make_agent_output("us_equities", pd.DataFrame())]

    with patch("ascent.monitoring.position_health._load_prices_wide", return_value=prices), \
         patch("ascent.monitoring.position_health._WEDGE_LOG", wedge_log):
        result = compute_position_health(date.today().isoformat(), weights, agent_outputs)

    assert "AAPL" in result
    assert "MSFT" in result
    # Return since rebalance should be populated
    assert result["AAPL"].return_since_rebalance is not None


def test_alpha_rank_computed_from_agent_outputs():
    from ascent.monitoring.position_health import compute_position_health

    symbols = ["AAPL", "MSFT", "GOOG"]
    weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}

    # Alpha scores: AAPL=0.9 (top), MSFT=0.5 (mid), GOOG=0.1 (bottom)
    alpha_row = pd.DataFrame({"AAPL": [0.9], "MSFT": [0.5], "GOOG": [0.1]})
    agent_outputs = [_make_agent_output("us_equities", alpha_row)]

    with patch("ascent.monitoring.position_health._load_prices_wide", return_value=None), \
         patch("ascent.monitoring.position_health._last_rebalance_date", return_value=None):
        result = compute_position_health(date.today().isoformat(), weights, agent_outputs)

    # AAPL has highest alpha → lowest rank_pct (best)
    assert result["AAPL"].alpha_rank_pct < result["GOOG"].alpha_rank_pct


def test_flag_deteriorating():
    from ascent.monitoring.position_health import compute_position_health

    weights = {"BAD": 0.5}
    # rank_pct > 0.60 AND return < -0.03 → DETERIORATING
    # Add many symbols with higher alpha to push BAD to bottom of universe
    other_syms = {f"GOOD{i}": [1.0 - i * 0.03] for i in range(20)}
    alpha_row = pd.DataFrame({"BAD": [0.01], **other_syms})
    agent_outputs = [_make_agent_output("us_equities", alpha_row)]

    # Prices that show -5% return since rebalance
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    n = len(idx)
    tail = min(10, n - 1)
    prices_data = pd.DataFrame({
        "BAD": [100.0] * (n - tail - 1) + [100.0] + [95.0] * tail,
    }, index=idx)

    rb_date = (date.today() - timedelta(days=10)).isoformat()

    with patch("ascent.monitoring.position_health._load_prices_wide", return_value=prices_data), \
         patch("ascent.monitoring.position_health._last_rebalance_date", return_value=rb_date):
        result = compute_position_health(date.today().isoformat(), weights, agent_outputs)

    assert result["BAD"].flag in ("DETERIORATING", "WATCH")


def test_flag_ok_for_strong_position():
    from ascent.monitoring.position_health import compute_position_health

    weights = {"GOOD": 0.5}
    # GOOD has highest alpha → lowest rank_pct, positive return
    alpha_row = pd.DataFrame({"GOOD": [1.0], "BAD1": [0.1], "BAD2": [0.2]})
    agent_outputs = [_make_agent_output("us_equities", alpha_row)]

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    n = len(idx)
    # Steadily rising prices → positive 252d momentum AND positive return since rebalance
    prices = [80.0 + i * 0.10 for i in range(n)]  # 80 → ~110 over 300 bdays
    prices_data = pd.DataFrame({"GOOD": prices}, index=idx)

    rb_date = (date.today() - timedelta(days=10)).isoformat()

    with patch("ascent.monitoring.position_health._load_prices_wide", return_value=prices_data), \
         patch("ascent.monitoring.position_health._last_rebalance_date", return_value=rb_date):
        result = compute_position_health(date.today().isoformat(), weights, agent_outputs)

    # Top alpha rank + positive return + rising momentum → must not DETERIORATE
    assert result["GOOD"].flag != "DETERIORATING"
    assert result["GOOD"].return_since_rebalance is not None
    assert result["GOOD"].return_since_rebalance > 0


def test_empty_weights_returns_empty():
    from ascent.monitoring.position_health import compute_position_health
    result = compute_position_health(date.today().isoformat(), {}, [])
    assert result == {}


def test_price_load_failure_returns_partial():
    from ascent.monitoring.position_health import compute_position_health
    weights = {"AAPL": 0.5}
    agent_outputs = [_make_agent_output("us_equities", pd.DataFrame())]
    with patch("ascent.monitoring.position_health._load_prices_wide", return_value=None), \
         patch("ascent.monitoring.position_health._last_rebalance_date", return_value=None):
        result = compute_position_health(date.today().isoformat(), weights, agent_outputs)
    assert "AAPL" in result
    assert result["AAPL"].return_since_rebalance is None
    assert result["AAPL"].flag == "OK"


def test_format_health_summary_sorts_worst_first():
    from ascent.monitoring.position_health import PositionHealth, format_health_summary
    health = {
        "A": PositionHealth("A", 0.3, -0.05, 0.1, -0.01, 0.75, "DETERIORATING"),
        "B": PositionHealth("B", 0.4, 0.02, 0.3, 0.01, 0.20, "OK"),
        "C": PositionHealth("C", 0.3, -0.02, 0.05, 0.0, 0.50, "WATCH"),
    }
    summary = format_health_summary(health)
    lines = summary.strip().split("\n")
    # DETERIORATING should come first
    assert "DETERIORATING" in lines[0]
    assert "WATCH" in lines[1]
    assert "OK" in lines[2]
