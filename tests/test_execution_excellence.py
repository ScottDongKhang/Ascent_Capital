"""tests/test_execution_excellence.py

16 tests for Plan 5 — Execution Excellence.
All tests mock Alpaca/network calls.
"""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Task 1: TWAP executor ─────────────────────────────────────────────────────

def test_twap_schedule_correct_slice_count():
    """build_twap_schedule returns exactly n_slices orders."""
    from ascent.execution.twap_executor import build_twap_schedule
    schedule = build_twap_schedule("AAPL", 60, "buy", datetime.now(), window_minutes=30, n_slices=6)
    assert len(schedule) == 6


def test_twap_schedule_shares_sum_to_total():
    """Sum of child order shares equals total_shares."""
    from ascent.execution.twap_executor import build_twap_schedule
    total = 100
    schedule = build_twap_schedule("MSFT", total, "sell", datetime.now(), window_minutes=60, n_slices=5)
    assert sum(s["shares"] for s in schedule) == total


def test_twap_schedule_minimum_slice_size():
    """Every slice has shares >= 1 (no zero-share orders)."""
    from ascent.execution.twap_executor import build_twap_schedule
    # 3 total shares with n_slices=10 → should reduce n_slices to 3
    schedule = build_twap_schedule("TSLA", 3, "buy", datetime.now(), n_slices=10)
    assert all(s["shares"] >= 1 for s in schedule)
    assert len(schedule) == 3  # reduced to match total_shares


def test_compute_twap_window_respects_urgency():
    """High urgency → window ≤ 15 minutes."""
    from ascent.execution.twap_executor import compute_twap_window
    window = compute_twap_window(adv=100_000, trade_size_shares=5_000, urgency="high")
    assert window <= 15


def test_should_use_twap_true_above_threshold():
    """Trade size 6% of ADV → should_use_twap returns True."""
    from ascent.execution.twap_executor import should_use_twap
    adv = 1_000_000
    trade = 60_000  # 6% of ADV
    assert should_use_twap(trade, adv, threshold_pct=0.05) is True


def test_should_use_twap_false_below_threshold():
    """Trade size 3% of ADV → should_use_twap returns False."""
    from ascent.execution.twap_executor import should_use_twap
    adv   = 1_000_000
    trade = 30_000  # 3% of ADV
    assert should_use_twap(trade, adv, threshold_pct=0.05) is False


def test_twap_disabled_uses_market_order():
    """TWAP_ENABLED=False → execute_twap logs only, returns 'disabled' status."""
    import ascent.execution.twap_executor as te
    original = te.TWAP_ENABLED
    te.TWAP_ENABLED = False
    try:
        result = te.execute_twap("AAPL", 10_000, "buy", adv=500_000, price=150.0)
        assert len(result) == 1
        assert result[0]["status"] == "disabled"
    finally:
        te.TWAP_ENABLED = original


# ── Task 2: Implementation shortfall ─────────────────────────────────────────

def test_is_delay_cost_positive_on_adverse_move():
    """Price moves up before submission (buy order) → delay_cost_bps > 0."""
    from ascent.execution.implementation_shortfall import compute_is
    result = compute_is(
        symbol="AAPL",
        fill_records=[{"fill_price": 155.0, "shares": 100}],
        eod_price=156.0,
        decision_price=150.0,
        arrival_price=153.0,
        side="buy",
    )
    assert result["delay_cost_bps"] > 0
    # delay = (153 - 150) / 150 * 10000 = 200 bps
    assert abs(result["delay_cost_bps"] - 200.0) < 1.0


def test_is_market_impact_positive_on_adverse_fill():
    """Fill worse than arrival price (buy) → market_impact_bps > 0."""
    from ascent.execution.implementation_shortfall import compute_is
    result = compute_is(
        symbol="MSFT",
        fill_records=[{"fill_price": 310.0, "shares": 50}],
        eod_price=311.0,
        decision_price=300.0,
        arrival_price=305.0,
        side="buy",
    )
    assert result["market_impact_bps"] > 0


def test_is_components_sum_to_total_is():
    """delay + impact + opportunity ≈ total_is_bps."""
    from ascent.execution.implementation_shortfall import compute_is
    result = compute_is(
        symbol="NVDA",
        fill_records=[{"fill_price": 870.0, "shares": 10}],
        eod_price=875.0,
        decision_price=860.0,
        arrival_price=865.0,
        side="buy",
    )
    total  = result["total_is_bps"]
    comps  = (result["delay_cost_bps"] or 0) + (result["market_impact_bps"] or 0) + (result["opportunity_cost_bps"] or 0)
    assert abs(total - comps) < 0.01


# ── Task 3: Capacity model ───────────────────────────────────────────────────

def test_market_impact_increases_with_size():
    """Larger trade_size_pct_adv → higher market impact bps."""
    from ascent.execution.capacity_model import estimate_market_impact
    impact_small = estimate_market_impact(0.02, volatility=0.20)
    impact_large = estimate_market_impact(0.10, volatility=0.20)
    assert impact_large > impact_small


def test_signal_breakeven_adv_positive():
    """compute_signal_breakeven_adv returns positive float for any valid IC."""
    from ascent.execution.capacity_model import compute_signal_breakeven_adv
    result = compute_signal_breakeven_adv(signal_ic=0.04)
    assert isinstance(result, float)
    assert result > 0


def test_capacity_report_returns_binding_constraint(tmp_path):
    """capacity_report result contains 'binding_constraint' key."""
    import pandas as pd
    import numpy as np
    from ascent.execution.capacity_model import capacity_report
    import ascent.execution.capacity_model as cm

    prices = pd.DataFrame({"AAPL": np.linspace(150, 160, 30)},
                          index=pd.bdate_range("2026-01-01", periods=30))
    weights = {"AAPL": 0.10, "MSFT": 0.08}

    with patch.object(cm, "_CAPACITY_LOG", tmp_path / "capacity_log.jsonl"):
        result = capacity_report(prices, weights)

    assert "binding_constraint" in result
    assert "overall_capacity" in result


# ── Task 4: Intraday trigger ─────────────────────────────────────────────────

def test_intraday_trigger_spy_crash():
    """SPY -3.5% + VIX 35 → regime_emergency trigger fires."""
    from ascent.execution.intraday_trigger import check_intraday_triggers
    portfolio_state = {"weights": {"SPY": 0.05, "AAPL": 0.10}, "nav": 100_000, "drawdown": 0.05}
    market_data     = {"spy_intraday_return": -0.035, "vix": 35.0}
    triggers = check_intraday_triggers(portfolio_state, market_data)
    trigger_types = [t["type"] for t in triggers]
    assert "regime_emergency" in trigger_types


def test_intraday_trigger_no_trigger_on_normal_day():
    """Normal market conditions → empty trigger list."""
    from ascent.execution.intraday_trigger import check_intraday_triggers
    portfolio_state = {"weights": {"AAPL": 0.10}, "nav": 100_000, "drawdown": 0.02}
    market_data     = {"spy_intraday_return": 0.005, "vix": 18.0}
    triggers = check_intraday_triggers(portfolio_state, market_data)
    assert triggers == []


# ── Task 5: Fill quality report ───────────────────────────────────────────────

def test_fill_quality_report_by_sleeve_present(tmp_path):
    """fill_quality_report returns dict with 'by_sleeve' key."""
    import ascent.execution.slippage_tracker as st
    with patch.object(st, "SLIPPAGE_LOG_PATH", tmp_path / "slippage_log.jsonl"):
        result = st.fill_quality_report(lookback_days=30)
    assert "by_sleeve" in result
    assert "n_trades" in result
