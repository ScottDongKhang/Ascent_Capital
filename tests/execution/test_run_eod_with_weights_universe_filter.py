"""
Regression test for a bug found by independent code review: run_eod_with_weights()
(the discovery / mini-rebalance / orchestrator-merged-weights path used by
run_all_agents.py) converted merged_weights straight to a pd.Series with no
universe filter and no max-weight re-cap -- unlike run_eod(), which already
runs both via filter_to_tradeable_universe() (see
tests/execution/test_universe_filter_cap.py).

This left run_eod_with_weights() able to select and size symbols the
walk-forward backtest never validated on (the "live trading and the
walk-forward backtest must draw from the same universe" gotcha in
CLAUDE.md), and able to submit orders that breach integrity constraint #3
(the max-weight hard cap) once a dropped symbol's weight redistributes onto
survivors.

These tests exercise run_eod_with_weights() end to end (with all external
I/O -- Alpaca, config, universe construction, order submission -- mocked
out) and assert the target_weights actually handed to compute_orders() has
been filtered to the tradeable universe and re-capped, exactly like
run_eod()'s.
"""
import pandas as pd
import pytest

import ascent.execution.eod_runner as runner
import ascent.execution.order_engine as order_engine
import ascent.config.settings as settings
import ascent.data.universe as universe
import ascent.utils.market_time as market_time
import ascent.execution.alpaca_broker as alpaca_broker


MAX_WEIGHT = 0.10


class _FakeBacktestCfg:
    max_weight = MAX_WEIGHT


class _FakeCfg:
    backtest = _FakeBacktestCfg()


def _merged_weights_with_out_of_universe_symbol():
    """
    11 in-universe survivors + 1 out-of-universe symbol ("ZZZ"), mirroring
    tests/execution/test_universe_filter_cap.py's setup: ZZZ's weight
    redistributing onto BIG (0.09 pre-filter, under the 10% cap) pushes BIG
    over the cap once the filtered book renormalizes, so this also exercises
    the re-cap step, not just the drop.
    """
    small = {f"S{i}": 0.05 for i in range(10)}  # sum 0.50
    weights = {**small, "BIG": 0.09, "ZZZ": 0.41}  # total 1.00
    tradeable = set(small) | {"BIG"}
    return weights, tradeable


@pytest.fixture(autouse=True)
def _mock_external_dependencies(monkeypatch):
    """
    Isolate run_eod_with_weights()'s universe-filter/re-cap logic from
    everything it depends on but doesn't own: config, the historical
    universe builder, market-time-aware "today", the Alpaca broker, and
    logging to disk.
    """
    _, tradeable = _merged_weights_with_out_of_universe_symbol()

    monkeypatch.setattr(settings, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(universe, "build_historical_universe", lambda **kw: object())
    monkeypatch.setattr(universe, "get_universe_on_date", lambda today, universe_df: tradeable)
    monkeypatch.setattr(market_time, "market_today", lambda: __import__("datetime").date(2026, 8, 25))

    monkeypatch.setattr(alpaca_broker, "get_positions", lambda: pd.DataFrame())
    monkeypatch.setattr(alpaca_broker, "get_portfolio_value", lambda: 100_000.0)

    # Avoid touching logs/eod_log.jsonl on disk.
    monkeypatch.setattr(runner, "_log_multi_run", lambda *a, **kw: None)


def test_run_eod_with_weights_filters_universe_and_recaps_max_weight(monkeypatch):
    merged_weights, tradeable = _merged_weights_with_out_of_universe_symbol()

    captured = {}

    def _fake_compute_orders(target_weights, current_positions, portfolio_value, features=None):
        captured["target_weights"] = target_weights
        # No orders needed -- keeps the test focused on the filter/re-cap
        # step, not order computation or submission.
        return [], pd.DataFrame()

    monkeypatch.setattr(order_engine, "compute_orders", _fake_compute_orders)

    runner.run_eod_with_weights(merged_weights, run_date="2026-08-25", force=True)

    assert "target_weights" in captured, "compute_orders was never called"
    result = captured["target_weights"]

    assert "ZZZ" not in result.index, (
        "ZZZ is not in build_historical_universe(strict=True, sp500_only=True) "
        "and must be filtered out before orders are computed"
    )
    assert set(result.index) == tradeable
    assert result.max() <= MAX_WEIGHT + 1e-9, (
        "BIG's weight must be re-capped after ZZZ's dropped weight "
        "redistributes onto survivors on renormalization"
    )
    assert result.sum() == pytest.approx(1.0)


def test_run_eod_with_weights_passes_through_when_all_symbols_tradeable(monkeypatch):
    """When nothing needs to be dropped or re-capped, weights pass through
    unchanged (aside from the Series conversion) -- confirms the new filter
    step is not itself distorting a clean book."""
    small = {f"S{i}": 0.10 for i in range(10)}  # sum 1.0, all under cap
    tradeable = set(small)

    import ascent.data.universe as _universe_mod
    monkeypatch.setattr(_universe_mod, "get_universe_on_date", lambda today, universe_df: tradeable)

    captured = {}

    def _fake_compute_orders(target_weights, current_positions, portfolio_value, features=None):
        captured["target_weights"] = target_weights
        return [], pd.DataFrame()

    monkeypatch.setattr(order_engine, "compute_orders", _fake_compute_orders)

    runner.run_eod_with_weights(small, run_date="2026-08-25", force=True)

    result = captured["target_weights"]
    assert set(result.index) == tradeable
    assert result.max() <= MAX_WEIGHT + 1e-9
    assert result.sum() == pytest.approx(1.0)


def test_run_eod_with_weights_logs_dropped_symbols(monkeypatch, capsys):
    merged_weights, _ = _merged_weights_with_out_of_universe_symbol()

    monkeypatch.setattr(
        order_engine, "compute_orders",
        lambda target_weights, current_positions, portfolio_value, features=None: ([], pd.DataFrame()),
    )

    runner.run_eod_with_weights(merged_weights, run_date="2026-08-25", force=True)

    out = capsys.readouterr().out
    assert "filter_to_tradeable_universe" in out
    assert "ZZZ" in out
    assert "1 dropped" in out
