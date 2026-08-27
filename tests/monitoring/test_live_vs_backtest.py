"""
tests/monitoring/test_live_vs_backtest.py

Covers:
  1. load_live_portfolio_values() sources settled returns via
     alpaca_broker.get_portfolio_history() (not same-day EOD-log pct_change),
     and doesn't crash on thin/missing history.
  2. _divergence_significance() distinguishes a within-cost-model-noise
     synthetic divergence from a clearly-real one.
  3. The None / 0.5 / real-probability cases from probabilistic_sharpe_ratio
     are handled distinctly and don't crash formatting/logging.
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from ascent.monitoring import live_vs_backtest as lvb
from ascent.monitoring.live_vs_backtest import (
    load_live_portfolio_values,
    compute_live_returns,
    build_comparison,
    export_live_vs_backtest,
    _divergence_significance,
    DIVERGENCE_PSR_THRESHOLD,
)


@pytest.fixture(autouse=True)
def _isolated_live_returns_cache(tmp_path, monkeypatch):
    """
    Give every test its own empty live-returns cache file so tests don't
    read/write the real data_cache/ on disk or leak state into each other
    via the new incremental-fetch cache.
    """
    monkeypatch.setattr(lvb, "LIVE_RETURNS_CACHE_PATH", tmp_path / "live_portfolio_returns.csv")
    yield


# ── Fix 1: settled-NAV swap ─────────────────────────────────────────────────

def test_load_live_portfolio_values_uses_settled_history():
    fake_history = {
        "2026-08-10": 0.004,
        "2026-08-11": -0.002,
        "2026-08-12": 0.001,
    }
    with patch(
        "ascent.execution.alpaca_broker.get_portfolio_history",
        return_value=fake_history,
    ):
        s = load_live_portfolio_values()

    assert not s.empty
    assert len(s) == 3
    # These are already returns, not NAV -- compute_live_returns is a
    # passthrough now, no pct_change should be applied.
    r = compute_live_returns(s)
    assert list(r.round(6)) == [0.004, -0.002, 0.001]


def test_load_live_portfolio_values_empty_history_no_crash():
    with patch(
        "ascent.execution.alpaca_broker.get_portfolio_history",
        return_value={},
    ):
        s = load_live_portfolio_values()
    assert s.empty
    # compute_live_returns must not crash on an empty series
    r = compute_live_returns(s)
    assert r.empty


def test_load_live_portfolio_values_single_day_no_crash():
    with patch(
        "ascent.execution.alpaca_broker.get_portfolio_history",
        return_value={"2026-08-10": 0.001},
    ):
        s = load_live_portfolio_values()
    assert len(s) == 1
    r = compute_live_returns(s)
    assert len(r) == 1


# ── Fix 2: significance check ────────────────────────────────────────────────

def _dates(n, start="2026-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def test_divergence_noise_case_not_flagged():
    # Small, zero-mean, cost-model-scale noise (~1-2bps/day) around zero --
    # should NOT be flagged as a significant divergence.
    rng = np.random.default_rng(42)
    idx = _dates(120)
    diff = pd.Series(rng.normal(loc=0.0, scale=0.0005, size=120), index=idx)

    result = _divergence_significance(diff)

    assert result["psr_status"] == "computed"
    assert result["psr_vs_zero"] is not None
    assert result["diverged_significantly"] is False


def test_divergence_real_case_flagged():
    # A persistent, strongly positive daily divergence -- should be flagged.
    idx = _dates(120)
    diff = pd.Series(np.full(120, 0.01) + np.random.default_rng(1).normal(0, 0.0005, 120), index=idx)

    result = _divergence_significance(diff)

    assert result["psr_status"] == "computed"
    assert result["psr_vs_zero"] is not None
    assert result["psr_vs_zero"] >= DIVERGENCE_PSR_THRESHOLD
    assert result["diverged_significantly"] is True


def test_divergence_cost_model_baseline_documented_in_output():
    idx = _dates(60)
    diff = pd.Series(np.random.default_rng(2).normal(0, 0.0003, 60), index=idx)
    result = _divergence_significance(diff)
    assert "cost_model_baseline_bps" in result
    assert result["cost_model_baseline_bps"]["low_participation"] > 0
    assert result["cost_model_baseline_bps"]["high_participation"] > \
        result["cost_model_baseline_bps"]["low_participation"]


# ── Fix 2b: None / 0.5 / real-probability handling ───────────────────────────

def test_divergence_uninformative_n_short_series():
    idx = _dates(3)
    diff = pd.Series([0.001, -0.001, 0.0005], index=idx)
    result = _divergence_significance(diff)
    assert result["psr_status"] == "uninformative_n"
    assert result["diverged_significantly"] is False
    # Must not crash formatting/logging even with psr_vs_zero possibly None
    _ = f"{result}"


def test_divergence_zero_variance_series_uninformative():
    idx = _dates(10)
    diff = pd.Series([0.0] * 10, index=idx)
    result = _divergence_significance(diff)
    assert result["psr_status"] == "uninformative_n"
    assert result["diverged_significantly"] is False


def test_divergence_formula_degenerate_case_not_coerced_to_half():
    # Construct inputs that drive the PSR denominator non-positive:
    # large positive skew + large observed Sharpe, per deflated_sharpe.py's
    # own docstring example (skew=+2.0, sharpe_observed=+3.0).
    from ascent.monitoring import live_vs_backtest as lvb

    with patch.object(lvb, "probabilistic_sharpe_ratio", return_value=None):
        idx = _dates(30)
        diff = pd.Series(np.random.default_rng(3).normal(0.001, 0.0002, 30), index=idx)
        result = lvb._divergence_significance(diff)

    assert result["psr_status"] == "formula_degenerate"
    assert result["psr_vs_zero"] is None
    assert result["diverged_significantly"] is False
    # Must not crash formatting/logging with a None value present
    _ = f"{result}"


def test_divergence_real_probability_not_confused_with_sentinels():
    from ascent.monitoring import live_vs_backtest as lvb

    with patch.object(lvb, "probabilistic_sharpe_ratio", return_value=0.7321):
        idx = _dates(30)
        diff = pd.Series(np.random.default_rng(4).normal(0.001, 0.0002, 30), index=idx)
        result = lvb._divergence_significance(diff)

    assert result["psr_status"] == "computed"
    assert result["psr_vs_zero"] == 0.7321
    assert result["diverged_significantly"] is False  # below threshold


# ── build_comparison integration ─────────────────────────────────────────────

def test_build_comparison_includes_significance_block():
    idx = _dates(40)
    bt = pd.Series(np.random.default_rng(5).normal(0.0005, 0.001, 40), index=idx)
    live = bt + np.random.default_rng(6).normal(0.0, 0.0003, 40)
    live = pd.Series(live, index=idx)

    payload = build_comparison(bt, live)

    assert "summary" in payload
    assert "significance" in payload
    sig = payload["significance"]
    assert "psr_status" in sig
    assert "diverged_significantly" in sig
    assert "cost_model_baseline_bps" in sig


# ── Issue 1: incremental-fetch cache ─────────────────────────────────────────

def test_second_call_within_ttl_skips_network():
    fake_history = {
        "2026-08-10": 0.004,
        "2026-08-11": -0.002,
        "2026-08-12": 0.001,
    }
    with patch(
        "ascent.execution.alpaca_broker.get_portfolio_history",
        return_value=fake_history,
    ) as mock_fetch:
        first = load_live_portfolio_values()
        second = load_live_portfolio_values()

    assert mock_fetch.call_count == 1  # second call served from cache, no network
    assert len(first) == 3
    assert list(second.round(6)) == list(first.round(6))


def test_call_after_ttl_expiry_uses_small_overlap_window_and_merges():
    first_history = {"2026-08-10": 0.004, "2026-08-11": -0.002}
    second_history = {"2026-08-11": -0.003, "2026-08-12": 0.001}  # overlap day corrected + 1 new day

    with patch(
        "ascent.execution.alpaca_broker.get_portfolio_history",
        return_value=first_history,
    ) as mock_fetch:
        first = load_live_portfolio_values()
    assert len(first) == 2

    # Force the cache to be treated as stale without waiting a real hour.
    with patch.object(lvb, "_cache_is_fresh", return_value=False):
        with patch(
            "ascent.execution.alpaca_broker.get_portfolio_history",
            return_value=second_history,
        ) as mock_fetch2:
            second = load_live_portfolio_values()

        # Only a small trailing window was requested, not a full year.
        _, kwargs = mock_fetch2.call_args
        assert kwargs["period"] == f"{lvb.LIVE_RETURNS_FETCH_OVERLAP_DAYS}D"

    # Merge: 3 distinct dates, with the overlapping day (08-11) taking the
    # freshly-fetched value, not the stale cached one.
    assert len(second) == 3
    assert round(float(second.loc[pd.Timestamp("2026-08-11")]), 6) == -0.003
    assert round(float(second.loc[pd.Timestamp("2026-08-12")]), 6) == 0.001


def test_first_call_with_empty_cache_fetches_full_year():
    with patch(
        "ascent.execution.alpaca_broker.get_portfolio_history",
        return_value={"2026-08-10": 0.001},
    ) as mock_fetch:
        load_live_portfolio_values()
    _, kwargs = mock_fetch.call_args
    assert kwargs["period"] == "1A"


# ── Issue 2: empty-ledger skips the network call entirely ───────────────────

def test_export_skips_live_fetch_when_backtest_ledger_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(lvb, "LEDGER_PATH", tmp_path / "no_such_ledger.csv")
    monkeypatch.setattr(lvb, "OUTPUT_PATH", tmp_path / "live_vs_backtest.json")

    with patch.object(lvb, "load_live_portfolio_values") as mock_live:
        result = export_live_vs_backtest()

    mock_live.assert_not_called()
    assert result == {}


# ── Issue 3: atomic write ────────────────────────────────────────────────────

def test_export_write_is_atomic_on_failure(tmp_path, monkeypatch):
    ledger = tmp_path / "ascent_daily_ledger.csv"
    ledger.write_text(
        "date,daily_return\n2026-01-01,0.001\n2026-01-02,-0.0005\n2026-01-03,0.0007\n"
        "2026-01-04,0.0002\n2026-01-05,0.0001\n"
    )
    output_path = tmp_path / "live_vs_backtest.json"
    output_path.write_text('{"existing": "good"}')  # pre-existing good file

    monkeypatch.setattr(lvb, "LEDGER_PATH", ledger)
    monkeypatch.setattr(lvb, "OUTPUT_PATH", output_path)

    fake_live = pd.Series(
        [0.001, -0.0004, 0.0006, 0.0002, 0.0001],
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )

    with patch.object(lvb, "load_live_portfolio_values", return_value=fake_live):
        with patch("json.dump", side_effect=ValueError("boom mid-write")):
            with pytest.raises(ValueError):
                export_live_vs_backtest()

    # The original file must survive intact -- no truncated/corrupt file,
    # and no stray .tmp file left behind.
    assert output_path.read_text() == '{"existing": "good"}'
    assert list(tmp_path.glob("*.tmp")) == []
