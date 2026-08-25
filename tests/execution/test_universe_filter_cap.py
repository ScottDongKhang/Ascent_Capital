"""
Tests for ascent/execution/eod_runner.py's filter_to_tradeable_universe().

Regression test for a bug found by independent code review: the universe
filter (build_historical_universe(strict=True, sp500_only=True)) drops
symbols that fell out of the tradeable universe, then renormalizes the
survivors to sum to 1.0 -- but without a re-cap step, that renormalization
can push a remaining position's weight past the optimizer's max_weight cap
(integrity constraint #3, enforced via `_water_fill_cap()` in
ascent/portfolio/optimizer.py "with a post-condition check"). Example: a
name sitting comfortably under the cap absorbs most of a dropped name's
weight once the book renormalizes, and comes out the other side above the
cap.

Note: `_water_fill_cap()`'s "cap too tight" branch (max_weight * n_survivors
< 1.0) falls back to equal weight rather than enforcing the cap -- tests
below use enough survivors (>=10 at a 10% cap) to stay in the real capping
branch, matching the CLAUDE.md example (~15.4%) rather than the
infeasible-cap fallback.
"""
import pandas as pd
import pytest

from ascent.execution.eod_runner import filter_to_tradeable_universe
from ascent.portfolio.optimizer import _water_fill_cap


MAX_WEIGHT = 0.10


def _survivors_and_dropped():
    """
    11 survivors + 1 dropped name, weights summing to 1.0. BIG (0.09, under
    the 10% cap pre-filter) absorbs most of dropped ZZZ's 0.41 once the
    filtered book renormalizes, pushing it over the cap. 11 survivors keeps
    11 * 0.10 = 1.1 >= 1.0, so the real water-filling branch engages instead
    of the infeasible-cap equal-weight fallback.
    """
    small = {f"S{i}": 0.05 for i in range(10)}  # sum 0.50
    weights = {**small, "BIG": 0.09, "ZZZ": 0.41}  # total 1.00
    tradeable = set(small) | {"BIG"}
    return pd.Series(weights), tradeable


def test_universe_filter_drop_pushes_survivor_over_cap_without_recap():
    """
    Sanity check that the bug is real: naive filter + renormalize (no re-cap)
    breaches max_weight. This mirrors the pre-fix behavior removed from
    eod_runner.run_eod().
    """
    target_weights, tradeable = _survivors_and_dropped()

    naive = target_weights[target_weights.index.isin(tradeable)]
    naive = naive[naive > 0].dropna()
    naive = naive / naive.sum()

    # BIG (0.09 pre-filter) renormalizes to 0.09 / 0.59 ~= 0.1525 -- over cap.
    assert naive["BIG"] > MAX_WEIGHT
    assert naive.max() > MAX_WEIGHT


def test_filter_to_tradeable_universe_respects_max_weight_cap():
    """
    The fixed helper must not let any surviving weight exceed max_weight,
    even though naive renormalization alone would breach it.
    """
    target_weights, tradeable = _survivors_and_dropped()

    result = filter_to_tradeable_universe(target_weights, tradeable, MAX_WEIGHT)

    assert set(result.index) == tradeable
    assert "ZZZ" not in result.index
    assert result.max() <= MAX_WEIGHT + 1e-9
    # _water_fill_cap's convention: always renormalizes fully to 1.0 (no
    # residual cash held back) when the cap is feasible for the survivor
    # count -- confirmed by reading its post-condition docstring
    # ("Post-condition: all weights <= max_weight + 1e-9, sum == 1.0").
    assert result.sum() == pytest.approx(1.0)


def test_filter_to_tradeable_universe_matches_water_fill_cap_directly():
    """
    The fix must reuse _water_fill_cap() itself (per integrity constraint #3),
    not a bespoke capping algorithm -- confirm the helper's output is exactly
    what calling _water_fill_cap() on the renormalized, filtered series would
    produce.
    """
    target_weights, tradeable = _survivors_and_dropped()

    result = filter_to_tradeable_universe(target_weights, tradeable, MAX_WEIGHT)

    filtered = target_weights[target_weights.index.isin(tradeable)]
    filtered = filtered[filtered > 0].dropna()
    renormalized = filtered / filtered.sum()
    expected = _water_fill_cap(renormalized, MAX_WEIGHT)

    pd.testing.assert_series_equal(
        result.sort_index(), expected.sort_index(), check_names=False
    )


def test_filter_to_tradeable_universe_noop_when_no_symbols_dropped():
    """
    When the universe filter drops nothing and no weight is over cap, weights
    should pass through unchanged.
    """
    target_weights = pd.Series({f"S{i}": 0.10 for i in range(10)})  # sum 1.0
    tradeable = set(target_weights.index)

    result = filter_to_tradeable_universe(target_weights, tradeable, MAX_WEIGHT)

    pd.testing.assert_series_equal(
        result.sort_index(), target_weights.sort_index(), check_names=False
    )


def test_filter_to_tradeable_universe_drops_zero_and_non_tradeable_symbols():
    target_weights = pd.Series(
        {f"S{i}": 0.095 for i in range(10)} | {"DROPPED": 0.05, "ZERO": 0.0}
    )
    tradeable = {f"S{i}" for i in range(10)} | {"ZERO"}

    result = filter_to_tradeable_universe(target_weights, tradeable, MAX_WEIGHT)

    assert "DROPPED" not in result.index
    assert "ZERO" not in result.index
    assert result.max() <= MAX_WEIGHT + 1e-9
    assert result.sum() == pytest.approx(1.0)


def test_filter_to_tradeable_universe_empty_result_when_nothing_tradeable():
    target_weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    tradeable: set = set()

    result = filter_to_tradeable_universe(target_weights, tradeable, MAX_WEIGHT)

    assert result.empty
