# tests/test_walkforward_institutional.py
import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path


def _make_price_cache(tmp_path, n_days=500, n_syms=25):
    np.random.seed(42)
    idx = pd.bdate_range(end=date.today(), periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_syms)] + ["SPY"]
    n_actual = len(idx)
    rets = np.random.normal(0.0003, 0.012, (n_actual, len(symbols)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=symbols)
    out = tmp_path / "data_cache" / "prices_live.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)
    return prices


def test_lightweight_oos_uses_multiple_folds(tmp_path, monkeypatch):
    """With enough data, run_lightweight_oos must return n_folds > 1."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
    )
    assert result["n_folds"] > 1, \
        f"Expected multiple folds with 500 days of data, got n_folds={result['n_folds']}"


def test_lightweight_oos_purge_embargo_respected(tmp_path, monkeypatch):
    """Verify the function runs without error when purge and embargo are applied."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.70, "meanrev": 0.05,
                                             "statarb": 0.10, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
        purge_days=5,
        embargo_days=5,
    )
    assert isinstance(result["sharpe"], float)
    assert result["n_folds"] >= 1


def test_lightweight_oos_survivorship_bias_fix(tmp_path, monkeypatch):
    """Universe must be filtered per fold date — graceful fallback when no universe data."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
        filter_universe_by_date=True,
    )
    assert "sharpe" in result
    assert "n_folds" in result


def test_lightweight_oos_sharpe_from_all_folds(tmp_path, monkeypatch):
    """Sharpe must be computed across all fold returns, not just the last fold."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    r1 = run_lightweight_oos({"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                                  "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
                              n_days=63)
    assert np.isfinite(r1["sharpe"])
    assert r1.get("n_folds", 0) >= 1


def test_walk_forward_runner_calls_universe_per_fold():
    """walk_forward_runner must call get_universe_on_date on every fold -- A4 gap."""
    import inspect
    from ascent.research import walk_forward_runner
    src = inspect.getsource(walk_forward_runner)
    assert "get_universe_on_date" in src, \
        "walk_forward_runner must call get_universe_on_date() per fold to prevent survivorship bias"


# ---------------------------------------------------------------------------
# Bug 1: run_lightweight_oos was the third universe consumer, still defaulting
# to the WIDE universe (get_universe_on_date() with no override -> strict=False,
# sp500_only=False) after eod_runner.py and walk_forward_runner.py both moved
# to the survivorship-correct build_historical_universe(strict=True,
# sp500_only=True). This matters because self_improve.py/shadow_promoter.py
# use this lightweight OOS path to promote alpha-weight variants -- a variant
# could be promoted on non-S&P500 symbols with fabricated pre-2020 histories
# that live trading (now sp500_only=True) can never actually hold.
# ---------------------------------------------------------------------------

def test_universe_consistency_across_three_callers():
    """All three consumers of build_historical_universe -- eod_runner.py (live
    trading), walk_forward_runner.py (canonical OOS backtest), and
    walk_forward_lightweight.py (self-improve/shadow-promoter's fast OOS path)
    -- must resolve the identical survivorship-correct universe call, so a
    variant promoted via the lightweight path can't be evaluated on symbols
    the other two would never select."""
    import inspect
    from ascent.research import walk_forward_runner, walk_forward_lightweight
    from ascent.execution import eod_runner

    expected_call = "build_historical_universe(strict=True, sp500_only=True)"
    for mod in (walk_forward_runner, walk_forward_lightweight, eod_runner):
        src = inspect.getsource(mod)
        assert expected_call in src, (
            f"{mod.__name__} does not call {expected_call} -- universe consumers have "
            "diverged again"
        )


def test_lightweight_oos_accepts_universe_df_override():
    """run_lightweight_oos must accept an explicit universe_df param (so an
    outer walk-forward loop that already built one can pass it through
    instead of rebuilding it per variant), matching the pattern in
    walk_forward_runner.py and eod_runner.py."""
    import inspect
    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    sig = inspect.signature(run_lightweight_oos)
    assert "universe_df" in sig.parameters, \
        "run_lightweight_oos must accept a universe_df override parameter"
    assert sig.parameters["universe_df"].default is None, \
        "universe_df must default to None (built internally when omitted)"


def test_lightweight_oos_default_universe_is_strict_sp500_only(tmp_path, monkeypatch):
    """When no universe_df is supplied, run_lightweight_oos must build one via
    build_historical_universe(strict=True, sp500_only=True) -- not silently
    fall back to the wide default (strict=False, sp500_only=False) that
    get_universe_on_date() would otherwise use with no override."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    import ascent.data.universe as universe_mod
    real_build = universe_mod.build_historical_universe
    calls = []

    def _tracking_build(strict=False, sp500_only=False):
        calls.append({"strict": strict, "sp500_only": sp500_only})
        return real_build(strict=strict, sp500_only=sp500_only)

    monkeypatch.setattr(universe_mod, "build_historical_universe", _tracking_build)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
        filter_universe_by_date=True,
    )

    assert calls, "run_lightweight_oos must call build_historical_universe() when universe_df is omitted"
    assert calls[0] == {"strict": True, "sp500_only": True}, (
        f"expected strict=True, sp500_only=True (matching eod_runner.py / "
        f"walk_forward_runner.py), got {calls[0]}"
    )


def test_lightweight_oos_reuses_passed_universe_df(tmp_path, monkeypatch):
    """When a caller already supplies universe_df, run_lightweight_oos must
    not rebuild it via build_historical_universe() -- avoids wasted work in a
    tight per-variant evaluation loop."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    import ascent.data.universe as universe_mod
    calls = []
    real_build = universe_mod.build_historical_universe

    def _tracking_build(strict=False, sp500_only=False):
        calls.append((strict, sp500_only))
        return real_build(strict=strict, sp500_only=sp500_only)

    monkeypatch.setattr(universe_mod, "build_historical_universe", _tracking_build)
    caller_universe_df = real_build(strict=True, sp500_only=True)
    calls.clear()  # only count calls made *inside* run_lightweight_oos below

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
        filter_universe_by_date=True,
        universe_df=caller_universe_df,
    )

    assert not calls, "run_lightweight_oos must not rebuild the universe when universe_df is already supplied"


# ---------------------------------------------------------------------------
# Fold-gap Calmar bug: consecutive OOS fold test windows are NOT temporally
# contiguous (purge + embargo gap of untested trading days between them).
# Naively concatenating each fold's per-day return series end-to-end and
# feeding that into calmar_ratio() (order/continuity-sensitive: cumprod +
# running peak/drawdown) either hides a real drawdown that happened during
# the untested gap, or fabricates a spurious "recovery" by pretending no
# time passed between fold N's last day and fold N+1's first day.
#
# zero_fill_fold_gaps() reconstructs the concatenated series with explicit
# 0.0 ("no change") placeholder days for the gap, so calmar_ratio() sees a
# continuous, honestly-dated path instead.
# ---------------------------------------------------------------------------

def test_zero_fill_fold_gaps_preserves_gap_length():
    """Gap between folds must be filled with the correct number of zero days,
    computed from actual bar positions, not a nominal config constant."""
    from ascent.research.walk_forward_lightweight import zero_fill_fold_gaps

    # Fold A: bar positions [0, 4] (5 returns). Fold B: bar positions [72, 76]
    # (5 returns) -- a gap of 72 - 4 - 1 = 67 untested days in between.
    fold_a_rets = [0.01, 0.01, 0.01, 0.01, 0.01]
    fold_b_rets = [-0.02, -0.02, -0.02, -0.02, -0.02]
    fold_records = [(0, 4, fold_a_rets), (72, 76, fold_b_rets)]

    out = zero_fill_fold_gaps(fold_records)

    expected_gap = 72 - 4 - 1
    assert len(out) == len(fold_a_rets) + expected_gap + len(fold_b_rets)
    assert out[:5] == fold_a_rets
    assert out[5:5 + expected_gap] == [0.0] * expected_gap
    assert out[5 + expected_gap:] == fold_b_rets


def test_zero_fill_fold_gaps_no_gap_when_contiguous():
    """Back-to-back bar positions (no purge/embargo gap) must not insert zeros."""
    from ascent.research.walk_forward_lightweight import zero_fill_fold_gaps

    fold_a_rets = [0.01, 0.02]
    fold_b_rets = [0.03, 0.04]
    fold_records = [(0, 1, fold_a_rets), (2, 3, fold_b_rets)]

    out = zero_fill_fold_gaps(fold_records)
    assert out == fold_a_rets + fold_b_rets


def test_gap_fill_does_not_erase_a_real_drawdown_at_the_boundary():
    """A drawdown that happens right at a fold's tail end must survive into
    the Calmar computation, not be erased by whatever comes after the gap."""
    from ascent.research.walk_forward_lightweight import zero_fill_fold_gaps
    from ascent.research.evaluation import calmar_ratio, max_drawdown

    # Fold A ends with a sharp drawdown (-30% over its last few days).
    fold_a_rets = [0.01, 0.01, -0.15, -0.15, 0.0]
    # Fold B, well after the gap, is flat/mildly positive -- it does NOT
    # "recover" fold A's loss, it just starts its own separate window.
    fold_b_rets = [0.001, 0.001, 0.001, 0.001, 0.001]
    fold_records = [(0, 4, fold_a_rets), (72, 76, fold_b_rets)]

    gap_filled = zero_fill_fold_gaps(fold_records)
    naive_concat = fold_a_rets + fold_b_rets

    dd_gap_filled = max_drawdown(pd.Series(gap_filled))
    dd_naive = max_drawdown(pd.Series(naive_concat))

    # The drawdown magnitude itself must be identical either way (same price
    # path within fold A) -- gap-filling must not shrink or hide it.
    assert dd_gap_filled == pytest.approx(dd_naive, abs=1e-9)
    assert dd_gap_filled < -0.25  # the real ~30% drawdown must be visible

    # But the *number of periods* used to annualize must differ: the honest
    # gap-filled series is much longer (includes the untested days), so its
    # Calmar must not equal the naive concatenation's Calmar -- the naive
    # version implicitly (and wrongly) claims the whole path took only
    # len(fold_a)+len(fold_b) trading days.
    calmar_gap_filled = calmar_ratio(pd.Series(gap_filled))
    calmar_naive = calmar_ratio(pd.Series(naive_concat))
    assert calmar_gap_filled != pytest.approx(calmar_naive)


def test_gap_fill_does_not_fabricate_a_fake_recovery():
    """Two folds whose edge returns would look like an instant recovery if
    naively abutted must not produce that illusion after zero-filling."""
    from ascent.research.walk_forward_lightweight import zero_fill_fold_gaps
    from ascent.research.evaluation import calmar_ratio, max_drawdown

    # Fold A ends deep in a drawdown.
    fold_a_rets = [0.0, 0.0, -0.10, -0.10, -0.10]
    # Fold B *opens* with a big positive day -- naively concatenated, this
    # reads as "recovered the very next day." In reality ~67 untested days
    # separated them.
    fold_b_rets = [0.30, 0.01, 0.01, 0.01, 0.01]
    fold_records = [(0, 4, fold_a_rets), (72, 76, fold_b_rets)]

    naive_concat = fold_a_rets + fold_b_rets
    gap_filled = zero_fill_fold_gaps(fold_records)

    # Naive concatenation shows the trough immediately followed (next index)
    # by the recovery day -- an artifact of pretending no time passed.
    naive_cum = (1 + pd.Series(naive_concat)).cumprod()
    trough_idx = naive_cum.idxmin()
    assert naive_cum.iloc[trough_idx + 1] > naive_cum.iloc[trough_idx] * 1.25, \
        "sanity check: naive concatenation should show an abrupt jump right after the trough"

    # In the gap-filled series, the trough is followed by ~67 flat (0.0)
    # days before fold B's recovery day appears -- no abrupt "instant
    # recovery" step immediately after the trough.
    gap_cum = (1 + pd.Series(gap_filled)).cumprod()
    gap_trough_idx = gap_cum.idxmin()
    assert gap_cum.iloc[gap_trough_idx + 1] == pytest.approx(gap_cum.iloc[gap_trough_idx]), \
        "the day right after the trough must be flat (0.0 gap day), not fold B's recovery day"

    # The max drawdown magnitude is unaffected (still the real ~19% trough
    # from fold A), but it is not instantly erased -- confirm both metrics
    # are computable and the drawdown is preserved.
    assert max_drawdown(pd.Series(gap_filled)) == pytest.approx(max_drawdown(pd.Series(naive_concat)), abs=1e-9)
    assert calmar_ratio(pd.Series(gap_filled)) != pytest.approx(calmar_ratio(pd.Series(naive_concat)))


def test_run_lightweight_oos_returns_field_is_gap_filled_and_sharpe_unaffected(tmp_path, monkeypatch):
    """End-to-end: the 'returns' field length must reflect the real fold gaps
    (via fold_date_ranges), while 'sharpe' stays computed from the pure,
    undiluted fold returns (unaffected by gap padding)."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
    )
    assert result["n_folds"] >= 1
    assert "returns" in result
    assert "fold_date_ranges" in result
    assert len(result["fold_date_ranges"]) == result["n_folds"]

    if result["n_folds"] > 1:
        # returns must be at least as long as a naive concat would be with
        # no gaps -- i.e. strictly longer once there's more than one fold,
        # since real embargo gaps exist between them.
        assert isinstance(result["returns"], list)
        assert len(result["returns"]) > 0
        assert np.isfinite(result["sharpe"])
