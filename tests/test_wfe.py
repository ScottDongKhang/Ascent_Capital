"""Unit tests for the Walk-Forward Efficiency (WFE) code path added to
ascent/research/walk_forward_runner.py: per-fold in-sample Sharpe
(_in_sample_fold_sharpe) and the WFE aggregation (_compute_wfe).

Deliberately does NOT run the full walk_forward_pipeline() -- that requires
the production price cache and universe machinery and is slow. Instead this
exercises the two new pure-ish functions directly with small synthetic data,
which is enough to prove the code path is correct without a multi-year walk.
"""
import json
import numpy as np
import pandas as pd
import pytest

from ascent.research.walk_forward_runner import (
    _in_sample_fold_sharpe,
    _compute_wfe,
    _WFE_SHARPE_CAP,
)
from ascent.research.evaluation import sharpe_ratio
from ascent.portfolio.optimizer import sector_constrained_weighted


def _synthetic_close(n_days=80, symbols=("AAA", "BBB", "CCC", "DDD", "EEE"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    rets = rng.normal(0.0006, 0.01, size=(n_days, len(symbols)))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list(symbols))
    return close


def _synthetic_alpha(close, seed=1):
    """A simple alpha panel aligned to close's index/columns -- values don't
    need to be causal here since this only tests the WFE plumbing, not the
    feature/alpha pipeline itself (that's exercised elsewhere)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, size=close.shape), index=close.index, columns=close.columns
    )


class TestInSampleFoldSharpe:
    def test_returns_finite_sharpe_with_enough_rebal_dates(self):
        close = _synthetic_close(n_days=80)
        alpha = _synthetic_alpha(close)
        train_start, train_end = close.index[0], close.index[60]
        rebal_dates_set = set(close.index[::5])  # every 5th day, like live cadence

        sh = _in_sample_fold_sharpe(
            hist_alpha=alpha,
            tradeable_symbols=list(close.columns),
            close_full=close,
            train_start=train_start,
            train_end=train_end,
            rebal_dates_set=rebal_dates_set,
            top_n=3,
            max_weight=0.5,
            max_per_sector=5,
            sector_map={},
        )
        assert np.isfinite(sh), f"expected a finite IS Sharpe, got {sh}"

    def test_nan_when_no_rebal_dates_in_window(self):
        close = _synthetic_close(n_days=40)
        alpha = _synthetic_alpha(close)
        train_start, train_end = close.index[0], close.index[10]
        # rebal dates entirely outside the training window
        rebal_dates_set = {close.index[30], close.index[35]}

        sh = _in_sample_fold_sharpe(
            hist_alpha=alpha,
            tradeable_symbols=list(close.columns),
            close_full=close,
            train_start=train_start,
            train_end=train_end,
            rebal_dates_set=rebal_dates_set,
            top_n=3,
            max_weight=0.5,
            max_per_sector=5,
            sector_map={},
        )
        assert np.isnan(sh)

    def test_nan_when_no_tradeable_symbols_in_alpha(self):
        close = _synthetic_close(n_days=40)
        alpha = _synthetic_alpha(close)
        train_start, train_end = close.index[0], close.index[20]
        rebal_dates_set = set(close.index[::5])

        sh = _in_sample_fold_sharpe(
            hist_alpha=alpha,
            tradeable_symbols=["NOT_A_REAL_SYMBOL"],
            close_full=close,
            train_start=train_start,
            train_end=train_end,
            rebal_dates_set=rebal_dates_set,
            top_n=3,
            max_weight=0.5,
            max_per_sector=5,
            sector_map={},
        )
        assert np.isnan(sh)

    def test_does_not_mutate_inputs(self):
        """The in-sample replay must not mutate hist_alpha or close_full --
        those are the same objects the OOS side of the fold still uses."""
        close = _synthetic_close(n_days=60)
        alpha = _synthetic_alpha(close)
        alpha_copy = alpha.copy()
        close_copy = close.copy()
        train_start, train_end = close.index[0], close.index[40]
        rebal_dates_set = set(close.index[::5])

        _in_sample_fold_sharpe(
            hist_alpha=alpha,
            tradeable_symbols=list(close.columns),
            close_full=close,
            train_start=train_start,
            train_end=train_end,
            rebal_dates_set=rebal_dates_set,
            top_n=3,
            max_weight=0.5,
            max_per_sector=5,
            sector_map={},
        )
        pd.testing.assert_frame_equal(alpha, alpha_copy)
        pd.testing.assert_frame_equal(close, close_copy)

    def test_missing_price_excluded_not_treated_as_zero_return(self):
        """Bug 3 regression: a NaN forward return (halt / late listing / data
        gap) must be EXCLUDED from the in-sample Sharpe calc, not treated as a
        real 0.0 return.

        We corrupt one symbol's prices to NaN over a chunk of the training
        window (a data gap) and compare the function's real output against a
        hand-built reconstruction of the OLD buggy behavior (fillna(0) applied
        directly to returns, so the gap counts as a real flat return whenever
        that symbol has nonzero weight). The fixed function must NOT match
        that buggy reconstruction whenever the gap actually carries exposure.
        """
        close = _synthetic_close(n_days=80, symbols=("AAA", "BBB", "CCC"))
        alpha = _synthetic_alpha(close)
        train_start, train_end = close.index[0], close.index[60]
        rebal_dates_set = set(close.index[::5])

        close_gap = close.copy()
        gap_slice = close_gap.index[20:30]
        close_gap.loc[gap_slice, "AAA"] = np.nan

        sh_gap = _in_sample_fold_sharpe(
            hist_alpha=alpha,
            tradeable_symbols=list(close.columns),
            close_full=close_gap,
            train_start=train_start,
            train_end=train_end,
            rebal_dates_set=rebal_dates_set,
            top_n=3,
            max_weight=0.5,
            max_per_sector=5,
            sector_map={},
        )
        assert np.isfinite(sh_gap)

        # Replicate the weight-generation logic used inside
        # _in_sample_fold_sharpe so we can build the buggy-reconstruction
        # comparison independently.
        is_cols = list(close.columns)
        is_dates = [d for d in alpha.index if train_start <= d <= train_end and d in rebal_dates_set]
        weight_rows = []
        for d in is_dates:
            d_weights = sector_constrained_weighted(
                alpha.loc[[d], is_cols], n=3, max_weight=0.5, max_per_sector=5,
                sector_map={}, regime_signal=None,
            )
            weight_rows.append(d_weights.reindex(columns=close.columns, fill_value=0.0))
        is_weights = pd.concat(weight_rows).sort_index()
        is_weights = is_weights[~is_weights.index.duplicated(keep="first")]
        is_days = close_gap.index[
            (close_gap.index >= is_weights.index[0]) & (close_gap.index <= train_end)
        ]
        is_weights_ff = is_weights.reindex(is_days).ffill().fillna(0.0)
        fwd_ret = close_gap.loc[is_days].pct_change().shift(-1).reindex(columns=is_weights_ff.columns)

        # Only a meaningful check if AAA actually carries nonzero weight
        # during the gap -- otherwise both strategies agree trivially.
        gap_has_exposure = (is_weights_ff.reindex(gap_slice)["AAA"].abs() > 0).any()
        assert gap_has_exposure, "test setup: AAA must carry weight during the gap to be a real check"

        # Buggy reconstruction: fillna(0) applied to RETURNS directly.
        buggy_rets = (is_weights_ff * fwd_ret.fillna(0.0)).sum(axis=1).iloc[:-1]
        buggy_sharpe = float(sharpe_ratio(buggy_rets, periods_per_year=252))

        assert sh_gap != pytest.approx(buggy_sharpe), (
            "in-sample Sharpe matches the fillna(0)-on-returns reconstruction; "
            "missing data is being treated as a real zero return instead of excluded"
        )


class TestComputeWFE:
    def test_normal_case(self):
        # OOS 0.8, mean IS 1.0 -> WFE 0.8
        wfe = _compute_wfe(oos_sharpe=0.8, fold_is_sharpes=[1.0, 1.0, 1.0])
        assert wfe == pytest.approx(0.8)

    def test_no_valid_is_sharpes_returns_none(self):
        assert _compute_wfe(oos_sharpe=1.0, fold_is_sharpes=[]) is None
        assert _compute_wfe(oos_sharpe=1.0, fold_is_sharpes=[float("nan"), float("nan")]) is None

    def test_all_negative_is_sharpes_returns_none(self):
        # Dividing by a non-positive in-sample Sharpe is meaningless -- must
        # not silently produce a sign-flipped or blown-up ratio.
        assert _compute_wfe(oos_sharpe=0.5, fold_is_sharpes=[-1.0, -0.5]) is None

    def test_non_finite_oos_returns_none(self):
        assert _compute_wfe(oos_sharpe=float("inf"), fold_is_sharpes=[1.0]) is None
        assert _compute_wfe(oos_sharpe=float("nan"), fold_is_sharpes=[1.0]) is None
        assert _compute_wfe(oos_sharpe=None, fold_is_sharpes=[1.0]) is None

    def test_extreme_oos_sharpe_is_capped(self):
        # OOS Sharpe of 100 with mean IS Sharpe of 1.0 would naively give WFE
        # of 100 -- must be capped at _WFE_SHARPE_CAP before dividing.
        wfe = _compute_wfe(oos_sharpe=100.0, fold_is_sharpes=[1.0])
        assert wfe == pytest.approx(_WFE_SHARPE_CAP)

        wfe_neg = _compute_wfe(oos_sharpe=-100.0, fold_is_sharpes=[1.0])
        assert wfe_neg == pytest.approx(-_WFE_SHARPE_CAP)

    def test_mixed_valid_and_invalid_is_sharpes_uses_only_valid(self):
        # nan and negative folds should be excluded from the mean, not zero
        # them out or crash.
        wfe = _compute_wfe(
            oos_sharpe=1.0,
            fold_is_sharpes=[2.0, float("nan"), -1.0, 2.0],
        )
        assert wfe == pytest.approx(0.5)  # 1.0 / mean([2.0, 2.0])


class TestAlphaOverridesProvenance:
    """Bug 2 regression: the wf_report JSON's _meta.alpha_overrides used to be
    a hardcoded literal {"meanrev": 0.5, "statarb": 0.5} regardless of what
    weights the run actually used. walk_forward_pipeline() now resolves the
    weights once via the same _load_active_alpha_weights()/_get_gated_weights()
    path build_alpha_stack() uses internally, threads that explicit dict into
    both build_alpha_stack() call sites in the fold loop, and writes the same
    dict into the report -- so the report can't silently diverge from the
    weights that were actually used.

    Running the full pipeline is deliberately avoided (see module docstring),
    so this is a source-level regression test (proving the literal is gone and
    the report is sourced from the resolved-weights variable) plus a check
    that the resolution utilities involved actually pick up a config override
    -- proving 'resolved_alpha_weights' would visibly differ from the old
    hardcoded default whenever the active config does.
    """

    def test_hardcoded_literal_is_gone(self):
        import inspect
        from ascent.research import walk_forward_runner
        src = inspect.getsource(walk_forward_runner)
        assert '"alpha_overrides": {"meanrev": 0.5, "statarb": 0.5}' not in src, (
            "alpha_overrides must not be a hardcoded literal -- it must reflect the "
            "weights actually resolved and used for the run"
        )

    def test_alpha_overrides_sourced_from_resolved_weights_variable(self):
        import inspect
        from ascent.research import walk_forward_runner
        src = inspect.getsource(walk_forward_runner)
        assert '"alpha_overrides": resolved_alpha_weights' in src, (
            "alpha_overrides must be written from a resolved_alpha_weights variable "
            "captured once at the top of walk_forward_pipeline()"
        )
        # Both build_alpha_stack() call sites in the fold loop must pass the same
        # resolved weights explicitly, rather than leaving alpha_weights=None and
        # letting build_alpha_stack() re-resolve (and potentially diverge) per fold.
        assert src.count("alpha_weights=resolved_alpha_weights") == 2, (
            "expected both build_alpha_stack() call sites in the fold loop to pass "
            "alpha_weights=resolved_alpha_weights explicitly"
        )

    def test_resolution_reflects_active_config_override(self, tmp_path, monkeypatch):
        """The resolution walk_forward_runner.py performs
        (_load_active_alpha_weights + _get_gated_weights) must pick up a
        non-default active_alpha_config.json, proving the captured
        resolved_alpha_weights would differ from the old
        {"meanrev": 0.5, "statarb": 0.5} literal whenever the config
        overrides it -- exactly the drift the hardcoded literal was silently
        hiding."""
        monkeypatch.chdir(tmp_path)
        custom_weights = {"meanrev": 0.30, "statarb": 0.70}
        config_path = tmp_path / "data_cache" / "active_alpha_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"global": custom_weights}))

        from ascent.alpha.stack import _load_active_alpha_weights, _get_gated_weights
        resolved = _get_gated_weights(_load_active_alpha_weights())

        assert resolved == custom_weights
        assert resolved != {"meanrev": 0.5, "statarb": 0.5}
