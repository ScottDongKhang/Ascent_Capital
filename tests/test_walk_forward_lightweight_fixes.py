# tests/test_walk_forward_lightweight_fixes.py
"""
Regression tests for the 3 confirmed code-review bugs in
ascent/research/walk_forward_lightweight.py:

  BUG 1: the per-fold survivorship-bias universe filter never executed
         because get_universe_on_date() returns a plain list, not a
         DataFrame, so the old `.empty`/`.columns` DataFrame-shape branching
         raised AttributeError every call, silently swallowed by the
         surrounding bare `except Exception: pass`.

  BUG 2: run_lightweight_oos() never applied
         ascent.research.walk_forward_runner.apply_delisting_terminal_credit()
         before computing fold returns, unlike the canonical
         walk_forward_pipeline() in walk_forward_runner.py.

  BUG 3: fold_rets was computed via oos_px.pct_change().dropna(), which
         drops the fold's first real observation as a NaN -- one fewer
         observation than the fold's actual day-span (test_start_i..
         test_end_i), even though fold_records / fold_date_ranges still
         record the full un-shifted span.
"""
import pandas as pd
import numpy as np
from datetime import date


DEFAULT_WEIGHTS = {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05}


def _make_price_cache(tmp_path, monkeypatch, n_days=500, n_syms=25, symbols=None):
    """Build a synthetic price frame and force run_lightweight_oos() to see it.

    _load_prices() tries the real package-root data_cache FIRST (has_data()
    check) and only falls back to a cwd-relative file when the real store
    lacks that cache name. Once real data_cache/prices_live.parquet actually
    exists in the repo (e.g. after a production walk-forward run copies it
    into a worktree), that fallback path is never reached and these tests'
    file-based isolation silently breaks -- they'd start reading real
    production prices instead of the synthetic frame. Monkeypatching
    _load_prices() directly removes that ambient-environment dependency.
    """
    np.random.seed(42)
    idx = pd.bdate_range(end=date.today(), periods=n_days)
    if symbols is None:
        symbols = [f"SYM{i:02d}" for i in range(n_syms)] + ["SPY"]
    n_actual = len(idx)
    rets = np.random.normal(0.0003, 0.012, (n_actual, len(symbols)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=symbols)
    out = tmp_path / "data_cache" / "prices_live.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)

    import ascent.research.walk_forward_lightweight as wfl_mod
    monkeypatch.setattr(wfl_mod, "_load_prices", lambda prices_cache: prices.copy())

    return prices


# ---------------------------------------------------------------------------
# BUG 1: universe filter must actually exclude a symbol invalid on the
# fold's date, given a small synthetic universe_df.
# ---------------------------------------------------------------------------

def test_universe_filter_excludes_out_of_window_symbol(tmp_path, monkeypatch):
    """A symbol whose universe window closed long before any fold date must
    never reach build_all_features() -- proves get_universe_on_date()'s list
    return value is actually being used to filter, not silently ignored."""
    monkeypatch.chdir(tmp_path)
    symbols = [f"SYM{i:02d}" for i in range(9)] + ["EXCLUDED", "SPY"]
    _make_price_cache(tmp_path, monkeypatch, n_days=500, symbols=symbols)

    # EXCLUDED's universe window closed in 2000 -- invalid on every fold date
    # a 500-business-day-ending-today cache could produce.
    universe_df = pd.DataFrame(
        [
            {"symbol": s, "start_date": pd.Timestamp("2000-01-01"), "end_date": pd.Timestamp("2099-01-01")}
            for s in [f"SYM{i:02d}" for i in range(9)]
        ]
        + [{"symbol": "EXCLUDED", "start_date": pd.Timestamp("2000-01-01"), "end_date": pd.Timestamp("2000-01-02")}]
    )

    captured_cols = []
    import ascent.features.feature_defs as feature_defs_mod
    real_build_all_features = feature_defs_mod.build_all_features

    def _capturing_build_all_features(close, **kwargs):
        captured_cols.append(set(close.columns))
        return real_build_all_features(close=close, **kwargs)

    monkeypatch.setattr(feature_defs_mod, "build_all_features", _capturing_build_all_features)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    run_lightweight_oos(
        config_overrides={"alpha_weights": DEFAULT_WEIGHTS},
        n_days=63,
        universe_df=universe_df,
        filter_universe_by_date=True,
    )

    assert captured_cols, "build_all_features must have been called for at least one fold"
    for cols in captured_cols:
        assert "EXCLUDED" not in cols, (
            "universe filter must exclude a symbol invalid on the fold date -- "
            "BUG 1 regression: get_universe_on_date() list mishandled as a DataFrame"
        )
        assert "SPY" in cols, "SPY must always survive the universe filter"


def test_universe_filter_is_noop_when_all_symbols_valid(tmp_path, monkeypatch):
    """Sanity check: when every symbol is valid on the fold date, the filter
    must not accidentally drop anything (no false positives from the fix)."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, monkeypatch, n_days=500, n_syms=10)

    universe_df = pd.DataFrame(
        [
            {"symbol": f"SYM{i:02d}", "start_date": pd.Timestamp("2000-01-01"), "end_date": pd.Timestamp("2099-01-01")}
            for i in range(10)
        ]
    )

    captured_cols = []
    import ascent.features.feature_defs as feature_defs_mod
    real_build_all_features = feature_defs_mod.build_all_features

    def _capturing_build_all_features(close, **kwargs):
        captured_cols.append(set(close.columns))
        return real_build_all_features(close=close, **kwargs)

    monkeypatch.setattr(feature_defs_mod, "build_all_features", _capturing_build_all_features)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    run_lightweight_oos(
        config_overrides={"alpha_weights": DEFAULT_WEIGHTS},
        n_days=63,
        universe_df=universe_df,
        filter_universe_by_date=True,
    )

    assert captured_cols
    for cols in captured_cols:
        assert "SPY" in cols
        for i in range(10):
            assert f"SYM{i:02d}" in cols


# ---------------------------------------------------------------------------
# BUG 2: apply_delisting_terminal_credit() must actually be called and its
# return value actually used before fold returns are computed.
# ---------------------------------------------------------------------------

def test_delisting_credit_is_called(tmp_path, monkeypatch):
    """run_lightweight_oos must call apply_delisting_terminal_credit() on the
    loaded price frame -- code-path parity with walk_forward_runner.py's
    walk_forward_pipeline()."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, monkeypatch, n_days=500, n_syms=10)

    calls = []
    import ascent.research.walk_forward_runner as wfr_mod
    real_credit = wfr_mod.apply_delisting_terminal_credit

    def _spy_credit(close_full, open_full=None):
        calls.append(close_full.shape)
        return real_credit(close_full, open_full)

    monkeypatch.setattr(wfr_mod, "apply_delisting_terminal_credit", _spy_credit)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": DEFAULT_WEIGHTS},
        n_days=63,
    )

    assert calls, "apply_delisting_terminal_credit must be called by run_lightweight_oos"
    assert result["n_folds"] >= 1


def test_delisting_credit_return_value_is_actually_used(tmp_path, monkeypatch):
    """The DataFrame apply_delisting_terminal_credit() returns must be the
    one folds are actually built from -- not merely called and discarded."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, monkeypatch, n_days=500, n_syms=10)

    def _truncating_credit(close_full, open_full=None):
        # Return far fewer rows than min_required (well below the ~199-row
        # floor for the default train/purge/n_days/embargo settings) so
        # that if -- and only if -- this returned frame is what folds get
        # built from, run_lightweight_oos must report n_folds == 0.
        return close_full.iloc[:50], open_full

    import ascent.research.walk_forward_runner as wfr_mod
    monkeypatch.setattr(wfr_mod, "apply_delisting_terminal_credit", _truncating_credit)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": DEFAULT_WEIGHTS},
        n_days=63,
    )

    assert result["n_folds"] == 0, (
        "run_lightweight_oos must use the frame returned by "
        "apply_delisting_terminal_credit(), not the original unmodified price_wide"
    )


# ---------------------------------------------------------------------------
# BUG 3: fold_rets must have as many observations as the fold's actual
# day-span (test_end_i - test_start_i + 1), not one fewer.
# ---------------------------------------------------------------------------

def test_fold_rets_length_matches_actual_day_span(tmp_path, monkeypatch):
    """Force exactly one fold (no gap-fill padding involved) and confirm the
    'returns' field -- which equals that single fold's fold_rets verbatim --
    has exactly n_days observations, matching fold_date_ranges' span."""
    monkeypatch.chdir(tmp_path)
    n_days, train_days, purge_days, embargo_days = 63, 126, 5, 5
    # Exactly min_required rows forces exactly one fold: see the loop-position
    # derivation in walk_forward_lightweight.py's fold-building while-loop.
    total_rows = train_days + purge_days + n_days + embargo_days
    _make_price_cache(tmp_path, monkeypatch, n_days=total_rows, n_syms=25)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": DEFAULT_WEIGHTS},
        n_days=n_days,
        train_days=train_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
        filter_universe_by_date=False,
    )

    assert result["n_folds"] == 1, (
        f"test setup expects exactly one fold with {total_rows} rows, got n_folds={result['n_folds']}"
    )
    assert len(result["fold_date_ranges"]) == 1
    assert len(result["returns"]) == n_days, (
        f"fold_rets must have {n_days} observations (the fold's actual day span "
        f"test_start_i..test_end_i), got {len(result['returns'])} -- BUG 3 off-by-one "
        f"from oos_px.pct_change().dropna() discarding the fold's first real return"
    )


def test_fold_rets_first_day_return_is_a_real_return_not_dropped(tmp_path, monkeypatch):
    """Directly exercises the pct_change-with-prior-bar fix: constructs a
    price series where the fold's very first day has a large, known jump,
    and confirms that jump survives into 'returns' rather than being
    silently dropped as the pct_change() NaN row."""
    monkeypatch.chdir(tmp_path)
    n_days, train_days, purge_days, embargo_days = 10, 30, 2, 2
    total_rows = train_days + purge_days + n_days + embargo_days

    idx = pd.bdate_range(end=date.today(), periods=total_rows)
    symbols = [f"SYM{i:02d}" for i in range(5)] + ["SPY"]
    np.random.seed(7)
    rets = np.random.normal(0.0002, 0.005, (total_rows, len(symbols)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=symbols)

    # test_start_i for this config (single fold) = train_days + purge_days + embargo_days
    test_start_i = train_days + purge_days + embargo_days
    # Inject a large, known jump on the fold's first real day so it is easy
    # to detect whether it was dropped.
    prices.iloc[test_start_i] = prices.iloc[test_start_i - 1] * 1.20

    out = tmp_path / "data_cache" / "prices_live.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)

    import ascent.research.walk_forward_lightweight as wfl_mod
    monkeypatch.setattr(wfl_mod, "_load_prices", lambda prices_cache: prices.copy())

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.0, "meanrev": 1.0, "statarb": 0.0,
                                             "ml": 0.0, "volatility": 0.0}},
        n_days=n_days,
        train_days=train_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
        filter_universe_by_date=False,
    )

    assert result["n_folds"] == 1
    assert len(result["returns"]) == n_days, (
        "the fold's first-day jump must be included in the return series -- if BUG 3 "
        "regresses, this list is one element short and the jump can be silently dropped"
    )
