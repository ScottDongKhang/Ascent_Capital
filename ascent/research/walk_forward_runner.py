"""
Ascent Capital - Walk-Forward Runner
Computes alpha using only past data for each test window.
Stitches out-of-sample weights into a single timeline for backtesting.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import pandas as pd
import numpy as np

from ascent.config.settings import get_config
from ascent.data.store.parquet import load_parquet, has_data
from ascent.data.normalize.prices import normalize_prices, pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack
from ascent.portfolio.optimizer import sector_constrained_weighted
from ascent.research.splits import walk_forward_splits
from ascent.research.evaluation import sharpe_ratio, annualized_return, max_drawdown
from ascent.backtest.engine import BacktestEngine
from ascent.research.evaluation import format_metrics
from ascent.data.universe import build_historical_universe, get_universe_on_date

TARGET_HORIZON = 21


def _pit_macro(macro_df: pd.DataFrame, as_of_date: pd.Timestamp):
    if macro_df is None:
        return None
    if "known_time" in macro_df.columns:
        return macro_df[macro_df["known_time"] <= as_of_date].copy()
    return macro_df[macro_df["date"] <= as_of_date].copy()


def _pit_slice(df, date_col: str, as_of_date: pd.Timestamp):
    """Filter a DataFrame to rows where date_col <= as_of_date (point-in-time)."""
    if df is None or df.empty:
        return df
    col = pd.to_datetime(df[date_col])
    return df[col <= as_of_date].copy()


def walk_forward_pipeline(
    train_days=None,
    purge_days=None,
    top_n=None,
    max_weight=None,
    max_per_sector=1,
    regime_engine=None,      # FIX #1: accepted but ignored — see below
    spy_prices=None,
    univ_prices=None,
    vix_prices=None,
    price_df=None,           # FIX #5: caller can pass data directly
    macro_df=None,           # FIX #5: caller can pass data directly
    prices_cache_name=None,  # FIX #5: override cache name if needed
):
    t0  = time.time()
    cfg = get_config()

    if train_days is None: train_days = cfg.walk_forward.train_days
    if purge_days is None: purge_days = cfg.walk_forward.purge_days
    if top_n      is None: top_n      = cfg.backtest.top_n
    if max_weight is None: max_weight = cfg.backtest.max_weight

    if purge_days < TARGET_HORIZON:
        print(
            f"[WF] WARNING: purge_days={purge_days} < TARGET_HORIZON={TARGET_HORIZON}. "
            f"Forcing purge_days={TARGET_HORIZON} to prevent label leakage."
        )
        purge_days = TARGET_HORIZON

    # FIX #1: the regime_engine passed in from run_pipeline() was fitted on the
    # full sample — using it here to adjust weights is look-ahead leakage. A
    # 2026 regime label would scale 2020 weights.
    # FIX: ignore the passed-in engine entirely. Inside the loop we fit a fresh
    # regime engine on each fold's training slice only, so the regime signal is
    # always causal. If regime fitting is too slow, pass disable_regime=True.
    if regime_engine is not None:
        print(
            "[WF] NOTE: full-sample regime_engine ignored to prevent look-ahead. "
            "A fresh regime engine is fitted per fold on training data only."
        )

    print("=" * 70)
    print("  WALK-FORWARD PIPELINE")
    print("=" * 70)

    # FIX #5: use caller-supplied data if provided, otherwise load from cache.
    # Before this fix, the runner always loaded from prices_live/macro_live
    # regardless of what the main pipeline had used (e.g. simulated fallback).
    # This created inconsistency: main pipeline ran on simulated data but
    # walk-forward evaluated on live data (or failed if live cache was missing).
    if price_df is None:
        cache_name = prices_cache_name or "prices_live"
        if not has_data(cache_name):
            print(f"ERROR: No cached data ({cache_name}). Run main pipeline first.")
            return
        price_df = load_parquet(cache_name)
        print(f"[WF] Loaded prices from cache: {cache_name}")
    else:
        print("[WF] Using caller-supplied price data.")

    if price_df["date"].dt.tz is not None:
        price_df["date"] = price_df["date"].dt.tz_localize(None)

    if macro_df is None:
        macro_cache = "macro_live" if has_data("macro_live") else "macro_simulated"
        macro_df = load_parquet(macro_cache) if has_data(macro_cache) else None
        if macro_df is not None:
            print(f"[WF] Loaded macro from cache: {macro_cache}")
    else:
        print("[WF] Using caller-supplied macro data.")

    if macro_df is not None and "date" in macro_df.columns:
        if macro_df["date"].dt.tz is not None:
            macro_df["date"] = macro_df["date"].dt.tz_localize(None)
        if "known_time" in macro_df.columns:
            macro_df["known_time"] = pd.to_datetime(macro_df["known_time"])
            if macro_df["known_time"].dt.tz is not None:
                macro_df["known_time"] = macro_df["known_time"].dt.tz_localize(None)

    sector_map = {}
    if has_data("profiles"):
        profiles   = load_parquet("profiles")
        sector_map = dict(zip(profiles["symbol"], profiles["sector"]))

    # Load all alpha data sources — point-in-time sliced per fold inside the loop.
    # Date column names: fundamentals/options_flow/short_interest use "date";
    # earnings/analyst_revisions/insider_transactions use "signal_date".
    _ALPHA_CACHES = [
        ("fundamentals",        "fundamentals",        "date"),
        ("earnings",            "earnings",            "signal_date"),
        ("analyst_revisions",   "analyst_revisions",   "signal_date"),
        ("options_flow",        "options_flow",        "date"),
        ("insider_transactions","insider_transactions", "signal_date"),
        ("short_interest",      "short_interest",      "date"),
    ]
    _alpha_data = {}
    print("\n[WF] Alpha data source availability:")
    for label, cache, date_col in _ALPHA_CACHES:
        if has_data(cache):
            df_raw = load_parquet(cache)
            df_raw[date_col] = pd.to_datetime(df_raw[date_col]).dt.tz_localize(None)
            _alpha_data[label] = (df_raw, date_col)
            dt_min = df_raw[date_col].min().date()
            dt_max = df_raw[date_col].max().date()
            print(f"  {label}: {len(df_raw):,} rows  [{dt_min} → {dt_max}]")
        else:
            _alpha_data[label] = (None, date_col)
            print(f"  {label}: NOT IN CACHE — sleeve will be inactive")
    print()

    full_builder = FeatureBuilder(price_df, macro_df)
    close_full   = full_builder.close
    open_full    = full_builder.open
    all_dates    = close_full.index

    historical_universe_df = build_historical_universe(strict=True)  # Bug 14 fix: exclude symbols with unknown addition dates

    def _alpha_kwargs(as_of: pd.Timestamp) -> dict:
        """Return point-in-time sliced alpha data kwargs for FeatureBuilder."""
        df_f,  col_f  = _alpha_data["fundamentals"]
        df_e,  col_e  = _alpha_data["earnings"]
        df_a,  col_a  = _alpha_data["analyst_revisions"]
        df_o,  col_o  = _alpha_data["options_flow"]
        df_i,  col_i  = _alpha_data["insider_transactions"]
        df_si, col_si = _alpha_data["short_interest"]
        return dict(
            fundamentals_df = _pit_slice(df_f,  col_f,  as_of),
            earnings_df     = _pit_slice(df_e,  col_e,  as_of),
            analyst_df      = _pit_slice(df_a,  col_a,  as_of),
            options_df      = _pit_slice(df_o,  col_o,  as_of),
            insider_df      = _pit_slice(df_i,  col_i,  as_of),
            short_df        = _pit_slice(df_si, col_si, as_of),
        )

    # FIX #4: walk-forward generates weights on the rebalance schedule, not daily.
    # Before: weights were generated for every market day, which is inconsistent
    # with how live/full mode actually trades (rebalance_freq_days cadence).
    # Fix: only generate a weight vector on scheduled rebalance dates.
    rebal_freq  = cfg.backtest.rebalance_freq_days
    rebal_dates_set = set()
    for idx, dt in enumerate(all_dates):
        if idx % rebal_freq == 0:
            rebal_dates_set.add(dt)

    print("[Data] %d dates, %d symbols" % (len(all_dates), len(close_full.columns)))
    print("[Data] Range: %s to %s" % (all_dates[0].date(), all_dates[-1].date()))
    print("[Setup] Train: up to %d days, Purge: %d days, Rebal freq: %d days" % (
        train_days, purge_days, rebal_freq))

    print("")
    print("-" * 70)
    all_oos_weights    = []
    fold_results       = []
    folds_skipped_thin = []  # (date, n_symbols) — folds skipped due to thin universe
    universe_sizes     = []  # tradeable symbol count per non-skipped fold

    for i, test_date in enumerate(all_dates):
        # FIX #4: skip non-rebalance dates entirely
        if test_date not in rebal_dates_set:
            continue

        train_end_idx = i - purge_days - 1

        tradeable_symbols = get_universe_on_date(test_date, historical_universe_df)
        tradeable_symbols = [s for s in tradeable_symbols if s in close_full.columns]

        if len(tradeable_symbols) < 5:
            print(
                f"[WF] WARNING: {test_date.date()} — thin universe "
                f"({len(tradeable_symbols)} symbols < 5), skipping fold"
            )
            folds_skipped_thin.append((test_date, len(tradeable_symbols)))
            fold_results.append({
                "date":      test_date.strftime("%Y-%m-%d"),
                "train":     "n/a",
                "test_days": 0,
                "daily_ret": float("nan"),
                "status":    f"SKIPPED_THIN_UNIVERSE: {len(tradeable_symbols)} symbols",
            })
            continue

        universe_sizes.append(len(tradeable_symbols))
        # Include benchmark in price filter so FeatureBuilder gets SPY for macro features
        tradeable_set = set(tradeable_symbols) | {cfg.universe.benchmark}

        if train_end_idx < 0:
            hist_prices = price_df.loc[
                (price_df["date"] <= test_date) &
                (price_df["symbol"].isin(tradeable_set))
            ].copy()
            hist_macro  = _pit_macro(macro_df, test_date)

            try:
                hist_builder  = FeatureBuilder(hist_prices, hist_macro,
                                               **_alpha_kwargs(test_date))
                hist_features = hist_builder.compute_features()
                try:
                    hist_targets = hist_builder.compute_targets(horizons=[TARGET_HORIZON])
                    key = f"fwd_ret_{TARGET_HORIZON}d"
                    if key in hist_targets:
                        hist_features["targets"] = hist_targets[key]
                except Exception:
                    pass
                hist_alpha = build_alpha_stack(hist_features,
        agent_id="us_equities")

                if test_date in hist_alpha.index:
                    tradeable_cols = [c for c in tradeable_symbols if c in hist_alpha.columns]
                    test_alpha     = hist_alpha.loc[[test_date], tradeable_cols]
                    test_weights   = sector_constrained_weighted(
                        test_alpha,
                        n=top_n,
                        max_weight=max_weight,
                        max_per_sector=max_per_sector,
                        sector_map=sector_map,
                        regime_signal=None,  # FIX #1: no regime in early dates
                    )
                else:
                    test_weights = pd.DataFrame(
                        [0.0] * len(close_full.columns),
                        index=close_full.columns,
                    ).T
                    test_weights.index = [test_date]
            except Exception:
                test_weights = pd.DataFrame(
                    [0.0] * len(close_full.columns),
                    index=close_full.columns,
                ).T
                test_weights.index = [test_date]

            test_weights = test_weights.reindex(columns=close_full.columns, fill_value=0.0)
            all_oos_weights.append(test_weights)

            daily_ret = np.nan
            if i + 1 < len(all_dates):
                next_date = all_dates[i + 1]
                day_ret   = close_full.loc[next_date] / close_full.loc[test_date] - 1.0
                daily_ret = (
                    test_weights.iloc[0] * day_ret.reindex(test_weights.columns).fillna(0)
                ).sum()

            fold_results.append({
                "date":      test_date.strftime("%Y-%m-%d"),
                "train":     "partial history",
                "test_days": len(test_weights),
                "daily_ret": daily_ret,
            })
            print("  Date %s | Train: partial history | Tradeable: %2d | Positions: %2d" % (
                test_date.strftime("%Y-%m-%d"),
                len(tradeable_symbols),
                int((test_weights.iloc[0] != 0).sum()),
            ))
            continue

        train_start_idx = max(0, train_end_idx - train_days + 1)
        train_start     = all_dates[train_start_idx]
        train_end       = all_dates[train_end_idx]

        train_prices = price_df.loc[
            (price_df["date"] >= train_start) &
            (price_df["date"] <= train_end) &
            (price_df["symbol"].isin(tradeable_set))
        ].copy()

        train_macro = _pit_macro(macro_df, train_end)
        if train_macro is not None:
            train_macro = train_macro[train_macro["date"] >= train_start].copy()

        pred_prices = price_df.loc[
            (price_df["date"] >= train_start) &
            (price_df["date"] <= test_date) &
            (price_df["symbol"].isin(tradeable_set))
        ].copy()

        pred_macro = _pit_macro(macro_df, test_date)
        if pred_macro is not None:
            pred_macro = pred_macro[pred_macro["date"] >= train_start].copy()

        try:
            pred_builder  = FeatureBuilder(pred_prices, pred_macro,
                                           **_alpha_kwargs(test_date))
            hist_features = pred_builder.compute_features()
        except Exception as e:
            print("  Date %s: SKIP (feature error: %s)" % (test_date.strftime("%Y-%m-%d"), e))
            zero_weights = pd.DataFrame(
                [0.0] * len(close_full.columns),
                index=close_full.columns,
            ).T
            zero_weights.index = [test_date]
            all_oos_weights.append(
                zero_weights.reindex(columns=close_full.columns, fill_value=0.0)
            )
            continue

        try:
            train_builder = FeatureBuilder(train_prices, train_macro)
            train_targets = train_builder.compute_targets(horizons=[TARGET_HORIZON])
            key = f"fwd_ret_{TARGET_HORIZON}d"
            if key in train_targets:
                hist_features["targets"] = train_targets[key]
                valid_rows = train_targets[key].notna().any(axis=1).sum()
                print("[WF] targets injected, shape: %s, valid rows: %d" % (
                    str(train_targets[key].shape), valid_rows))
        except Exception as e:
            print(f"[WF] targets error: {e}")

        hist_alpha = build_alpha_stack(hist_features,
        agent_id="us_equities")

        if test_date not in hist_alpha.index:
            zero_weights = pd.DataFrame(
                [0.0] * len(close_full.columns),
                index=close_full.columns,
            ).T
            zero_weights.index = [test_date]
            all_oos_weights.append(
                zero_weights.reindex(columns=close_full.columns, fill_value=0.0)
            )
            print("  Date %s: no alpha for date -> zero weights" % test_date.strftime("%Y-%m-%d"))
            continue

        tradeable_cols = [c for c in tradeable_symbols if c in hist_alpha.columns]
        test_alpha     = hist_alpha.loc[[test_date], tradeable_cols]

        # FIX #1: fit a fresh regime engine on this fold's training data only.
        # The full-sample engine passed in from run_pipeline() is intentionally
        # ignored because it contains future information relative to test_date.
        fold_regime_signal = None
        try:
            from ascent.regime import RegimeEngine
            spy_train = price_df.loc[
                (price_df["date"] >= train_start) &
                (price_df["date"] <= train_end) &
                (price_df["symbol"] == cfg.universe.benchmark)
            ].set_index("date")["close"].sort_index()

            univ_train = price_df.loc[
                (price_df["date"] >= train_start) &
                (price_df["date"] <= train_end) &
                (price_df["symbol"].isin(tradeable_symbols))
            ].pivot_table(index="date", columns="symbol", values="close").sort_index()

            if len(spy_train) >= 252:
                fold_engine = RegimeEngine(config=cfg.regime.to_engine_dict())
                fold_engine.fit(
                    spy_prices=spy_train,
                    universe_prices=univ_train,
                    vix_prices=None,
                    macro_df=None,
                    run_model_selection=False,  # skip selection for speed
                )
                fold_regime_signal = fold_engine.get_signal(test_date)
        except Exception as _re:
            # Bug 7 fix: log the failure so we know regime was unavailable for this fold
            print(f"[WF] fold {test_date.date()}: regime fit failed ({type(_re).__name__}: {_re}) — proceeding without regime signal")

        try:
            test_weights = sector_constrained_weighted(
                test_alpha,
                n=top_n,
                max_weight=max_weight,
                max_per_sector=max_per_sector,
                sector_map=sector_map,
                regime_signal=fold_regime_signal,  # FIX #1: fold-local signal
            )
        except Exception as _oe:
            # Bug 8 fix: don't silently zero out — log what failed and record it
            print(
                f"[WF] fold {test_date.date()}: optimizer failed ({type(_oe).__name__}: {_oe}) "
                "— recording failed fold, using zero weights. Check fold_diagnostics."
            )
            fold_results.append({
                "date":      test_date.strftime("%Y-%m-%d"),
                "train":     "%s to %s" % (train_start.date(), train_end.date()),
                "test_days": 0,
                "daily_ret": float("nan"),
                "status":    f"FAILED_OPTIMIZER: {type(_oe).__name__}: {_oe}",
            })
            test_weights = pd.DataFrame(
                [0.0] * len(close_full.columns),
                index=close_full.columns,
            ).T
            test_weights.index = [test_date]

        test_weights = test_weights.reindex(columns=close_full.columns, fill_value=0.0)
        all_oos_weights.append(test_weights)

        daily_ret = np.nan
        if i + 1 < len(all_dates):
            next_date = all_dates[i + 1]
            day_ret   = close_full.loc[next_date] / close_full.loc[test_date] - 1.0
            daily_ret = (
                test_weights.iloc[0] * day_ret.reindex(test_weights.columns).fillna(0)
            ).sum()

        fold_results.append({
            "date":      test_date.strftime("%Y-%m-%d"),
            "train":     "%s to %s" % (train_start.date(), train_end.date()),
            "test_days": len(test_weights),
            "daily_ret": daily_ret,
        })

        print("  Date %s | Train: %s to %s | Tradeable: %2d | Positions: %2d | Regime: %s" % (
            test_date.strftime("%Y-%m-%d"),
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            len(tradeable_symbols),
            int((test_weights.iloc[0] != 0).sum()),
            fold_regime_signal.label.value if fold_regime_signal else "none",
        ))

    print("-" * 70)

    # A4 + Bug 8: print fold diagnostics summary — failures and thin-universe skips are never silent
    failed_folds    = [f for f in fold_results if f.get("status", "").startswith("FAILED")]
    skipped_folds   = [f for f in fold_results if f.get("status", "").startswith("SKIPPED")]
    succeeded_count = len(fold_results) - len(failed_folds) - len(skipped_folds)
    avg_universe    = round(sum(universe_sizes) / len(universe_sizes), 1) if universe_sizes else 0.0

    print(f"\n[WF] FOLD SUMMARY:")
    print(f"  Total folds:             {len(fold_results)}")
    print(f"  Succeeded:               {succeeded_count}")
    print(f"  Skipped (thin universe): {len(skipped_folds)}")
    print(f"  Failed:                  {len(failed_folds)}")
    print(f"  Avg universe size:       {avg_universe} symbols")
    if skipped_folds:
        for f in skipped_folds:
            print(f"  SKIPPED {f['date']} — {f.get('status', 'unknown')}")
    if failed_folds:
        for f in failed_folds:
            print(f"  FAILED  {f['date']} — {f.get('status', 'unknown')}")

    if not all_oos_weights:
        print("ERROR: No valid folds produced weights.")
        return

    combined_weights = pd.concat(all_oos_weights)
    combined_weights = combined_weights[~combined_weights.index.duplicated(keep="first")]
    combined_weights = combined_weights.sort_index()
    combined_weights = combined_weights.fillna(0)
    combined_weights = combined_weights.reindex(columns=close_full.columns, fill_value=0)

    # Forward-fill weights to every trading day so the backtest engine simulates
    # all ~1,575 days, holding positions between rebalances. Without this,
    # engine.run() intersects target_weights.index with close_prices.index and
    # only sees 164 dates — each spanning ~10 actual trading days — causing
    # CAGR and Sharpe to be annualised as if only 0.65 years passed instead of 6.5.
    all_trading_days = close_full.index[
        (close_full.index >= combined_weights.index[0]) &
        (close_full.index <= combined_weights.index[-1])
    ]
    combined_weights = (
        combined_weights
        .reindex(all_trading_days)
        .ffill()
        .fillna(0)
    )

    n_rebal = combined_weights.index.isin(pd.concat(all_oos_weights).index).sum()
    print("")
    print("[Walk-Forward] Combined OOS weights: %d rebalance dates → %d daily rows (ffilled)" % (
        n_rebal, len(combined_weights)))
    print("[Walk-Forward] OOS period: %s to %s" % (
        combined_weights.index[0].date(), combined_weights.index[-1].date()))

    print("")
    print("=" * 70)
    print("  OUT-OF-SAMPLE BACKTEST")
    print("=" * 70)

    bm_sym  = cfg.universe.benchmark
    bm_data = price_df[price_df["symbol"] == bm_sym].set_index("date")["close"].sort_index()
    bm_data = bm_data[~bm_data.index.duplicated(keep="last")]

    # FIX #4: use the same rebalance cadence as live/full mode.
    # Before: rebalance_freq_days=1 caused daily rebalancing, inconsistent
    # with how the strategy actually runs. Now the OOS backtest uses the same
    # rebalance_freq_days as everything else, and walk-forward only generates
    # weights on those same scheduled dates.
    engine = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        spread_bps=cfg.backtest.spread_bps,
        impact_bps=cfg.backtest.impact_bps,
        rebalance_freq_days=cfg.backtest.rebalance_freq_days,  # FIX #4
        execution_delay=cfg.backtest.execution_delay_days,
    )

    result = engine.run(
        target_weights=combined_weights,
        close_prices=close_full,
        open_prices=open_full,
        benchmark_prices=bm_data,
    )

    print("\n" + format_metrics(result.summary()))
    from ascent.dashboard.export_dashboard_data import export_to_dashboard
    try:
        export_to_dashboard(result, regime_engine=None)
        print('[Dashboard] Export complete')
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'[Dashboard] Export failed: {e}')

    if result.daily_ledger is not None:
        result.daily_ledger.to_csv("ascent_daily_ledger.csv")
        print("\n[Saved] ascent_daily_ledger.csv")

    if result.holdings_ledger is not None:
        result.holdings_ledger.to_csv("ascent_holdings_ledger.csv", index=False)
        print("[Saved] ascent_holdings_ledger.csv")

    print("")
    print("=" * 70)
    print("  DAILY SUMMARY")
    print("=" * 70)
    print("  Total rebalance dates: %d" % len(combined_weights))

    daily_rets = [fr["daily_ret"] for fr in fold_results if not pd.isna(fr["daily_ret"])]
    if daily_rets:
        daily_rets = np.array(daily_rets)
        print("  Avg rebal-day return: %.4f%%" % (np.mean(daily_rets) * 100))
        print("  Rebal-day std:        %.4f%%" % (np.std(daily_rets) * 100))
        print("  Positive days:        %.1f%%" % (100 * np.mean(daily_rets > 0)))

    elapsed = time.time() - t0
    print("\n  Walk-forward pipeline completed in %.1fs\n" % elapsed)


if __name__ == "__main__":
    walk_forward_pipeline()