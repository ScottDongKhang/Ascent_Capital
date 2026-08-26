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
from ascent.research.evaluation import sharpe_ratio, annualized_return, max_drawdown, lo_adjusted_sharpe_ratio
from ascent.research.deflated_sharpe import deflated_sharpe_ratio, KNOWN_TRIAL_COUNT
from ascent.backtest.engine import BacktestEngine
from ascent.research.evaluation import format_metrics
from ascent.data.universe import build_historical_universe, get_universe_on_date, DELISTING_TERMINAL_TERMS

TARGET_HORIZON = 21

# Cap used when folding an infinite (zero-variance, all-positive) Sharpe into
# the WFE ratio, so one degenerate fold can't blow up the aggregate. Matches
# the convention in the retired ascent/research/wf_framework/metrics.py
# (PerformanceAnalyzer.walk_forward_efficiency's _INF_CAP).
_WFE_SHARPE_CAP = 3.0


def _in_sample_fold_sharpe(
    hist_alpha: pd.DataFrame,
    tradeable_symbols: list,
    close_full: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    rebal_dates_set: set,
    top_n,
    max_weight,
    max_per_sector,
    sector_map: dict,
) -> float:
    """In-sample Sharpe for one walk-forward fold.

    Replays the SAME weighting logic used for the fold's OOS decision
    (sector_constrained_weighted with the same top_n/max_weight/max_per_sector)
    over the fold's own training window, using the alpha already computed for
    that window in `hist_alpha` (no second call to build_alpha_stack, no
    re-fetch of features). Weights are generated on the training window's own
    rebalance-cadence dates (the same cadence the OOS side trades on),
    forward-filled to daily, and multiplied by realized next-day training-
    window returns to get a daily-return series whose Sharpe is the fold's
    in-sample Sharpe.

    This is deliberately in-sample -- that's the entire point of WFE (asking
    "how did this exact signal do on the data it was fit on"). It does not
    touch the OOS test_alpha/test_weights computed for this fold, and it does
    not call build_alpha_stack a second time or bypass the IC gate -- the
    alpha values it replays through the optimizer are the same ones that came
    out of the single build_alpha_stack(hist_features, ...) call already made
    for this fold's OOS decision.

    Returns float('nan') if there isn't enough training-window data to form a
    return series (e.g. very early folds with a short training window).
    """
    is_cols = [c for c in tradeable_symbols if c in hist_alpha.columns]
    if not is_cols:
        return float("nan")

    is_dates = [
        d for d in hist_alpha.index
        if train_start <= d <= train_end and d in rebal_dates_set
    ]
    if not is_dates:
        return float("nan")

    weight_rows = []
    for d in is_dates:
        try:
            d_weights = sector_constrained_weighted(
                hist_alpha.loc[[d], is_cols],
                n=top_n,
                max_weight=max_weight,
                max_per_sector=max_per_sector,
                sector_map=sector_map,
                regime_signal=None,  # in-sample replay: no per-day regime refit
            )
        except Exception:
            continue
        weight_rows.append(
            d_weights.reindex(columns=close_full.columns, fill_value=0.0)
        )

    if not weight_rows:
        return float("nan")

    is_weights = pd.concat(weight_rows).sort_index()
    is_weights = is_weights[~is_weights.index.duplicated(keep="first")]

    is_days = close_full.index[
        (close_full.index >= is_weights.index[0]) & (close_full.index <= train_end)
    ]
    if len(is_days) < 6:
        return float("nan")

    is_weights_ff = is_weights.reindex(is_days).ffill().fillna(0.0)
    fwd_ret = close_full.loc[is_days].pct_change().shift(-1).reindex(columns=is_weights_ff.columns)

    # Do NOT fillna(0.0) on fwd_ret directly: a NaN forward return means the price
    # is genuinely missing that day (trading halt, late listing, data gap), not
    # that the symbol was flat. Filling it with 0.0 there would fabricate a real
    # observation and bias this in-sample Sharpe (and therefore WFE) toward zero
    # whenever data is merely missing.
    #
    # Fix: zero the WEIGHT for that (day, symbol) cell wherever the forward
    # return is missing, then redistribute the freed weight, proportionally,
    # among the OTHER symbols that do have a valid return that day -- so a
    # halted/missing name doesn't just silently shrink that day's invested
    # weight (which would itself understate volatility) and doesn't get
    # counted as flat. If every held symbol is missing data on a given day
    # (nothing to redistribute onto), that day's contribution is left at zero
    # -- equivalent to excluding the day, since there is no valid signal left
    # to measure it with.
    valid_ret = fwd_ret.notna()
    masked_weights = is_weights_ff.where(valid_ret, 0.0)
    orig_total = is_weights_ff.sum(axis=1)
    masked_total = masked_weights.sum(axis=1)
    # Rescale factor per day so total invested weight is preserved wherever
    # possible; 0/0 and x/0 both safely become 0 (nothing to redistribute
    # onto), not fabricated exposure.
    rescale = (orig_total / masked_total).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    redistributed_weights = masked_weights.mul(rescale, axis=0)

    # `.fillna(0.0)` on the returns here is inert numerically (it only avoids
    # 0 * NaN = NaN propagating into the sum) since the paired weight is
    # already zero everywhere a return is missing.
    daily_rets = (redistributed_weights * fwd_ret.fillna(0.0)).sum(axis=1)
    daily_rets = daily_rets.iloc[:-1]  # last row has no forward return to pair with

    if len(daily_rets) < 5:
        return float("nan")

    return float(sharpe_ratio(daily_rets, periods_per_year=252))


def _compute_wfe(oos_sharpe: float, fold_is_sharpes: list) -> float | None:
    """Walk-Forward Efficiency = OOS Sharpe / mean(in-sample fold Sharpe).

    Convention note: the retired ascent/research/wf_framework/metrics.py
    defined WFE as mean(OOS_Sharpe_fold / IS_Sharpe_fold) -- a ratio computed
    per fold and then averaged, because that framework refit an optimizer
    per fold and had a distinct OOS return series per fold to pair with it.
    walk_forward_runner.py only computes ONE OOS Sharpe (on the single
    stitched multi-fold backtest via BacktestEngine), so there is no
    per-fold OOS Sharpe to pair against each per-fold IS Sharpe. This uses
    the algebraically simpler, equally standard form instead: aggregate the
    per-fold IS Sharpes first (mean), then take one ratio against the
    overall stitched OOS Sharpe.

    Edge cases (never produce a nonsensical ratio):
      - no fold produced a finite, positive IS Sharpe -> None (can't measure
        "in-sample skill" at all, so WFE is undefined, not zero or infinite).
      - mean IS Sharpe <= 0 -> None, for the same reason (dividing by a
        non-positive in-sample Sharpe produces a sign-flipped or blown-up
        ratio that doesn't mean what WFE is supposed to mean).
      - non-finite OOS Sharpe -> None.
      - a finite but very large OOS Sharpe is capped at +/-_WFE_SHARPE_CAP
        before the division, matching the retired framework's handling of
        zero-variance (infinite) Sharpe folds.
    """
    valid_is = [s for s in fold_is_sharpes if s is not None and np.isfinite(s) and s > 0]
    if not valid_is:
        return None

    mean_is_sharpe = float(np.mean(valid_is))
    if mean_is_sharpe <= 0:
        return None

    if oos_sharpe is None or not np.isfinite(oos_sharpe):
        return None

    oos = oos_sharpe
    if oos > _WFE_SHARPE_CAP:
        oos = _WFE_SHARPE_CAP
    elif oos < -_WFE_SHARPE_CAP:
        oos = -_WFE_SHARPE_CAP

    return float(oos / mean_is_sharpe)


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


def _nearest_prior_index_date(index: pd.DatetimeIndex, target: pd.Timestamp):
    """Latest date in `index` that is <= target, or None if none exists."""
    avail = index[index <= target]
    if len(avail) == 0:
        return None
    return avail.max()


def apply_delisting_terminal_credit(
    close_full: pd.DataFrame,
    open_full: pd.DataFrame | None = None,
):
    """
    Credit real terminal (deal-close) values for the bounded, already-sourced
    set of delisted symbols in ascent.data.universe.DELISTING_TERMINAL_TERMS,
    instead of letting a departing symbol's price simply go NaN on its real
    closure date -- which is what happens today: close_full has no more rows
    for a delisted symbol past its last traded date, so pct_change() on that
    column is NaN forever after, and the return calc's own .fillna(0) (see
    ascent/backtest/engine.py's `daily_returns = close.pct_change().fillna(0)`
    and this module's per-fold diagnostic `day_ret`) silently turns "position
    got acquired for real money" into "position earned exactly 0% forever,"
    inflating OOS performance whenever a held name happens to get acquired.

    Mechanics: for each symbol in DELISTING_TERMINAL_TERMS that has price
    history in close_full, find the nearest trading day <= its REMOVED_STOCKS
    closure date (`removed_date`) and, if that day doesn't already have a
    real price (nothing here overrides real market data), inject the implied
    terminal per-share value there:
      - "cash": the flat cash_amount.
      - "stock" / "cash_and_stock": exchange_ratio * the acquirer's own close
        price, looked up from close_full itself (falling back to the nearest
        prior trading day if the acquirer has no price exactly on the deal's
        close_date), plus cash_amount for cash_and_stock. The acquirer price
        is NEVER hardcoded -- it comes from whatever price data this run is
        actually using, so the credited value stays internally consistent.
    Only that single day's price is touched. Every day after it is left as
    whatever close_full already had there (typically still NaN, since the
    symbol really did stop trading) -- pct_change() from the injected
    terminal price to NaN the following day is itself NaN and gets
    fillna(0)'d, which is correct: the position is gone, no more price
    movement to attribute to it. Because build_historical_universe() ends
    the symbol's tradeable window at that same REMOVED_STOCKS date, no later
    fold's optimizer can select it again -- the credited position cannot
    carry forward or double-count.

    Look-ahead reasoning (CLAUDE.md integrity constraint #1 -- this must not
    reopen it): this is not a new form of look-ahead. The deal terms are
    real, public facts that were knowable as of their announcement/close,
    and the only thing injected here is the single terminal price implied by
    those already-public terms, placed AT the real historical date it
    actually took effect -- never earlier. Every fold whose test_date +
    holding window falls entirely before a symbol's closure date sees
    exactly what it saw before this function existed: real market data (or
    a genuine absence of it), untouched. Only a fold whose test/next-test
    window actually straddles the real closure date is affected. This is
    also not a change in what kind of thing this backtest models: a live
    Alpaca paper account holding one of these names would have been
    force-closed by the broker at real-world deal-close (cash-for-shares
    settles to cash automatically; stock-for-stock converts to the acquirer's
    shares) -- crediting the terminal value here just models that mechanic
    more faithfully instead of pretending the position quietly evaporates.
    """
    if not DELISTING_TERMINAL_TERMS:
        return close_full, open_full

    close_full = close_full.copy()
    if open_full is not None:
        open_full = open_full.copy()

    for sym, terms in DELISTING_TERMINAL_TERMS.items():
        if sym not in close_full.columns:
            continue  # no price history for this symbol at all -- nothing to credit

        removed_date = pd.Timestamp(terms["removed_date"])
        closure_day  = _nearest_prior_index_date(close_full.index, removed_date)
        if closure_day is None:
            continue

        existing = close_full.loc[closure_day, sym]
        if pd.notna(existing):
            # Real data already covers the closure date -- never override real data.
            continue

        sym_series   = close_full[sym]
        valid_before = sym_series.loc[sym_series.index <= closure_day].dropna()
        if valid_before.empty:
            continue  # never had real data for this symbol -- no "held at" price to anchor to

        deal_type = terms["deal_type"]
        if deal_type == "cash":
            terminal_value = terms["cash_amount"]
        elif deal_type in ("stock", "cash_and_stock"):
            acquirer = terms["acquirer_symbol"]
            close_lookup_day = _nearest_prior_index_date(
                close_full.index, pd.Timestamp(terms["close_date"])
            )
            acquirer_price = None
            if acquirer in close_full.columns and close_lookup_day is not None:
                acq_series = close_full[acquirer]
                acq_valid  = acq_series.loc[acq_series.index <= close_lookup_day].dropna()
                if not acq_valid.empty:
                    acquirer_price = float(acq_valid.iloc[-1])
            if acquirer_price is None:
                print(
                    f"[WF] delisting credit: {sym} -- no acquirer ({acquirer}) price "
                    f"available near {terms['close_date']}, skipping credit "
                    "(falls back to the pre-existing silent-drop behavior)"
                )
                continue
            terminal_value = terms["exchange_ratio"] * acquirer_price
            if deal_type == "cash_and_stock":
                terminal_value += terms["cash_amount"]
        else:
            continue

        close_full.loc[closure_day, sym] = terminal_value
        if open_full is not None and sym in open_full.columns:
            # Keep open/close consistent on the closure day so this credit fires
            # correctly regardless of whether closure_day happens to land on a
            # rebalance date (engine.py uses open-vs-close for rebalance days,
            # close-vs-close drift otherwise -- see BacktestEngine.run()).
            open_full.loc[closure_day, sym] = terminal_value

    return close_full, open_full


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

    # Credit real terminal (deal-close) values for the bounded set of delisted
    # symbols in DELISTING_TERMINAL_TERMS instead of letting their price series
    # just go NaN on closure -- see apply_delisting_terminal_credit()'s
    # docstring for the mechanics and the look-ahead reasoning.
    close_full, open_full = apply_delisting_terminal_credit(close_full, open_full)

    all_dates    = close_full.index

    # sp500_only=True: restricts to S&P 500 members + REMOVED_STOCKS.
    # The 807 S&P 400 symbols all default to 2020-01-01 (unknown addition dates),
    # which inflates OOS returns by trading a forward-selected winner universe.
    # Option B (full S&P 400 constituent history) will fix this properly.
    historical_universe_df = build_historical_universe(strict=True, sp500_only=True)

    # Resolve the alpha sleeve weights actually used for this run, once, up front --
    # both build_alpha_stack() call sites below pass this explicitly (instead of
    # leaving alpha_weights=None and letting build_alpha_stack re-resolve internally
    # per fold) so the resolved dict is guaranteed to be what every fold used, and so
    # the wf_report JSON's _meta.alpha_overrides below reflects ground truth instead
    # of a hardcoded {"meanrev": 0.5, "statarb": 0.5} guess.
    from ascent.alpha.stack import _load_active_alpha_weights, _get_gated_weights
    resolved_alpha_weights = _get_gated_weights(_load_active_alpha_weights())

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
    fold_is_sharpes    = []  # per-fold in-sample Sharpe, for Walk-Forward Efficiency

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
        agent_id="us_equities", alpha_weights=resolved_alpha_weights)

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
        agent_id="us_equities", alpha_weights=resolved_alpha_weights)

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

        # WFE: in-sample Sharpe for this fold, replaying the same weighting
        # logic over the training window using the alpha already computed
        # above. See _in_sample_fold_sharpe docstring — deliberately
        # in-sample, does not affect test_alpha/test_weights below.
        fold_is_sharpe = _in_sample_fold_sharpe(
            hist_alpha, tradeable_symbols, close_full,
            train_start, train_end, rebal_dates_set,
            top_n, max_weight, max_per_sector, sector_map,
        )
        fold_is_sharpes.append(fold_is_sharpe)

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

        print("  Date %s | Train: %s to %s | Tradeable: %2d | Positions: %2d | Regime: %s | IS_Sh: %s" % (
            test_date.strftime("%Y-%m-%d"),
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            len(tradeable_symbols),
            int((test_weights.iloc[0] != 0).sum()),
            fold_regime_signal.label.value if fold_regime_signal else "none",
            ("%.2f" % fold_is_sharpe) if np.isfinite(fold_is_sharpe) else "n/a",
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

    wf_summary = result.summary()
    print("\n" + format_metrics(wf_summary))

    # --- Walk-Forward Efficiency -------------------------------------------
    # WFE = OOS Sharpe (this stitched multi-fold backtest) / mean(per-fold
    # in-sample Sharpe collected during the loop above). See _compute_wfe's
    # docstring for the edge-case handling and how this differs from the
    # retired wf_framework's per-fold-ratio-then-mean convention.
    valid_fold_is_sharpes = [s for s in fold_is_sharpes if np.isfinite(s)]
    mean_is_sharpe = float(np.mean([s for s in valid_fold_is_sharpes if s > 0])) \
        if any(s > 0 for s in valid_fold_is_sharpes) else float("nan")
    wfe = _compute_wfe(wf_summary.get("sharpe"), fold_is_sharpes)

    # --- Lo (2002) autocorrelation-adjusted Sharpe -------------------------
    # ADDITIONAL metric alongside (never replacing) wf_summary["sharpe"]. The
    # q-day rebalance cadence (weights forward-filled daily between
    # rebalances) mechanically induces positive serial correlation in the
    # daily OOS return series, which the naive annualized Sharpe does not
    # correct for. See lo_adjusted_sharpe_ratio()'s docstring in
    # ascent/research/evaluation.py for the formula and edge-case handling.
    lo_sharpe = lo_adjusted_sharpe_ratio(
        result.portfolio_returns, q=cfg.backtest.rebalance_freq_days
    )
    print("[Lo-2002] Autocorrelation-adjusted Sharpe (q=%d): %.3f  (naive: %.3f)" % (
        cfg.backtest.rebalance_freq_days, lo_sharpe, wf_summary.get("sharpe", float("nan"))
    ))

    # --- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) --------------
    # ADDITIONAL metric alongside (never replacing) wf_summary["sharpe"].
    # Corrects for selection bias across KNOWN_TRIAL_COUNT distinct
    # strategy/config trials this project has run (curated list + citations
    # in ascent/research/deflated_sharpe.py) and for non-normal returns
    # (skew/kurtosis, already computed by compute_all_metrics() above and
    # reused here rather than recomputed). See deflated_sharpe_ratio()'s
    # docstring for the SR-variance-across-trials fallback this uses since
    # per-trial Sharpe values aren't logged anywhere in this codebase.
    dsr = deflated_sharpe_ratio(
        sharpe_observed=wf_summary.get("sharpe", 0.0),
        n_trials=KNOWN_TRIAL_COUNT,
        skew=wf_summary.get("skewness", 0.0),
        kurtosis=wf_summary.get("kurtosis", 0.0),
        n_obs=wf_summary.get("n_days", len(result.portfolio_returns)),
    )
    print("[DSR] Deflated Sharpe Ratio (n_trials=%d): %.3f" % (KNOWN_TRIAL_COUNT, dsr))

    print("")
    print("[WFE] Folds with a computable IS Sharpe: %d / %d" % (
        len(valid_fold_is_sharpes), len(fold_is_sharpes)))
    if np.isfinite(mean_is_sharpe):
        print("[WFE] Mean in-sample Sharpe: %.3f" % mean_is_sharpe)
    else:
        print("[WFE] Mean in-sample Sharpe: n/a (no fold produced a positive, finite IS Sharpe)")
    if wfe is not None:
        label = "acceptable" if wfe >= 0.5 else "OVERFIT"
        print("[WFE] Walk-Forward Efficiency: %.3f  (%s)" % (wfe, label))
    else:
        print("[WFE] Walk-Forward Efficiency: n/a (see edge-case handling in _compute_wfe)")

    try:
        # NOTE: pre-existing bug found while testing the persistence patch
        # below -- ascent/dashboard/export_dashboard_data.py was deleted in
        # commit ab392bb ("chore: remove dashboard HTML, 20in20 reports, and
        # unused UI/reporting modules") but this import was left outside the
        # try/except, so it raised ModuleNotFoundError unconditionally and
        # crashed the pipeline before it could reach the ledger/report saves
        # below. Moved inside the try (previously already the intended
        # failure-handling for this block) so a missing/broken dashboard
        # exporter degrades gracefully instead of aborting the whole run.
        # Not part of the requested persistence change; flagged separately.
        from ascent.dashboard.export_dashboard_data import export_to_dashboard
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

    # --- Persistence: daily strategy + benchmark return series -------------
    # docs/target_architecture/24_beta_decomposition_analysis.md found that no
    # per-day OOS return series was ever persisted alongside the summary-stat
    # report, forcing a prior audit to ALGEBRAICALLY estimate a beta-hedged
    # Sharpe from aggregate stats. result.portfolio_returns (net daily strategy
    # return) and result.benchmark_returns (SPY daily pct-change, same
    # common_dates index) already hold exactly that series in memory inside
    # BacktestEngine.run() / BacktestResult above. This writes them out
    # verbatim, additively -- it does not change any existing computation,
    # the existing JSON summary report, or the existing ledger CSVs.
    try:
        from ascent.utils.market_time import market_today
        _run_date = market_today().isoformat()
    except Exception:
        _run_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    daily_returns_path = os.path.join(
        "outputs", "wf_results", f"wf_daily_returns_{_run_date}.csv"
    )
    try:
        os.makedirs(os.path.dirname(daily_returns_path), exist_ok=True)
        _daily_df = pd.DataFrame({"strategy_return": result.portfolio_returns})
        if result.benchmark_returns is not None:
            _daily_df["benchmark_return"] = result.benchmark_returns.reindex(
                _daily_df.index
            )
        _daily_df.index.name = "date"
        _daily_df.to_csv(daily_returns_path)
        print(f"\n[Saved] {daily_returns_path} ({len(_daily_df)} rows)")
    except Exception as _pe:
        print(f"[WF] WARNING: failed to persist daily returns series: {_pe}")

    # --- Persistence: wf_report_<date>.json, matching the schema
    # ascent/reporting/verified_numbers.py::load_wf_report() reads (the same
    # shape scripts/generate_wf_report_from_runner.py hand-packages from a
    # log file for the CANONICAL_WF_ARTIFACT pointer). Writing it here closes
    # the gap that script's docstring calls out: "wfe": null because this
    # runner didn't track per-fold in-sample Sharpe. It now does, so this
    # artifact carries a real "wfe" value end-to-end to canonical_wf() if a
    # run of this pipeline is ever repointed to as the canonical artifact.
    # This does NOT touch CANONICAL_WF_ARTIFACT or CURRENT_VERIFIED_NUMBERS.md
    # -- promoting a run to "canonical" remains a deliberate, separate act.
    wf_report_path = os.path.join("outputs", "wf_results", f"wf_report_{_run_date}.json")
    try:
        os.makedirs(os.path.dirname(wf_report_path), exist_ok=True)
        wf_report = {
            "cagr":         wf_summary.get("cagr"),
            "volatility":   wf_summary.get("volatility"),
            "sharpe":       wf_summary.get("sharpe"),
            "sharpe_lo_adjusted": lo_sharpe,
            "deflated_sharpe_ratio": dsr,
            "sortino":      wf_summary.get("sortino"),
            "max_drawdown": wf_summary.get("max_drawdown"),
            "win_rate":     wf_summary.get("hit_rate"),
            "wfe":          wfe,
            "alpha":        wf_summary.get("alpha"),
            "beta":         wf_summary.get("beta"),
            "n_folds":      succeeded_count,
            "n_oos_days":   wf_summary.get("n_days", len(result.portfolio_returns)),
            "_meta": {
                "framework": "ascent/research/walk_forward_runner.py (walk_forward_pipeline)",
                "oos_window": "%s -> %s" % (
                    combined_weights.index[0].date(), combined_weights.index[-1].date()
                ),
                "mean_is_sharpe": mean_is_sharpe if np.isfinite(mean_is_sharpe) else None,
                "n_folds_with_is_sharpe": len(valid_fold_is_sharpes),
                "wfe_definition": (
                    "OOS Sharpe (stitched multi-fold backtest) / mean(per-fold "
                    "in-sample Sharpe), capped OOS Sharpe at +/-3.0. See "
                    "_compute_wfe() in walk_forward_runner.py."
                ),
                "alpha_overrides": resolved_alpha_weights,
                "sharpe_lo_adjusted_q": cfg.backtest.rebalance_freq_days,
                "sharpe_lo_adjusted_definition": (
                    "Lo (2002) autocorrelation-adjusted annualized Sharpe on "
                    "result.portfolio_returns, q=cfg.backtest.rebalance_freq_days. "
                    "See lo_adjusted_sharpe_ratio() in ascent/research/evaluation.py."
                ),
                "deflated_sharpe_ratio_n_trials": KNOWN_TRIAL_COUNT,
                "deflated_sharpe_ratio_definition": (
                    "Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio: "
                    "probability true Sharpe > 0 after correcting for "
                    "selection bias across KNOWN_TRIAL_COUNT trials (curated, "
                    "see ascent/research/deflated_sharpe.py) and non-normal "
                    "returns (skew/kurtosis of result.portfolio_returns, "
                    "n_obs=wf_summary['n_days']). SR-variance-across-trials "
                    "input uses the documented Mertens (2002) fallback -- see "
                    "deflated_sharpe_ratio() docstring."
                ),
            },
        }
        with open(wf_report_path, "w") as f:
            import json
            json.dump(wf_report, f, indent=2, default=float)
        print(f"[Saved] {wf_report_path}")
    except Exception as _we:
        print(f"[WF] WARNING: failed to persist wf_report JSON: {_we}")

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

    if wfe is not None:
        print("  Walk-Forward Efficiency: %.3f" % wfe)
    else:
        print("  Walk-Forward Efficiency: n/a")
    print("  Lo-adjusted Sharpe (q=%d): %.3f  (naive: %.3f)" % (
        cfg.backtest.rebalance_freq_days, lo_sharpe, wf_summary.get("sharpe", float("nan"))
    ))
    print("  Deflated Sharpe Ratio (n_trials=%d): %.3f" % (KNOWN_TRIAL_COUNT, dsr))

    elapsed = time.time() - t0
    print("\n  Walk-forward pipeline completed in %.1fs\n" % elapsed)


if __name__ == "__main__":
    walk_forward_pipeline()