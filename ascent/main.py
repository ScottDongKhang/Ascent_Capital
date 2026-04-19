#!/usr/bin/env python3
"""
Ascent Capital — Main Pipeline Runner
End-to-end: data → features → alpha → portfolio → backtest → report
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ascent.config.settings import get_config, Config
from ascent.data.ingest.simulated import generate_price_data, generate_macro_data
from ascent.data.normalize.prices import normalize_prices, normalize_macro, pivot_prices
from ascent.data.store.parquet import save_parquet, load_parquet, has_data, validate_cache
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack
from ascent.portfolio.optimizer import top_n_equal_weight, rank_weighted, sector_constrained_weighted
from ascent.backtest.engine import BacktestEngine
from ascent.research.evaluation import format_metrics


_LIVE_STALE_DAYS = 3
_SIM_STALE_DAYS  = 0


def load_or_fetch_prices(cfg: Config, live: bool) -> tuple[pd.DataFrame, str]:
    """Returns (price_df, cache_name_used)."""
    cache_name       = "prices_live" if live else "prices_simulated"
    stale_days       = _LIVE_STALE_DAYS if live else _SIM_STALE_DAYS
    required_symbols = cfg.universe.symbols + [cfg.universe.benchmark]

    # Hub bypass: if the centralized hub ran recently, trust its data and skip
    # validate_cache. validate_cache would fail because required_end=today but
    # today's close isn't available until after market close. The hub guarantees
    # data is as fresh as it can be — re-fetching wastes time and hits rate limits.
    if live:
        try:
            from ascent.data.hub import hub_is_fresh
            if hub_is_fresh():
                df = load_parquet(cache_name)
                if not df.empty:
                    print("[Data] Hub data is fresh — loading prices_live.parquet (skipping re-fetch)")
                    return df, cache_name
        except Exception:
            pass  # fall through to validate_cache path

    ok, reason = validate_cache(
        cache_name,
        required_start=cfg.backtest.start_date,
        required_end=cfg.backtest.end_date if live else None,
        required_symbols=required_symbols,
        stale_days=stale_days,
    )

    if ok:
        print("[Data] Loading cached price data...")
        return load_parquet(cache_name), cache_name

    print(f"[Data] Cache invalid ({reason}) — refreshing...")

    if live:
        print("[Data] Fetching live price data...")
        try:
            from ascent.data.ingest.yahoo import fetch_universe_daily
            df = fetch_universe_daily(
                required_symbols,
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
        except Exception as e:
            print(f"[Data] Yahoo failed: {e}")
            df = pd.DataFrame()

        if not df.empty:
            fetched_symbols = set(df["symbol"].unique())
            required_set    = set(required_symbols)
            missing_symbols = required_set - fetched_symbols
            if missing_symbols:
                print(
                    f"[Data] WARNING: live fetch returned only {len(fetched_symbols)} of "
                    f"{len(required_set)} required symbols. "
                    f"Missing: {sorted(missing_symbols)} — likely delisted, using available symbols."
                )
                # do not wipe df — use whatever Yahoo returned

        if df.empty:
            print("[Data] No complete live data — falling back to simulated")
            df = generate_price_data(
                required_symbols,
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
            df = normalize_prices(df)
            # Bug 11 fix: write to a distinct cache name so future runs never
            # confuse this simulated fallback with real fetched market data.
            # "prices_live" is reserved for genuine Yahoo/Polygon data only.
            fallback_cache = "prices_live_fallback_simulated"
            save_parquet(df, fallback_cache)
            print(
                f"[Data] WARNING: live fetch failed — saved SIMULATED fallback to "
                f"'{fallback_cache}' (NOT prices_live). "
                "Walk-forward will use simulated data."
            )
            return df, fallback_cache

    else:
        print("[Data] Generating simulated price data...")
        df = generate_price_data(
            required_symbols,
            cfg.backtest.start_date,
            cfg.backtest.end_date,
        )

    df = normalize_prices(df)
    save_parquet(df, cache_name)
    print(f"[Data] {len(df)} rows, {df['symbol'].nunique()} symbols cached")
    return df, cache_name


def load_or_fetch_macro(cfg: Config, live: bool) -> tuple[pd.DataFrame, str]:
    """Returns (macro_df, cache_name_used)."""
    cache_name = "macro_live" if live else "macro_simulated"
    stale_days = _LIVE_STALE_DAYS if live else _SIM_STALE_DAYS

    # FIX #3: use series_id consistently — simulated.py and fred.py both write
    # series_id, not series. validate_cache was checking symbol_col="series"
    # which never matched, so the cache was always treated as missing.
    ok, reason = validate_cache(
        cache_name,
        required_start=cfg.backtest.start_date,
        required_end=cfg.backtest.end_date if live else None,
        stale_days=stale_days,
        symbol_col="series_id",   # FIX #3: was "series"
    )

    if ok:
        print("[Data] Loading cached macro data...")
        return load_parquet(cache_name), cache_name

    print(f"[Data] Macro cache invalid ({reason}) — refreshing...")

    if live:
        try:
            from ascent.data.ingest.fred import fetch_all_macro
            df = fetch_all_macro(
                start_date=cfg.backtest.start_date,
                end_date=cfg.backtest.end_date,
            )
        except Exception as e:
            print(f"[Data] FRED failed ({e}), using simulated macro")
            df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)
        if df.empty:
            df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)
    else:
        print("[Data] Generating simulated macro data...")
        df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)

    df = normalize_macro(df)
    save_parquet(df, cache_name)
    return df, cache_name


def run_pipeline(
    live: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance_days: int | None = None,
) -> tuple:
    cfg = get_config()
    if not cfg.backtest.end_date:
        from datetime import date
        cfg.backtest.end_date = pd.bdate_range(end="today", periods=1)[0].strftime("%Y-%m-%d")
        print(f"[Config] End date set to today: {cfg.backtest.end_date}")

    if start_date:     cfg.backtest.start_date         = start_date
    if end_date:       cfg.backtest.end_date            = end_date
    if top_n:          cfg.backtest.top_n               = top_n
    if rebalance_days: cfg.backtest.rebalance_freq_days = rebalance_days

    t0 = time.time()

    print("\n" + "=" * 70)
    print("  STEP 1: DATA INGESTION")
    print("=" * 70)

    # FIX #2: capture the cache name actually used so we can pass it to
    # walk_forward_pipeline — avoiding hardcoded "prices_live" lookups later
    price_df, price_cache_name = load_or_fetch_prices(cfg, live)
    macro_df, macro_cache_name = load_or_fetch_macro(cfg, live)

    benchmark_sym  = cfg.universe.benchmark
    benchmark_mask = price_df["symbol"] == benchmark_sym
    benchmark_df   = price_df[benchmark_mask].copy()
    universe_df    = price_df[~benchmark_mask].copy()

    symbols = sorted(universe_df["symbol"].unique())
    print(f"[Data] Universe: {len(symbols)} symbols")
    print(f"[Data] Date range: {universe_df['date'].min().date()} → {universe_df['date'].max().date()}")

    print("\n" + "=" * 70)
    print("  STEP 1.5: REGIME ENGINE")
    print("=" * 70)

    regime_engine = None
    regime_signal = None
    spy_wide      = None
    univ_wide     = None
    vix_series    = None

    try:
        from ascent.regime import RegimeEngine, apply_regime_to_portfolio

        spy_wide = (
            benchmark_df.set_index("date")["close"]
            .sort_index()
            .pipe(lambda s: s[~s.index.duplicated(keep="last")])
        )
        univ_wide = (
            universe_df.pivot_table(index="date", columns="symbol", values="close")
            .sort_index()
        )

        vix_series = None
        market_prices_df = None
        if live:
            try:
                import yfinance as yf
                vix_raw = yf.download(
                    "^VIX",
                    start=cfg.backtest.start_date,
                    end=cfg.backtest.end_date,
                    progress=False,
                )["Close"].squeeze()
                vix_raw.index = pd.to_datetime(vix_raw.index).tz_localize(None)
                vix_series = vix_raw
                print("[Regime] VIX data fetched")
            except Exception as e:
                print(f"[Regime] VIX fetch skipped: {e}")

            # Fetch credit/yield instruments for enhanced regime detection
            try:
                import yfinance as _yf
                _mkt_raw = _yf.download(
                    ["HYG", "LQD", "TLT", "IEF"],
                    start=str(cfg.backtest.start_date),
                    end=str(cfg.backtest.end_date),
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if isinstance(_mkt_raw.columns, pd.MultiIndex):
                    _close_key = [c for c in _mkt_raw.columns.get_level_values(0) if str(c).lower() == "close"]
                    if _close_key:
                        _mkt_raw = _mkt_raw[_close_key[0]]
                market_prices_df = _mkt_raw.reindex(columns=["HYG", "LQD", "TLT", "IEF"])
                market_prices_df.index = pd.to_datetime(market_prices_df.index).tz_localize(None)
                print(f"[Regime] Credit/yield instruments fetched: {market_prices_df.shape}")
            except Exception as _e:
                print(f"[Regime] Credit/yield fetch skipped: {_e}")

        if hasattr(spy_wide.index, "tz") and spy_wide.index.tz is not None:
            spy_wide.index = spy_wide.index.tz_localize(None)
        if hasattr(univ_wide.index, "tz") and univ_wide.index.tz is not None:
            univ_wide.index = univ_wide.index.tz_localize(None)

        macro_wide = None
        if macro_df is not None and not macro_df.empty:
            try:
                # FIX #3: pivot on series_id — matches what simulated.py and
                # fred.py write. The old "series" column doesn't exist in either.
                pivot_col = "series_id" if "series_id" in macro_df.columns else "series"
                macro_wide = macro_df.pivot_table(
                    index="date", columns=pivot_col, values="value"
                ).sort_index()
                if hasattr(macro_wide.index, "tz") and macro_wide.index.tz is not None:
                    macro_wide.index = macro_wide.index.tz_localize(None)
                print(f"[Regime] Macro panel: {macro_wide.shape[1]} series, "
                      f"{macro_wide.shape[0]} dates")
            except Exception as e:
                print(f"[Regime] Macro pivot failed ({e}) — fitting without macro")
                macro_wide = None

        regime_engine = RegimeEngine(config=cfg.regime.to_engine_dict())
        regime_engine.fit(
            spy_prices=spy_wide,
            universe_prices=univ_wide,
            vix_prices=vix_series,
            macro_df=macro_wide,
            market_prices=market_prices_df,
            run_model_selection=True,
        )

        last_date     = spy_wide.index[-1]
        regime_signal = regime_engine.get_signal(last_date)

        if regime_signal:
            print(
                f"[Regime] K={regime_engine.best_k}  "
                f"label={regime_signal.label.value}  "
                f"risk_mult={regime_signal.risk_multiplier:.2f}  "
                f"entropy={regime_signal.entropy:.3f}"
            )
        else:
            print("[Regime] Engine fitted but no signal returned — continuing without regime")

    except Exception as exc:
        print(f"[Regime] Engine failed ({exc}) — pipeline continues without regime adjustment")
        regime_engine = None
        regime_signal = None

    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    from ascent.data.store.parquet import has_data as _hd, load_parquet as _lp
    fundamentals_df = None
    if _hd("fundamentals"):
        try:
            fundamentals_df = _lp("fundamentals")
            print(f"[Alpha] Fundamentals loaded: {len(fundamentals_df)} rows")
        except Exception as _fe:
            print(f"[Alpha] Fundamentals load failed: {_fe}")

    builder  = FeatureBuilder(universe_df, macro_df, fundamentals_df=fundamentals_df)
    features = builder.compute_features()
    targets  = builder.compute_targets(horizons=[1, 5, 21])

    print(f"[Features] Computed {len(features)} features across {len(builder.symbols)} symbols")
    print(f"[Features] Date range: {builder.dates.min().date()} → {builder.dates.max().date()}")

    print("\n" + "=" * 70)
    print("  STEP 3: ALPHA GENERATION")
    print("=" * 70)

    # Inject 21-day forward returns into features dict so ML sleeve can train
    if "fwd_ret_21d" in targets:
        features["targets"] = targets["fwd_ret_21d"]

    alpha = build_alpha_stack(features, regime_signal=None)
    print(f"[Alpha] Composite alpha computed: {alpha.shape}")
    print("[Alpha] NOTE: regime adjustment disabled in full-mode (look-ahead risk). "
          "Regime is applied causally in walk-forward mode only.")

    if "fwd_ret_21d" in targets:
        fwd    = targets["fwd_ret_21d"]
        common = alpha.index.intersection(fwd.index)
        if len(common) > 100:
            ic_series = alpha.loc[common].corrwith(fwd.loc[common], axis=1)
            mean_ic   = ic_series.mean()
            print(f"[Alpha] Mean IC (21d fwd return): {mean_ic:.4f}")
            print(f"[Alpha] IC t-stat: {mean_ic / (ic_series.std() / np.sqrt(len(ic_series))):.2f}")

    print("\n" + "=" * 70)
    print("  STEP 4: PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    from ascent.data.store.parquet import has_data as _hd, load_parquet as _lp
    sector_map = {}
    if _hd("profiles"):
        _prof      = _lp("profiles")
        sector_map = dict(zip(_prof["symbol"], _prof["sector"]))
        print("[Portfolio] Sector constraint: max 1 per sector")

    target_weights = sector_constrained_weighted(
        alpha,
        n=cfg.backtest.top_n,
        max_weight=cfg.backtest.max_weight,
        max_per_sector=1,
        sector_map=sector_map,
        regime_signal=None,
    )

    warmup = 252 + 21
    if len(target_weights) > warmup:
        target_weights = target_weights.iloc[warmup:]

    try:
        spy_close         = benchmark_df.set_index("date")["close"].sort_index()
        spy_close         = spy_close[~spy_close.index.duplicated(keep="last")]
        spy_ma200         = spy_close.rolling(200, min_periods=150).mean()
        spy_below_ma      = spy_close < spy_ma200
        spy_below_aligned = spy_below_ma.reindex(target_weights.index, method="ffill").fillna(False)
        below_dates       = spy_below_aligned[spy_below_aligned].index
        if len(below_dates) > 0:
            target_weights.loc[below_dates] = target_weights.loc[below_dates] * 0.70
            pct = len(below_dates) / max(len(target_weights), 1) * 100
            print(f"[Portfolio] SPY 200d MA filter: {len(below_dates)} dates below MA ({pct:.1f}%) → 30% exposure cut")
        else:
            print("[Portfolio] SPY 200d MA filter: SPY above MA on all dates — no cuts applied")
    except Exception as _e:
        print(f"[Portfolio] SPY 200d MA filter skipped: {_e}")

    active_dates = (target_weights.sum(axis=1) > 0.01).sum()
    print(f"[Portfolio] Target weights: {target_weights.shape}")
    print(f"[Portfolio] Active rebalance dates: {active_dates}")
    print(f"[Portfolio] Top-{cfg.backtest.top_n} rank-weighted, max weight {cfg.backtest.max_weight:.0%}")

    print("\n" + "=" * 70)
    print("  STEP 5: BACKTEST")
    print("=" * 70)

    engine = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        spread_bps=cfg.backtest.spread_bps,
        impact_bps=cfg.backtest.impact_bps,
        rebalance_freq_days=cfg.backtest.rebalance_freq_days,
        execution_delay=cfg.backtest.execution_delay_days,
    )

    close    = builder.close
    open_    = builder.open
    bm_close = benchmark_df.set_index("date")["close"].sort_index()
    bm_close = bm_close[~bm_close.index.duplicated(keep="last")]

    result = engine.run(
        target_weights=target_weights,
        close_prices=close,
        open_prices=open_,
        benchmark_prices=bm_close,
    )


    if regime_engine is not None:
        try:
            from ascent.regime.diagnostics import regime_occupancy_table, regime_return_stats
            sigs = regime_engine.get_signal_series()
            if not sigs.empty:
                print("\n── Regime Occupancy ──────────────────────────────────")
                print(regime_occupancy_table(sigs).to_string())
                spy_ret = bm_close.pct_change().dropna()
                print("\n── Per-Regime Return Stats ───────────────────────────")
                print(regime_return_stats(spy_ret, sigs).to_string())
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"\n  Pipeline completed in {elapsed:.1f}s\n")

    if regime_engine is not None:
        try:
            import os
            os.makedirs("dashboard", exist_ok=True)
            regime_engine.save_for_intel(dashboard_dir="dashboard")
        except Exception as _e:
            print(f"[Regime] Intel export failed: {_e}")
    else:
        print("[Regime] No regime engine - Intel export skipped")

    return result, regime_engine, spy_wide, univ_wide, vix_series, target_weights, price_df, macro_df, price_cache_name


def main():
    parser = argparse.ArgumentParser(description="Ascent Capital — Quant Trading Platform")
    parser.add_argument("--live",          action="store_true")
    parser.add_argument("--start",         type=str, default=None)
    parser.add_argument("--end",           type=str, default=None)
    parser.add_argument("--top-n",         type=int, default=None)
    parser.add_argument("--rebalance",     type=int, default=None)
    parser.add_argument("--clear-cache",   action="store_true")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass all parquet caches and re-fetch/re-generate data")

    args = parser.parse_args()

    if args.force_refresh:
        import os
        os.environ["ASCENT_FORCE_REFRESH"] = "1"
        print("[Cache] Force-refresh enabled — all caches will be ignored this run")

    if args.clear_cache:
        import shutil
        cfg = get_config()
        if cfg.data_dir.exists():
            shutil.rmtree(cfg.data_dir)
            print("Cache cleared.")

    run_pipeline(
        live=args.live,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        rebalance_days=args.rebalance,
    )


if __name__ == "__main__":
    main()