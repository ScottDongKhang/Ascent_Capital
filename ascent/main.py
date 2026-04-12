#!/usr/bin/env python3
"""
Ascent Capital — Main Pipeline Runner
End-to-end: data → features → alpha → portfolio → backtest → report

Usage:
    python -m ascent.main                    # Run with simulated data
    python -m ascent.main --live             # Run with real API data
    python -m ascent.main --start 2020-01-01 --end 2024-12-31
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Load .env so API keys are available when get_config() runs
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ascent.config.settings import get_config, Config
from ascent.data.ingest.simulated import generate_price_data, generate_macro_data
from ascent.data.normalize.prices import normalize_prices, normalize_macro, pivot_prices
from ascent.data.store.parquet import save_parquet, load_parquet, has_data
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack, alpha_to_ranks
from ascent.portfolio.optimizer import top_n_equal_weight, rank_weighted, sector_constrained_weighted
from ascent.backtest.engine import BacktestEngine
from ascent.backtest.reports import print_report
from ascent.research.evaluation import format_metrics


def load_or_fetch_prices(cfg: Config, live: bool) -> pd.DataFrame:
    """Load cached prices or fetch from APIs."""
    cache_name = "prices_live" if live else "prices_simulated"

    if has_data(cache_name):
        print("[Data] Loading cached price data...")
        df = load_parquet(cache_name)
        return df
    if live:
        print("[Data] Fetching live price data...")
        try:
            from ascent.data.ingest.tiingo import fetch_universe_daily
            df = fetch_universe_daily(
                cfg.universe.symbols + [cfg.universe.benchmark],
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
        except (ValueError, Exception) as e:
            print(f"[Data] Tiingo failed ({e}), falling back to simulated")
            df = generate_price_data(
                cfg.universe.symbols + [cfg.universe.benchmark],
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
        if df.empty:
            print("[Data] Live fetch returned no data, trying Tiingo then simulated...")
            try:
                from ascent.data.ingest.tiingo import fetch_universe_daily
                df = fetch_universe_daily(
                    cfg.universe.symbols + [cfg.universe.benchmark],
                    cfg.backtest.start_date,
                    cfg.backtest.end_date,
                )
            except Exception:
                pass
            if df.empty:
                df = generate_price_data(
                    cfg.universe.symbols + [cfg.universe.benchmark],
                    cfg.backtest.start_date,
                    cfg.backtest.end_date,
                )
    else:
        print("[Data] Generating simulated price data...")
        df = generate_price_data(
            cfg.universe.symbols + [cfg.universe.benchmark],
            cfg.backtest.start_date,
            cfg.backtest.end_date,
        )

    # Normalize and cache
    df = normalize_prices(df)
    save_parquet(df, cache_name)
    print(f"[Data] {len(df)} rows, {df['symbol'].nunique()} symbols cached")
    return df


def load_or_fetch_macro(cfg: Config, live: bool) -> pd.DataFrame:
    """Load cached macro data or generate."""
    cache_name = "macro_live" if live else "macro_simulated"

    if has_data(cache_name):
        print("[Data] Loading cached macro data...")
        return load_parquet(cache_name)

    if live:
        try:
            from ascent.data.ingest.fred import fetch_all_macro
            df = fetch_all_macro(start_date=cfg.backtest.start_date, end_date=cfg.backtest.end_date)
        except Exception as e:
            print(f"[Data] FRED failed ({e}), using simulated macro")
            df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)
        if df.empty:
            print("[Data] FRED returned no data, using simulated macro")
            df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)
    else:
        print("[Data] Generating simulated macro data...")
        df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)

    df = normalize_macro(df)
    save_parquet(df, cache_name)
    return df


def run_pipeline(
    live: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance_days: int | None = None,
) -> None:
    """Run the full Ascent Capital pipeline."""
    cfg = get_config()
    # Auto-fill end_date with today if empty
    if not cfg.backtest.end_date:
        from datetime import date
        cfg.backtest.end_date = date.today().strftime("%Y-%m-%d")
        print(f"[Config] End date set to today: {cfg.backtest.end_date}")

    if start_date:
        cfg.backtest.start_date = start_date
    if end_date:
        cfg.backtest.end_date = end_date
    if top_n:
        cfg.backtest.top_n = top_n
    if rebalance_days:
        cfg.backtest.rebalance_freq_days = rebalance_days

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    # 1. DATA
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 1: DATA INGESTION")
    print("=" * 70)

    price_df = load_or_fetch_prices(cfg, live)
    macro_df = load_or_fetch_macro(cfg, live)

    # Separate benchmark
    benchmark_sym = cfg.universe.benchmark
    benchmark_mask = price_df["symbol"] == benchmark_sym
    benchmark_df = price_df[benchmark_mask].copy()
    universe_df = price_df[~benchmark_mask].copy()

    symbols = sorted(universe_df["symbol"].unique())
    print(f"[Data] Universe: {len(symbols)} symbols")
    print(f"[Data] Date range: {universe_df['date'].min().date()} → {universe_df['date'].max().date()}")

    # ══════════════════════════════════════════════════════════════
    # 2. FEATURES
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE COMPUTATION")
    print("=" * 70)

    builder = FeatureBuilder(universe_df, macro_df)
    features = builder.compute_features()
    targets = builder.compute_targets(horizons=[1, 5, 21])

    print(f"[Features] Computed {len(features)} features across {len(builder.symbols)} symbols")
    print(f"[Features] Date range: {builder.dates.min().date()} → {builder.dates.max().date()}")

    # ══════════════════════════════════════════════════════════════
    # 3. ALPHA
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 3: ALPHA GENERATION")
    print("=" * 70)

    alpha = -build_alpha_stack(features)  # flip: raw signal is inverted
    ranks = alpha_to_ranks(alpha)
    print(f"[Alpha] Composite alpha computed: {alpha.shape}")

    # Sanity: check alpha correlation with forward returns
    if "fwd_ret_21d" in targets:
        fwd = targets["fwd_ret_21d"]
        common = alpha.index.intersection(fwd.index)
        if len(common) > 100:
            # Cross-sectional IC (Information Coefficient)
            ic_series = alpha.loc[common].corrwith(fwd.loc[common], axis=1)
            mean_ic = ic_series.mean()
            print(f"[Alpha] Mean IC (21d fwd return): {mean_ic:.4f}")
            print(f"[Alpha] IC t-stat: {mean_ic / (ic_series.std() / np.sqrt(len(ic_series))):.2f}")

    # ══════════════════════════════════════════════════════════════
    # 4. PORTFOLIO CONSTRUCTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 4: PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    # Load sector data for diversification
    from ascent.data.store.parquet import has_data as _hd
    sector_map = {}
    if _hd("profiles"):
        from ascent.data.store.parquet import load_parquet as _lp
        _prof = _lp("profiles")
        sector_map = dict(zip(_prof["symbol"], _prof["sector"]))
        print("[Portfolio] Sector constraint: max 1 per sector")
    target_weights = sector_constrained_weighted(
        alpha,
        n=cfg.backtest.top_n,
        max_weight=cfg.backtest.max_weight,
        max_per_sector=1,
        sector_map=sector_map,
    )

    # Drop early dates where features aren't ready (warmup period)
    warmup = 252 + 21  # longest lookback + buffer
    if len(target_weights) > warmup:
        target_weights = target_weights.iloc[warmup:]

    active_dates = (target_weights.sum(axis=1) > 0.01).sum()
    print(f"[Portfolio] Target weights: {target_weights.shape}")
    print(f"[Portfolio] Active rebalance dates: {active_dates}")
    print(f"[Portfolio] Top-{cfg.backtest.top_n} rank-weighted, max weight {cfg.backtest.max_weight:.0%}")

    # ══════════════════════════════════════════════════════════════
    # 5. BACKTEST
    # ══════════════════════════════════════════════════════════════
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

    close = builder.close
    open_ = builder.open

    # Benchmark
    bm_close = benchmark_df.set_index("date")["close"].sort_index()
    bm_close = bm_close[~bm_close.index.duplicated(keep="last")]

    result = engine.run(
        target_weights=target_weights,
        close_prices=close,
        open_prices=open_,
        benchmark_prices=bm_close,
    )

    # ══════════════════════════════════════════════════════════════
    # 6. REPORT
    # ══════════════════════════════════════════════════════════════
    print_report(result, title="ASCENT CAPITAL — FULL BACKTEST")

    elapsed = time.time() - t0
    print(f"  Pipeline completed in {elapsed:.1f}s\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Ascent Capital — Quant Trading Platform")
    parser.add_argument("--live", action="store_true", help="Use live API data")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=None, help="Number of positions")
    parser.add_argument("--rebalance", type=int, default=None, help="Rebalance frequency (days)")
    parser.add_argument("--mode", type=str, default="walkforward", choices=["walkforward", "full"], help="Run mode")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached data")

    args = parser.parse_args()

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

    if args.mode == "walkforward":
        print("\n" + "#" * 70)
        print("  RUNNING WALK-FORWARD OUT-OF-SAMPLE EVALUATION")
        print("#" * 70)
        from ascent.research.walk_forward_runner import walk_forward_pipeline
        walk_forward_pipeline()


if __name__ == "__main__":
    main()
