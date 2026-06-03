#!/usr/bin/env python
"""
Ascent Walk-Forward OOS Evaluation
====================================
Full 243-combo grid search on real prices_live data.
Runtime: ~30 minutes (22 folds x 243 combos with smart caching).

Usage:
    .venv/bin/python scripts/run_ascent_wf.py [--smoke]

    --smoke : fast test on 2019-2021 subset (~3 folds, ~5 min)

Output (in outputs/wf_results/):
    wf_report_YYYY-MM-DD.json    — full performance report
    wf_equity_YYYY-MM-DD.csv     — stitched OOS equity curve vs SPY
    wf_folds_YYYY-MM-DD.json     — fold metadata (dates, best params)
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

from ascent.data.store.parquet import load_parquet, has_data
from ascent.research.wf_framework import (
    WalkForwardEngine,
    WindowGenerator,
    ExecutionConfig,
    AscentPortfolioStrategy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test on 2019-2021 subset (~3 folds)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    if not has_data("prices_live"):
        print("ERROR: prices_live.parquet not found. Run main pipeline first.")
        sys.exit(1)

    print("Loading prices_live.parquet...")
    prices = load_parquet("prices_live")
    prices["date"] = pd.to_datetime(prices["date"])

    if args.smoke:
        prices = prices[prices["date"] <= "2021-12-31"]
        print("  SMOKE MODE: truncated to 2019-2021")

    n_symbols  = prices["symbol"].nunique()
    date_range = f"{prices['date'].min().date()} → {prices['date'].max().date()}"
    n_rows     = len(prices)
    print(f"  {n_symbols} symbols, {date_range}, {n_rows:,} rows")

    # SPY benchmark returns
    spy_df = prices[prices["symbol"] == "SPY"][["date", "close"]].copy()
    spy    = spy_df.sort_values("date").set_index("date")["close"].pct_change().fillna(0)
    spy.name = "SPY"

    # ------------------------------------------------------------------ #
    # 2. Build engine                                                      #
    # ------------------------------------------------------------------ #
    engine = WalkForwardEngine(
        strategy_cls = AscentPortfolioStrategy,
        window_generator = WindowGenerator(
            is_days      = 252,
            oos_days     = 63,
            purge_days   = 21,
            embargo_days = 5,
            window_type  = "rolling",
            step_days    = 63,
        ),
        exec_config = ExecutionConfig(
            slippage_model     = "atr",
            atr_multiplier     = 0.10,
            commission_pct     = 0.0005,
            borrow_rate_annual = 0.0,
        ),
        # Constraint: leaves >= 25% for the other 12 sleeves
        constraint_fn = lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
        rf_annual     = 0.04,
        verbose       = True,
    )

    n_combos = 1
    for v in AscentPortfolioStrategy().param_grid.values():
        n_combos *= len(v)
    estimated_min = 30 if not args.smoke else 5
    print(f"\nGrid: {n_combos} combos × {engine.wg.is_days}d IS / "
          f"{engine.wg.oos_days}d OOS  (constraint: trend+statarb ≤ 75%)")
    print(f"Estimated runtime: ~{estimated_min} minutes\n")

    # ------------------------------------------------------------------ #
    # 3. Run                                                               #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    equity_curve, report = engine.run(prices, spy)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    # ------------------------------------------------------------------ #
    # 4. Save results                                                      #
    # ------------------------------------------------------------------ #
    out_dir = Path("outputs/wf_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    suffix = f"_smoke_{today}" if args.smoke else f"_{today}"

    def _serialise(v):
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    # Report
    report_path = out_dir / f"wf_report{suffix}.json"
    with open(report_path, "w") as f:
        json.dump({k: _serialise(v) for k, v in report.items()}, f, indent=2)
    print(f"Report      → {report_path}")

    # Equity curve (strategy + SPY aligned)
    equity_path = out_dir / f"wf_equity{suffix}.csv"
    equity_df = equity_curve.rename("strategy").to_frame()
    spy_eq = (1 + spy.reindex(equity_curve.index).fillna(0)).cumprod()
    spy_eq = spy_eq / spy_eq.iloc[0]
    equity_df["spy"] = spy_eq
    equity_df.to_csv(equity_path)
    print(f"Equity CSV  → {equity_path}")

    # Fold metadata
    folds_path = out_dir / f"wf_folds{suffix}.json"
    fold_meta  = [
        {
            "fold_id":   w.fold_id,
            "is_start":  str(w.is_start.date()),
            "oos_start": str(w.oos_start.date()),
            "oos_end":   str(w.oos_end.date()),
        }
        for w in engine.last_windows_
    ]
    with open(folds_path, "w") as f:
        json.dump(fold_meta, f, indent=2)
    print(f"Fold meta   → {folds_path}")

    # ------------------------------------------------------------------ #
    # 5. Summary                                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 52)
    print(f"  OOS period : {equity_curve.index[0].date()} → {equity_curve.index[-1].date()}")
    print(f"  Folds      : {report['n_folds']}")
    print(f"  OOS days   : {report['n_oos_days']}")
    print(f"  CAGR       : {report['cagr']*100:+.2f}%")
    print(f"  Sharpe     : {report['sharpe']:.3f}  (rf=4%)")
    print(f"  Sortino    : {report['sortino']:.3f}")
    print(f"  Max DD     : {report['max_drawdown']*100:.2f}%")
    print(f"  Win Rate   : {report['win_rate']*100:.1f}%")
    if np.isfinite(report.get("alpha", float("nan"))):
        print(f"  Alpha/SPY  : {report['alpha']*100:+.2f}%")
    wfe = report["wfe"]
    wfe_label = "acceptable" if np.isfinite(wfe) and wfe >= 0.5 else "OVERFIT WARNING"
    print(f"  WFE        : {wfe:.3f}  ({wfe_label})")
    print("=" * 52)
    print(f"\nRuntime: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
