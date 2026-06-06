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
    MultiAssetPortfolioStrategy,
    FullOrchestrationStrategy,
)


# Simplified sleeve config: removes LLM fundamental + narrative (untracked/expensive),
# reduces mean reversion from 5% → 2% (IC-t=0.35, noise). Trend absorbs freed weight.
SIMPLIFIED_ALPHA_OVERRIDES = {
    "llm_fundamental": 0.00,
    "narrative":       0.00,
    "meanrev":         0.02,
}


# Fixed params — bypass IS optimizer (WFE negative means it adds no value)
# Chosen as the median of IS-selected params across all 21 folds
FIXED_PARAMS = {"trend_weight": 0.38, "statarb_weight": 0.15, "mom_window": 252}

# High-Sharpe config: wider universe, corrected vol target, no IS overfit
HIGH_SHARPE_CONFIG = {
    "top_n": 25,           # 15→25: reduces idiosyncratic vol ~20%
    "max_weight": 0.08,    # 10%→8%: prevents single-stock dominance
    "fixed_params": FIXED_PARAMS,
}


def _build_engine(
    alpha_overrides: dict | None = None,
    strategy_kwargs: dict | None = None,
) -> "WalkForwardEngine":
    """Return a configured WalkForwardEngine, optionally with sleeve/strategy overrides."""
    strategy_cls = AscentPortfolioStrategy
    _overrides    = strategy_kwargs or {}
    if alpha_overrides:
        _alpha_ov = alpha_overrides
        class _OverriddenStrategy(AscentPortfolioStrategy):
            def __init__(self, **kwargs):
                kwargs.setdefault("alpha_weights_override", _alpha_ov)
                for k, v in _overrides.items():
                    kwargs.setdefault(k, v)
                super().__init__(**kwargs)
        strategy_cls = _OverriddenStrategy
    elif _overrides:
        _strat_ov = _overrides
        class _OverriddenStrategy(AscentPortfolioStrategy):
            def __init__(self, **kwargs):
                for k, v in _strat_ov.items():
                    kwargs.setdefault(k, v)
                super().__init__(**kwargs)
        strategy_cls = _OverriddenStrategy

    return WalkForwardEngine(
        strategy_cls     = strategy_cls,
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
        constraint_fn = lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
        rf_annual     = 0.04,
        verbose       = True,
    )


def _make_engine(strategy_cls, borrow_rate=0.0) -> "WalkForwardEngine":
    return WalkForwardEngine(
        strategy_cls     = strategy_cls,
        window_generator = WindowGenerator(
            is_days=252, oos_days=63, purge_days=21, embargo_days=5,
            window_type="rolling", step_days=63,
        ),
        exec_config = ExecutionConfig(
            slippage_model="atr", atr_multiplier=0.10,
            commission_pct=0.0005, borrow_rate_annual=borrow_rate,
        ),
        constraint_fn = lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
        rf_annual=0.04,
        verbose=True,
    )


def _build_engine_multi_asset() -> "WalkForwardEngine":
    """Multi-asset engine: equity momentum + macro trend + FRED regime detection."""
    return _make_engine(MultiAssetPortfolioStrategy)


def _build_engine_long_short() -> "WalkForwardEngine":
    """Long/short engine: long top-15 + short bottom-5 momentum (100L/50S)."""
    class _LongShortStrategy(AscentPortfolioStrategy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.short_n = 5  # short bottom-5 momentum stocks
    return _make_engine(_LongShortStrategy, borrow_rate=0.005)  # 0.5% borrow cost


def _run_one(engine, prices, spy, label: str, suffix: str) -> dict:
    """Run one WF pass, save results, return report dict."""
    t0 = time.time()
    equity_curve, report = engine.run(prices, spy)
    elapsed = time.time() - t0
    print(f"\n[{label}] Completed in {elapsed/60:.1f} min")

    out_dir = Path("outputs/wf_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _s(v):
        if isinstance(v, (float, np.floating)): return float(v)
        if isinstance(v, (int, np.integer)):    return int(v)
        return v

    report_path = out_dir / f"wf_report{suffix}.json"
    with open(report_path, "w") as f:
        json.dump({k: _s(v) for k, v in report.items()}, f, indent=2)

    equity_path = out_dir / f"wf_equity{suffix}.csv"
    equity_df   = equity_curve.rename("strategy").to_frame()
    spy_eq      = (1 + spy.reindex(equity_curve.index).fillna(0)).cumprod()
    spy_eq      = spy_eq / spy_eq.iloc[0]
    equity_df["spy"] = spy_eq
    equity_df.to_csv(equity_path)

    # Per-fold metrics — now populated via engine.last_fold_results_
    analyser   = engine.analyser
    fold_metas = engine.last_windows_
    fold_res   = getattr(engine, "last_fold_results_", [])
    fold_res_by_id = {fr.fold_id: fr for fr in fold_res}
    fold_meta_list = []
    for w in fold_metas:
        fr = fold_res_by_id.get(w.fold_id)
        entry = {
            "fold_id":   w.fold_id,
            "is_start":  str(w.is_start.date()),
            "oos_start": str(w.oos_start.date()),
            "oos_end":   str(w.oos_end.date()),
            "is_sharpe":  _s(fr.is_sharpe) if fr else None,
            "oos_sharpe": _s(analyser.sharpe(fr.oos_returns)) if fr else None,
            "oos_cagr":   _s(analyser.cagr(fr.oos_returns))   if fr else None,
            "wfe":        _s(analyser.sharpe(fr.oos_returns) / fr.is_sharpe)
                          if fr and fr.is_sharpe and abs(fr.is_sharpe) > 1e-6 else None,
        }
        fold_meta_list.append(entry)

    folds_path = out_dir / f"wf_folds{suffix}.json"
    with open(folds_path, "w") as f:
        json.dump(fold_meta_list, f, indent=2)

    print(f"  Report  → {report_path}")
    print(f"  Equity  → {equity_path}")
    print(f"  Folds   → {folds_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test on 2019-2021 subset (~3 folds)")
    parser.add_argument("--simplified", action="store_true",
                        help="Run simplified sleeve config (no LLM/narrative, reduced meanrev)")
    parser.add_argument("--compare", action="store_true",
                        help="Run current vs simplified back-to-back")
    parser.add_argument("--high-sharpe", action="store_true",
                        help="High-Sharpe config: top_n=25, max_weight=8%, fixed params")
    parser.add_argument("--compare-hs", action="store_true",
                        help="Compare current vs high-sharpe config")
    parser.add_argument("--multi-asset", action="store_true",
                        help="Multi-asset: equity momentum + macro trend + FRED regime detection")
    parser.add_argument("--long-short", action="store_true",
                        help="Long/short: long top-N, short bottom-5 momentum (100L/50S)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run current vs multi-asset vs long-short (3-way comparison)")
    parser.add_argument("--live-system", action="store_true",
                        help="Backtest the full live 4-agent orchestration (US/macro/intl/alt)")
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

    # SPY benchmark returns — strip tz to match engine's tz-naive date stripping
    spy_df = prices[prices["symbol"] == "SPY"][["date", "close"]].copy()
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    if spy_df["date"].dt.tz is not None:
        spy_df["date"] = spy_df["date"].dt.tz_localize(None)
    spy = spy_df.sort_values("date").set_index("date")["close"].pct_change().fillna(0)
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

    n_combos      = 1
    for v in AscentPortfolioStrategy().param_grid.values():
        n_combos *= len(v)
    estimated_min = 30 if not args.smoke else 5
    mode_label    = "smoke" if args.smoke else "full"
    print(f"\nGrid: {n_combos} combos × 252d IS / 63d OOS  ({mode_label})")

    today = date.today().isoformat()
    smoke_tag = "_smoke" if args.smoke else ""

    # ------------------------------------------------------------------ #
    # 3. Run — one config or compare two                                  #
    # ------------------------------------------------------------------ #
    configs = []
    if args.live_system:
        configs = [("live_system", None, None, "live_system", f"_live{smoke_tag}_{today}")]
    elif args.compare_all:
        # 3-way: current vs multi-asset vs long-short
        configs = [
            ("current",     None, None, None,               f"_current{smoke_tag}_{today}"),
            ("multi_asset", None, None, "multi_asset",       f"_ma{smoke_tag}_{today}"),
            ("long_short",  None, None, "long_short",        f"_ls{smoke_tag}_{today}"),
        ]
    elif args.multi_asset:
        configs = [("multi_asset", None, None, "multi_asset", f"_ma{smoke_tag}_{today}")]
    elif args.long_short:
        configs = [("long_short", None, None, "long_short", f"_ls{smoke_tag}_{today}")]
    elif args.compare:
        configs = [
            ("current",    None, None, None,               f"_current{smoke_tag}_{today}"),
            ("simplified", SIMPLIFIED_ALPHA_OVERRIDES, None, None, f"_simplified{smoke_tag}_{today}"),
        ]
    elif args.compare_hs:
        configs = [
            ("current",     None, None, None,              f"_current{smoke_tag}_{today}"),
            ("high_sharpe", None, HIGH_SHARPE_CONFIG, None, f"_hs{smoke_tag}_{today}"),
        ]
    elif args.simplified:
        configs = [("simplified", SIMPLIFIED_ALPHA_OVERRIDES, None, None, f"_simplified{smoke_tag}_{today}")]
    elif args.high_sharpe:
        configs = [("high_sharpe", None, HIGH_SHARPE_CONFIG, None, f"_hs{smoke_tag}_{today}")]
    else:
        configs = [("current", None, None, None, f"{smoke_tag}_{today}")]

    results = {}
    for config in configs:
        label = config[0]
        if len(config) == 5:
            label, alpha_overrides, strat_kwargs, strategy_type, suffix = config
        else:
            label, alpha_overrides, strat_kwargs, suffix = config
            strategy_type = None

        print(f"\n{'='*52}")
        print(f"  Running: {label.upper()}")
        print(f"{'='*52}")

        if strategy_type == "live_system":
            engine = _make_engine(FullOrchestrationStrategy)
        elif strategy_type == "multi_asset":
            engine = _build_engine_multi_asset()
        elif strategy_type == "long_short":
            engine = _build_engine_long_short()
        else:
            engine = _build_engine(alpha_overrides, strat_kwargs)

        report = _run_one(engine, prices, spy, label, suffix)
        results[label] = report

    # ------------------------------------------------------------------ #
    # 4. Print comparison (if --compare)                                  #
    # ------------------------------------------------------------------ #
    if (args.compare or args.compare_hs) and len(results) == 2:
        keys = list(results.keys())
        cur, simp = results[keys[0]], results[keys[1]]
        print(f"\n{'='*60}")
        print(f"  COMPARISON: {keys[0]} vs {keys[1]}")
        print(f"{'='*60}")
        print(f"  {'Metric':<18} {'Current':>10} {'Simplified':>12} {'Delta':>10}")
        print(f"  {'-'*52}")
        metrics = [
            ("Sharpe",   "sharpe",       lambda v: f"{v:.3f}"),
            ("CAGR",     "cagr",         lambda v: f"{v*100:+.2f}%"),
            ("Max DD",   "max_drawdown", lambda v: f"{v*100:.2f}%"),
            ("WFE",      "wfe",          lambda v: f"{v:.3f}"),
            ("Alpha/SPY","alpha",        lambda v: f"{v*100:+.2f}%"),
        ]
        for name, key, fmt in metrics:
            cv = cur.get(key, float("nan"))
            sv = simp.get(key, float("nan"))
            cv_s = fmt(cv) if np.isfinite(cv) else "N/A"
            sv_s = fmt(sv) if np.isfinite(sv) else "N/A"
            delta_s = ""
            if np.isfinite(cv) and np.isfinite(sv):
                d = sv - cv
                delta_s = f"{'+' if d >= 0 else ''}{d:.3f}"
            print(f"  {name:<18} {cv_s:>10} {sv_s:>12} {delta_s:>10}")
        print(f"{'='*60}")
        winner = "simplified" if simp.get("sharpe", 0) > cur.get("sharpe", 0) else "current"
        print(f"  Winner by Sharpe: {winner.upper()}")


if __name__ == "__main__":
    main()
