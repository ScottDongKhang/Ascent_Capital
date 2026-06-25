"""
EDGE THESIS TEST #1 — Is Ascent's edge just momentum factor beta?

Two arms, SAME engine / window / costs / clean cache:
  FULL  = verified config (simplified: llm/narrative zeroed, meanrev 0.02, optimizer on)
  TREND = pure trend sleeve (fixed_params trend=1.0, statarb=0.0 -> all other sleeves 0),
          custom constraint allowing trend=1.0.

Falsifier: if TREND matches or beats FULL on Sharpe/alpha, the multi-sleeve
machinery adds nothing over momentum alone — the "edge" is factor beta.

Validation: the FULL arm should reproduce the verified Sharpe ~0.41 / CAGR ~10.3%.
If it doesn't, the harness is wrong and the comparison is void.
"""
import sys, json, time
from pathlib import Path

REPO = "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
sys.path.insert(0, REPO)

import pandas as pd
from ascent.data.store.parquet import load_parquet
from ascent.research.wf_framework import (
    WalkForwardEngine, WindowGenerator, ExecutionConfig, AscentPortfolioStrategy,
)

CLEAN = "prices_live_clean_refetch"
SMOKE = "--smoke" in sys.argv
SIMPLIFIED = {"llm_fundamental": 0.0, "narrative": 0.0, "meanrev": 0.02}


def load_clean():
    prices = load_parquet(CLEAN)
    prices["date"] = pd.to_datetime(prices["date"])
    if SMOKE:
        prices = prices[prices["date"] <= "2021-12-31"]
    spy_df = prices[prices["symbol"] == "SPY"][["date", "close"]].copy()
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    if spy_df["date"].dt.tz is not None:
        spy_df["date"] = spy_df["date"].dt.tz_localize(None)
    spy = spy_df.sort_values("date").set_index("date")["close"].pct_change().fillna(0)
    spy.name = "SPY"
    return prices, spy


def _window():
    return WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5,
                           window_type="rolling", step_days=63)


def _exec():
    return ExecutionConfig(slippage_model="atr", atr_multiplier=0.10,
                           commission_pct=0.0005, borrow_rate_annual=0.0)


def full_engine():
    class _Full(AscentPortfolioStrategy):
        def __init__(self, **kw):
            kw.setdefault("alpha_weights_override", SIMPLIFIED)
            super().__init__(**kw)
    return WalkForwardEngine(
        strategy_cls=_Full, window_generator=_window(), exec_config=_exec(),
        constraint_fn=lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
        rf_annual=0.04, verbose=False)


def trend_only_engine():
    class _Trend(AscentPortfolioStrategy):
        def __init__(self, **kw):
            kw.setdefault("fixed_params",
                          {"trend_weight": 1.0, "statarb_weight": 0.0, "mom_window": 252})
            super().__init__(**kw)
    return WalkForwardEngine(
        strategy_cls=_Trend, window_generator=_window(), exec_config=_exec(),
        constraint_fn=lambda p: True,  # allow pure trend
        rf_annual=0.04, verbose=False)


def run(label, engine, prices, spy):
    t0 = time.time()
    equity, report = engine.run(prices, spy)
    dt = time.time() - t0
    keep = {k: report.get(k) for k in
            ("sharpe", "cagr", "alpha", "beta", "max_drawdown", "volatility",
             "win_rate", "wfe", "n_folds", "n_oos_days")}
    keep["_seconds"] = round(dt, 1)
    print(f"[{label}] " + "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in keep.items()), flush=True)
    return keep


def main():
    print(f"SMOKE={SMOKE}  cache={CLEAN}", flush=True)
    prices, spy = load_clean()
    print(f"loaded {prices['symbol'].nunique()} symbols, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}", flush=True)

    out = {}
    out["FULL"]  = run("FULL",  full_engine(),       prices, spy)
    out["TREND"] = run("TREND", trend_only_engine(), prices, spy)

    # Verdict
    fs, ts = out["FULL"]["sharpe"], out["TREND"]["sharpe"]
    fa, ta = out["FULL"]["alpha"],  out["TREND"]["alpha"]
    out["_verdict"] = {
        "sharpe_full_minus_trend": fs - ts,
        "alpha_full_minus_trend":  fa - ta,
        "reading": ("FULL adds value over momentum alone"
                    if (fs - ts) > 0.05 else
                    "FULL does NOT beat momentum alone — edge is ~factor beta"),
    }
    print("VERDICT:", json.dumps(out["_verdict"], indent=2), flush=True)

    suffix = "_smoke" if SMOKE else ""
    Path(f"{REPO}/outputs/wf_results/edge_test1_momentum{suffix}.json").write_text(
        json.dumps(out, indent=2))
    print("wrote", f"outputs/wf_results/edge_test1_momentum{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
