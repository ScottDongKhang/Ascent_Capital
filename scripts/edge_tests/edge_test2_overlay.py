"""
EDGE THESIS TEST #2 — Does the 200MA + vol-target overlay help or cost?

OVERLAY-OFF arm: identical to the FULL arm of test #1 (same SIMPLIFIED alpha
config, same optimizer, same window/cache) EXCEPT the 200MA overlay and the
vol-target overlay are overridden to identity (no-ops). No live code is touched
— the overrides live in a subclass here.

Compare to the FULL (overlay-ON) result from edge_test1_momentum.json
(Sharpe 0.486, same harness).

Falsifier (EDGE_THESIS H2): if removing the overlays IMPROVES Sharpe over the
OOS window, the overlay is a cost, not an edge.
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


def overlay_off_engine():
    class _NoOverlay(AscentPortfolioStrategy):
        def __init__(self, **kw):
            kw.setdefault("alpha_weights_override", SIMPLIFIED)
            super().__init__(**kw)
        # Disable both overlays — pass weights straight through.
        def _apply_200ma_overlay(self, data, weights_at_rebal, *a, **k):
            return weights_at_rebal
        def _apply_vol_target(self, data, weights_at_rebal, *a, **k):
            return weights_at_rebal
    return WalkForwardEngine(
        strategy_cls=_NoOverlay, window_generator=WindowGenerator(
            is_days=252, oos_days=63, purge_days=21, embargo_days=5,
            window_type="rolling", step_days=63),
        exec_config=ExecutionConfig(slippage_model="atr", atr_multiplier=0.10,
                                    commission_pct=0.0005, borrow_rate_annual=0.0),
        constraint_fn=lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
        rf_annual=0.04, verbose=False)


def main():
    prices, spy = load_clean()
    print(f"loaded {prices['symbol'].nunique()} symbols, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}", flush=True)
    t0 = time.time()
    equity, report = overlay_off_engine().run(prices, spy)
    keep = {k: report.get(k) for k in
            ("sharpe", "cagr", "alpha", "beta", "max_drawdown", "volatility",
             "win_rate", "wfe", "n_folds", "n_oos_days")}
    keep["_seconds"] = round(time.time() - t0, 1)
    print("[OVERLAY_OFF] " + "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in keep.items()), flush=True)

    # Compare to FULL (overlay ON) from test #1
    try:
        full = json.load(open(f"{REPO}/outputs/wf_results/edge_test1_momentum.json"))["FULL"]
    except Exception:
        full = {}
    out = {"OVERLAY_ON_full": full, "OVERLAY_OFF": keep}
    if full:
        out["_verdict"] = {
            "sharpe_off_minus_on": keep["sharpe"] - full["sharpe"],
            "reading": ("Overlay is a COST — removing it improves Sharpe"
                        if keep["sharpe"] - full["sharpe"] > 0.02 else
                        "Overlay helps or is neutral"),
        }
        print("VERDICT:", json.dumps(out["_verdict"], indent=2), flush=True)
    suffix = "_smoke" if SMOKE else ""
    Path(f"{REPO}/outputs/wf_results/edge_test2_overlay{suffix}.json").write_text(json.dumps(out, indent=2))
    print("wrote", f"outputs/wf_results/edge_test2_overlay{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
