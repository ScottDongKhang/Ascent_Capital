#!/usr/bin/env python
"""
Walk-forward comparison of the two new risk overlays.

Runs three configurations in ONE process so prices are loaded once and
nothing but the overlay config differs between runs (both plans require
"change nothing else between the two runs"):

  1. baseline    — vol_target_reference="spy",      crash overlay OFF  (current prod)
  2. stratvol    — vol_target_reference="strategy", crash overlay OFF
  3. crashoverlay— vol_target_reference="spy",      crash overlay ON (x0.50)

Artifacts use the filenames the plans reference:
  outputs/wf_results/wf_report_cashfix_2026-07-27.json       (baseline)
  outputs/wf_results/wf_report_stratvol_2026-07-27.json      (plan 3 treatment)
  outputs/wf_results/wf_report_crashoverlay_2026-07-27.json  (plan 4 treatment)

Sequential on purpose: parallel runs would write the shared parquet/ML
caches concurrently, and this repo has already had two data-corruption
incidents from concurrent/blended cache writes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve()
# scratchpad path -> use the worktree explicitly
WORKTREE = Path("/Users/scott/IdeaProjects/ascent-capital/.claude/worktrees/risk-mgmt")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "scripts"))

from ascent.data.store.parquet import load_parquet, has_data          # noqa: E402
from ascent.config.settings import get_config                          # noqa: E402
import ascent.portfolio.exposure as exposure                           # noqa: E402

import run_ascent_wf as R                                              # noqa: E402


# --------------------------------------------------------------------- #
# Recording proxies: capture the scale series the decision rules need.
# The WF path does not surface overlay `meta`, and the scales cannot be
# reconstructed post-hoc (they depend on the realized book). Wrapping the
# module attributes is non-invasive — production code is untouched.
# --------------------------------------------------------------------- #
STATS = {"realized": [], "spy_vol": [], "crash": []}

_orig_realized = exposure.realized_vol_scale
_orig_voltgt   = exposure.vol_target_scale
_orig_crash    = exposure.momentum_crash_scale


def _rec_realized(*a, **k):
    out = _orig_realized(*a, **k)
    if len(out):
        STATS["realized"].append(np.asarray(out, dtype=float))
    return out


def _rec_voltgt(*a, **k):
    out = _orig_voltgt(*a, **k)
    if len(out):
        STATS["spy_vol"].append(np.asarray(out, dtype=float))
    return out


def _rec_crash(*a, **k):
    out = _orig_crash(*a, **k)
    if len(out):
        STATS["crash"].append(np.asarray(out, dtype=float))
    return out


exposure.realized_vol_scale   = _rec_realized
exposure.vol_target_scale     = _rec_voltgt
exposure.momentum_crash_scale = _rec_crash


def _reset_stats():
    for v in STATS.values():
        v.clear()


def _summarize_stats(ref: str) -> dict:
    """Mean/min of the vol scale actually applied, plus crash-cut counts."""
    # With vol_reference="strategy", _apply_vol_target calls realized_vol_scale
    # directly; with "spy" it calls vol_target_scale (which delegates, so
    # realized also fires — use the outer wrapper for the spy case).
    key = "realized" if ref == "strategy" else "spy_vol"
    arrs = STATS[key]
    flat = np.concatenate(arrs) if arrs else np.array([])
    crash_flat = (np.concatenate(STATS["crash"])
                  if STATS["crash"] else np.array([]))
    return {
        "vol_scale_n":     int(flat.size),
        "vol_scale_mean":  round(float(flat.mean()), 4) if flat.size else None,
        "vol_scale_min":   round(float(flat.min()), 4) if flat.size else None,
        "vol_scale_at_floor_pct": (
            round(float((flat <= 0.2501).mean()) * 100, 2) if flat.size else None),
        "crash_scale_n":   int(crash_flat.size),
        "crash_cut_n":     int((crash_flat < 1.0).sum()) if crash_flat.size else 0,
    }


def main():
    import os
    smoke = os.getenv("SMOKE") == "1"
    # Production `prices_live` cannot be used: as of 2026-07-28 it carries
    # 322,868 duplicate (symbol, calendar_day) rows from tz-offset variants
    # (00:00/04:00/05:00 America/New_York) written by the yfinance_hub daily
    # path. The WF framework does NOT dedupe on read (only ascent/main.py
    # does), so the inflated date axis yields 47 overlapping folds and the
    # engine dies in _finalise with "cannot reindex on an axis with duplicate
    # labels". The clean staging cache is what the canonical VERIFIED WF
    # number (Sharpe 0.41, 21 folds) was produced from, so using it also makes
    # the baseline a sanity check on this harness.
    cache = os.getenv("WF_CACHE", "prices_live_clean_refetch")
    assert has_data(cache), f"no {cache} cache"
    print(f"Loading {cache}.parquet... (smoke={smoke})", flush=True)
    prices = load_parquet(cache)
    prices["date"] = pd.to_datetime(prices["date"])
    try:
        if prices["date"].dt.tz is not None:
            prices["date"] = prices["date"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    prices["date"] = prices["date"].dt.normalize()

    # Defensive dedupe on (symbol, calendar_day). A no-op on the clean cache
    # (verified 0 dups); prevents a silent repeat of the crash above if the
    # cache is ever swapped via WF_CACHE.
    _before = len(prices)
    prices = (prices.sort_values("date")
                    .drop_duplicates(subset=["symbol", "date"], keep="last")
                    .reset_index(drop=True))
    if len(prices) != _before:
        print(f"  WARNING: dropped {_before - len(prices):,} duplicate "
              f"(symbol, calendar_day) rows", flush=True)
    else:
        print("  dedupe guard: 0 duplicates (clean)", flush=True)

    if smoke:
        prices = prices[prices["date"] <= "2021-12-31"]
        print("  SMOKE: truncated to <= 2021-12-31", flush=True)
    print(f"  {prices['symbol'].nunique()} symbols, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}, "
          f"{len(prices):,} rows", flush=True)

    spy_df = prices[prices["symbol"] == "SPY"][["date", "close"]].copy()
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    if spy_df["date"].dt.tz is not None:
        spy_df["date"] = spy_df["date"].dt.tz_localize(None)
    spy = spy_df.sort_values("date").set_index("date")["close"].pct_change().fillna(0)
    spy.name = "SPY"

    cfg = get_config()
    runs = [
        ("baseline",     "spy",      False, "_cashfix_2026-07-27"),
        ("stratvol",     "strategy", False, "_stratvol_2026-07-27"),
        ("crashoverlay", "spy",      True,  "_crashoverlay_2026-07-27"),
    ]
    if smoke:
        runs = [(l, r, c, s + "_SMOKE") for l, r, c, s in runs]

    out = {}
    for label, ref, crash_on, suffix in runs:
        cfg.backtest.vol_target_reference = ref
        cfg.backtest.momentum_crash_overlay_enabled = crash_on
        cfg.backtest.momentum_crash_multiplier = 0.50
        _reset_stats()

        print(f"\n{'='*60}\n  {label.upper()}  "
              f"(vol_ref={ref}, crash={'ON' if crash_on else 'OFF'})\n{'='*60}",
              flush=True)

        engine = R._build_engine(None, None)
        report = R._run_one(engine, prices, spy, label, suffix)
        rec = {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
               for k, v in report.items()
               if not isinstance(v, (list, dict, pd.Series, pd.DataFrame))}
        rec["_overlay_stats"] = _summarize_stats(ref)
        rec["_config"] = {"vol_target_reference": ref,
                          "momentum_crash_overlay_enabled": crash_on,
                          "source_cache": cache}
        out[label] = rec
        print(f"[{label}] overlay stats: {rec['_overlay_stats']}", flush=True)

    # restore defaults in the live singleton
    cfg.backtest.vol_target_reference = "spy"
    cfg.backtest.momentum_crash_overlay_enabled = False

    dest = Path("/private/tmp/claude-501/-Users-scott-IdeaProjects-ascent-capital"
                "/90b29668-ed23-4e79-b592-f034c37e198b/scratchpad"
                "/wf_overlay_comparison.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {dest}", flush=True)

    # compact comparison table
    keys = ["sharpe", "cagr", "volatility", "max_drawdown", "beta",
            "alpha", "win_rate", "wfe", "n_folds", "n_oos_days"]
    print("\n" + "=" * 78)
    print(f"{'metric':<20}" + "".join(f"{lbl:>19}" for lbl in out))
    print("=" * 78)
    for k in keys:
        row = f"{k:<20}"
        for lbl in out:
            v = out[lbl].get(k)
            row += f"{v:>19.4f}" if isinstance(v, float) else f"{str(v):>19}"
        print(row)
    for k in ("vol_scale_mean", "vol_scale_min", "vol_scale_at_floor_pct",
              "crash_cut_n"):
        row = f"{k:<20}"
        for lbl in out:
            v = out[lbl]["_overlay_stats"].get(k)
            row += f"{v:>19.4f}" if isinstance(v, float) else f"{str(v):>19}"
        print(row)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
