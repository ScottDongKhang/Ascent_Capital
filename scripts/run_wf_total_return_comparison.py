#!/usr/bin/env python
"""
Run the canonical walk-forward pipeline against a total-return
(split/dividend-adjusted) price cache instead of the unadjusted `prices_live`
cache, and report the coverage window used.

Why this exists
----------------
`ascent/data/normalize/prices.py` has no split/dividend adjustment logic, so
the current CANONICAL_WF_ARTIFACT (Sharpe 0.415, see
`outputs/wf_results/wf_report_clean_2026-08-15.json`) was built on unadjusted
closes. That distorts momentum/mean-reversion signals for dividend-paying
names around ex-div dates and any (rare, for this universe) split events.

A total-return-adjusted staging cache already exists:
`data_cache/prices_live_clean_refetch.parquet` (936 symbols), built
2026-06-22 specifically for this reason but deliberately never swapped into
live `prices_live` (see CURRENT_VERIFIED_NUMBERS.md's "Production cache
note"). This script feeds that staging cache into
`ascent/research/walk_forward_runner.py::walk_forward_pipeline()` via its
`prices_cache_name=` override (search "FIX #5" in that file) -- a WF-only
substitution that never touches `prices_live` or live trading.

Coverage gap
------------
The staging cache covers 2020-01-02 -> 2026-06-22 (936 symbols). The
canonical WF window is 2020-01-02 -> 2026-07-15 (per
wf_report_clean_2026-08-15.json's _meta.oos_window). This run is therefore
missing the final ~3.5 weeks (2026-06-23 -> 2026-07-15) of the canonical
window -- walk_forward_pipeline derives its OOS window directly from the
price data's own max date, so no code change is needed to produce a
shorter, honestly-truncated run; it happens automatically because the
staging cache simply has no rows past 2026-06-22.

This script does not attempt to regenerate/extend the staging cache (that
would require a fresh live-data fetch, out of scope and riskier -- see task
instructions). It runs what exists and reports the actual window covered.

Usage
-----
    .venv/bin/python scripts/run_wf_total_return_comparison.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGING_CACHE = "prices_live_clean_refetch"


def main() -> int:
    from ascent.data.store.parquet import load_parquet, has_data

    if not has_data(STAGING_CACHE):
        print(f"ERROR: staging cache '{STAGING_CACHE}' not found in data_cache/. "
              f"This script does not regenerate it -- copy it in from another "
              f"checkout first.")
        return 1

    price_df = load_parquet(STAGING_CACHE)
    dmin, dmax = price_df["date"].min(), price_df["date"].max()
    n_symbols = price_df["symbol"].nunique()
    print(f"[compare] Staging cache '{STAGING_CACHE}': {n_symbols} symbols, "
          f"{dmin.date()} -> {dmax.date()} ({len(price_df)} rows)")
    print("[compare] Canonical WF window (unadjusted baseline) is "
          "2020-01-02 -> 2026-07-15 per wf_report_clean_2026-08-15.json. "
          "This run will therefore cover a shorter, truncated window ending "
          f"at {dmax.date()} -- the pipeline derives its OOS window from the "
          "price data's own max date, so this is not a silent truncation, "
          "it is the actual data available.")

    from ascent.research.walk_forward_runner import walk_forward_pipeline

    t0 = time.time()
    result = walk_forward_pipeline(prices_cache_name=STAGING_CACHE)
    elapsed = time.time() - t0
    print(f"\n[compare] walk_forward_pipeline() finished in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
