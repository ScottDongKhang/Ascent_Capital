#!/usr/bin/env python
"""Run the proof audit against real repo data and print a human-readable summary.

    .venv/bin/python scripts/run_proof_audit.py
"""
from __future__ import annotations

from ascent.data.store.parquet import load_parquet
from ascent.data.normalize.prices import pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.analyst.proof_audit.run import run


def _dedupe_prices_by_calendar_day(price_df):
    """Collapse `prices_live` to one row per (symbol, calendar day).

    `prices_live` recurrently carries two intraday timestamps for the same
    trading day (e.g. 2022-12-21 19:00:00-05:00 and 2022-12-22 00:00:00-05:00),
    with *different* symbols populated on each timestamp -- not identical
    duplicate rows, so a plain `~index.duplicated()` on the raw timestamp
    misses it. Once `pivot_prices()` pivots on the raw timestamp, this splits
    a single trading day across two index rows, each mostly-NaN, which starves
    `forward_return_matrix()` (`pct_change().shift(-1)`) of adjacent same-day
    data and collapses nearly every alpha-sleeve/agent row to
    INSUFFICIENT_DATA.

    Fix: normalize `date` to the calendar day *before* pivoting (both here and
    inside `FeatureBuilder`, which pivots the same long-format frame
    internally), then drop true per-symbol/day duplicates keeping the
    chronologically last row -- matching the project's existing `keep="last"`
    convention (see `ascent/main.py`). This must happen on the long-format
    frame, not on the already-pivoted matrix: a naive
    `keep="last"` on the pivoted index would discard whichever timestamp has
    fewer symbols, silently dropping most of that day's real closes instead
    of merging the two timestamps' complementary coverage.
    """
    price_df = price_df.sort_values("date").copy()
    price_df["date"] = price_df["date"].dt.normalize()
    price_df = price_df.drop_duplicates(subset=["symbol", "date"], keep="last")
    return price_df


def main() -> int:
    price_df = load_parquet("prices_live")
    price_df = _dedupe_prices_by_calendar_day(price_df)
    prices = pivot_prices(price_df, field="close")
    features = FeatureBuilder(price_df).compute_features()
    rows = run(features, prices)

    print(f"{'component':30s} {'kind':14s} {'method':16s} {'metric':>10s} {'p':>8s} {'n':>5s}  verdict")
    for r in sorted(rows, key=lambda r: (r.kind, r.component)):
        metric = f"{r.metric:.4f}" if r.metric is not None else "n/a"
        p = f"{r.p_value:.4f}" if r.p_value is not None else "n/a"
        print(f"{r.component:30s} {r.kind:14s} {r.method:16s} {metric:>10s} {p:>8s} {r.sample_size:5d}  {r.verdict}")

    n_keep = sum(1 for r in rows if r.verdict == "KEEP")
    n_cut = sum(1 for r in rows if r.verdict == "CUT")
    n_insufficient = sum(1 for r in rows if r.verdict == "INSUFFICIENT_DATA")
    print(f"\nKEEP={n_keep} CUT={n_cut} INSUFFICIENT_DATA={n_insufficient} (total {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
