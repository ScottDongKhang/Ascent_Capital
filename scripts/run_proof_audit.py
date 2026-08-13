#!/usr/bin/env python
"""Run the proof audit against real repo data and print a human-readable summary.

    .venv/bin/python scripts/run_proof_audit.py
"""
from __future__ import annotations

from ascent.data.store.parquet import load_parquet
from ascent.data.normalize.prices import pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.analyst.proof_audit.run import _dedupe_prices_by_calendar_day, run


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

    flagged = [r for r in sorted(rows, key=lambda r: (r.kind, r.component)) if r.reason]
    if flagged:
        print("\nwhy these rows are not clean measurements:")
        for r in flagged:
            print(f"  {r.component:30s} {r.reason}")

    n_keep = sum(1 for r in rows if r.verdict == "KEEP")
    n_cut = sum(1 for r in rows if r.verdict == "CUT")
    n_insufficient = sum(1 for r in rows if r.verdict == "INSUFFICIENT_DATA")
    print(f"\nKEEP={n_keep} CUT={n_cut} INSUFFICIENT_DATA={n_insufficient} (total {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
