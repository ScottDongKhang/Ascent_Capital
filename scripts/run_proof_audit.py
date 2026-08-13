#!/usr/bin/env python
"""Run the proof audit against real repo data and print a human-readable summary.

    .venv/bin/python scripts/run_proof_audit.py
"""
from __future__ import annotations

from ascent.data.store.parquet import has_data, load_parquet
from ascent.data.normalize.prices import pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.analyst.proof_audit.run import (
    _AGENT_PRICE_CACHES,
    _dedupe_prices_by_calendar_day,
    _load_agent_price_matrix,
    run,
)


def main() -> int:
    price_df = load_parquet("prices_live")
    # Strip timezone from price dates -- yfinance returns tz-aware (America/New_York), but the
    # fundamental/earnings/analyst/options/insider/short panels built inside FeatureBuilder
    # strip tz internally (see ascent/features/feature_defs.py), so a tz-aware `close` index
    # compared against those tz-naive panel indices raises
    # "Cannot compare dtypes datetime64[us] and datetime64[us, America/New_York]" inside the
    # alpha sleeve functions' own `.reindex(close.index, ...)` calls. Mirrors the identical fix
    # already applied in ascent/main.py's run_pipeline (grep "Strip timezone from price dates").
    if price_df["date"].dtype.tz is not None:
        price_df = price_df.copy()
        price_df["date"] = price_df["date"].dt.tz_localize(None)
    price_df = _dedupe_prices_by_calendar_day(price_df)
    prices = pivot_prices(price_df, field="close")

    fundamentals_df = load_parquet("fundamentals") if has_data("fundamentals") else None
    earnings_df = load_parquet("earnings") if has_data("earnings") else None
    analyst_df = load_parquet("analyst_revisions") if has_data("analyst_revisions") else None
    options_df = load_parquet("options_flow") if has_data("options_flow") else None
    insider_df = load_parquet("insider_transactions") if has_data("insider_transactions") else None
    short_df = load_parquet("short_interest") if has_data("short_interest") else None

    features = FeatureBuilder(
        price_df,
        fundamentals_df=fundamentals_df,
        earnings_df=earnings_df,
        analyst_df=analyst_df,
        options_df=options_df,
        insider_df=insider_df,
        short_df=short_df,
    ).compute_features()

    agent_prices = {}
    for agent_name, cache_name in _AGENT_PRICE_CACHES.items():
        if not has_data(cache_name):
            print(f"[proof_audit] {cache_name} missing -- {agent_name} falls back to shared prices")
            continue
        matrix = _load_agent_price_matrix(agent_name, cache_name)
        if matrix is not None:
            agent_prices[agent_name] = matrix

    rows = run(features, prices, agent_prices=agent_prices)

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
