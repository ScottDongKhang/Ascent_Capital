#!/usr/bin/env python
"""Run the full proof audit: Path A (sleeves + agents) + Path B (subsystems) -> scorecard.

    .venv/bin/python -m ascent.analyst.proof_audit.run

Requires: data_cache prices_live parquet (for features/prices) and
logs/counterfactual_daily.jsonl (for the subsystem tracks). Missing inputs fail that
component's row as INSUFFICIENT_DATA, not the whole run -- see the plan's Task 7 for the
per-component try/except boundary.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.counterfactual_scorer import score_subsystem
from ascent.analyst.proof_audit.scorecard import DEFAULT_MIN_SAMPLE, ScorecardRow, verdict, write_scorecard
from ascent.analyst.proof_audit.stats import ICResult
from ascent.analyst.proof_audit.wf_scorer import score_agent, score_sleeve

log = logging.getLogger(__name__)

_DEFERRED_REASON = {
    "deferred": "requires live-logged signal history, not re-simulation -- out of scope for this audit",
    "covered_by_sleeves": "covered by per-sleeve rows; not scored standalone",
}


def _row_for_deferred(name: str, kind: str, method: str) -> ScorecardRow:
    log.info("proof_audit: %s (%s) skipped -- %s", name, method, _DEFERRED_REASON[method])
    return ScorecardRow(
        component=name, kind=kind, method=method,
        metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
    )


def _row_from_result(name: str, kind: str, method: str, result: ICResult) -> ScorecardRow:
    return ScorecardRow(
        component=name, kind=kind, method=method,
        metric=result.ic_mean, p_value=result.p_value, sample_size=result.n,
        verdict=verdict(result, min_sample=DEFAULT_MIN_SAMPLE),
    )


def run(features: dict, prices, out_path: Path | None = None) -> list[ScorecardRow]:
    if out_path is None:
        out_path = Path("outputs/analyst") / f"proof_audit_{date.today().isoformat()}.json"

    rows: list[ScorecardRow] = []
    for c in COMPONENTS:
        try:
            if c.method in _DEFERRED_REASON:
                rows.append(_row_for_deferred(c.name, c.kind, c.method))
            elif c.kind == "alpha_sleeve":
                rows.append(_row_from_result(c.name, c.kind, c.method, score_sleeve(c.name, features, prices)))
            elif c.kind == "agent":
                rows.append(_row_from_result(c.name, c.kind, c.method, score_agent(c.name, prices)))
            elif c.kind == "subsystem":
                rows.append(_row_from_result(c.name, c.kind, c.method, score_subsystem(c.name)))
        except Exception as exc:
            log.warning("proof_audit: %s failed (%s) -- marking INSUFFICIENT_DATA", c.name, exc)
            rows.append(ScorecardRow(
                component=c.name, kind=c.kind, method=c.method,
                metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
            ))

    write_scorecard(rows, out_path)
    log.info("proof_audit: wrote %d rows to %s", len(rows), out_path)
    return rows


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
    frame, not on the already-pivoted matrix: a naive `keep="last"` on the
    pivoted index would discard whichever timestamp has fewer symbols,
    silently dropping most of that day's real closes instead of merging the
    two timestamps' complementary coverage.
    """
    price_df = price_df.sort_values("date").copy()
    price_df["date"] = price_df["date"].dt.normalize()
    price_df = price_df.drop_duplicates(subset=["symbol", "date"], keep="last")
    return price_df


if __name__ == "__main__":
    import argparse

    from ascent.data.store.parquet import load_parquet
    from ascent.data.normalize.prices import pivot_prices
    from ascent.features.build_features import FeatureBuilder

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    price_df = load_parquet("prices_live")
    price_df = _dedupe_prices_by_calendar_day(price_df)
    prices = pivot_prices(price_df, field="close")
    features = FeatureBuilder(price_df).compute_features()
    run(features, prices, out_path=args.out)
