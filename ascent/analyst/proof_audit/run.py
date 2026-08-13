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


if __name__ == "__main__":
    import argparse

    from ascent.data.store.parquet import load_parquet
    from ascent.data.normalize.prices import pivot_prices
    from ascent.features.build_features import FeatureBuilder

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    price_df = load_parquet("prices_live")
    prices = pivot_prices(price_df, field="close")
    features = FeatureBuilder(price_df).compute_features()
    run(features, prices, out_path=args.out)
