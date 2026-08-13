#!/usr/bin/env python
"""Run the full proof audit: Path A (sleeves + agents) + Path B (subsystems) -> scorecard.

    .venv/bin/python -m ascent.analyst.proof_audit.run

Requires: data_cache prices_live parquet (for features/prices) and
logs/counterfactual_daily.jsonl (for the subsystem tracks). Missing inputs fail that
component's row as INSUFFICIENT_DATA, not the whole run -- see the plan's Task 7 for the
per-component try/except boundary.
"""
from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.counterfactual_scorer import score_subsystem
from ascent.analyst.proof_audit.forward_returns import eligible_dates
from ascent.analyst.proof_audit.scorecard import DEFAULT_MIN_SAMPLE, ScorecardRow, verdict, write_scorecard
from ascent.analyst.proof_audit.stats import ICResult
from ascent.analyst.proof_audit.wf_scorer import DegenerateSignalError, score_agent, score_sleeve

log = logging.getLogger(__name__)

_DEFERRED_REASON = {
    "deferred": "requires live-logged signal history, not re-simulation -- out of scope for this audit",
    "covered_by_sleeves": "covered by per-sleeve rows; not scored standalone",
}

# Sleeves whose signal functions need altdata / earnings-tone input frames that this CLI does
# not load at all (no parquet cache wired up yet). The real-data CLI loads fundamentals,
# earnings, analyst, options, insider and short-interest frames (see __main__ below), so those
# six score for real now -- only altdata and earnings_tone remain a disclosed input gap rather
# than a measured verdict.
_MISSING_INPUT_SLEEVES = {"altdata", "earnings_tone"}
_MISSING_INPUT_REASON = "requires altdata/earnings-tone input data not loaded by this CLI"

_DEGENERATE_SUFFIX = " -- likely missing feature inputs or wrong universe"

_DUPLICATE_AGENT_REASON = (
    "identical to another agent's score ({others}) -- signal matrices are not genuinely "
    "independent, likely a shared/wrong universe bug"
)

# Path B approximations, disclosed per-row so a reader of the JSON alone can see that three
# subsystem rows are one measurement under three names. See counterfactual_scorer.py's
# docstring for the full argument.
_SUBSYSTEM_REASONS = {
    "regime_overlay": (
        "approximated onto the same track pair as earned_authority pending a dedicated "
        "counterfactual track (see counterfactual_scorer.py docstring)"
    ),
    "hedge_overlay": (
        "approximated onto the same track pair as earned_authority pending a dedicated "
        "counterfactual track (see counterfactual_scorer.py docstring)"
    ),
    "earned_authority": (
        "canonical row for the (track_d, track_astar) pair; regime_overlay and hedge_overlay "
        "are approximated onto this same pair and report the same number"
    ),
    "debate_judge_intervention": (
        "track_b is total-return, track_d is split-only -- delta includes the dividend stream "
        "alongside the judge-intervention effect"
    ),
}


def _row_for_deferred(name: str, kind: str, method: str) -> ScorecardRow:
    log.info("proof_audit: %s (%s) skipped -- %s", name, method, _DEFERRED_REASON[method])
    return ScorecardRow(
        component=name, kind=kind, method=method,
        metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
        reason=_DEFERRED_REASON[method],
    )


def _row_from_result(
    name: str, kind: str, method: str, result: ICResult, reason: str | None = None
) -> ScorecardRow:
    return ScorecardRow(
        component=name, kind=kind, method=method,
        metric=result.ic_mean, p_value=result.p_value, sample_size=result.n,
        verdict=verdict(result, min_sample=DEFAULT_MIN_SAMPLE),
        reason=reason,
    )


def _failure_row(component, reason: str) -> ScorecardRow:
    return ScorecardRow(
        component=component.name, kind=component.kind, method=component.method,
        metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
        reason=reason,
    )


def _flag_duplicate_agent_scores(rows: list[ScorecardRow]) -> list[ScorecardRow]:
    """Downgrade agent rows that report byte-identical numbers to each other.

    Only AGENT rows: two alpha sleeves reading the same price panel may legitimately land on
    the same measurement, but two *agents* that are supposed to trade different universes
    cannot -- an exact tie means they were fed the same matrix, which makes both numbers
    measurement artifacts rather than measurements of the named agent.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.kind == "agent" and r.metric is not None:
            groups[(r.metric, r.p_value, r.sample_size)].append(i)

    out = list(rows)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        names = [rows[i].component for i in indices]
        log.warning(
            "proof_audit: agents %s report an identical (metric, p_value, sample_size) -- "
            "downgrading all of them to INSUFFICIENT_DATA",
            ", ".join(sorted(names)),
        )
        for i in indices:
            others = sorted(n for n in names if n != rows[i].component)
            out[i] = dataclasses.replace(
                rows[i],
                verdict="INSUFFICIENT_DATA",
                reason=_DUPLICATE_AGENT_REASON.format(others=", ".join(others)),
            )
    return out


def run(features: dict, prices, out_path: Path | None = None) -> list[ScorecardRow]:
    if out_path is None:
        out_path = Path("outputs/analyst") / f"proof_audit_{date.today().isoformat()}.json"

    # eligible_dates does one point-in-time universe lookup per date and depends only on
    # prices -- compute it once here instead of once per scored component.
    try:
        dates = eligible_dates(prices)
    except Exception as exc:  # pragma: no cover -- defensive; a bad price matrix fails every row
        log.warning("proof_audit: eligible_dates failed (%s) -- scoring per-component", exc)
        dates = None

    rows: list[ScorecardRow] = []
    for c in COMPONENTS:
        try:
            if c.method in _DEFERRED_REASON:
                rows.append(_row_for_deferred(c.name, c.kind, c.method))
            elif c.kind == "alpha_sleeve":
                result = score_sleeve(c.name, features, prices, dates=dates)
                rows.append(_row_from_result(c.name, c.kind, c.method, result))
            elif c.kind == "agent":
                result = score_agent(c.name, prices, dates=dates)
                rows.append(_row_from_result(c.name, c.kind, c.method, result))
            elif c.kind == "subsystem":
                rows.append(_row_from_result(
                    c.name, c.kind, c.method, score_subsystem(c.name),
                    reason=_SUBSYSTEM_REASONS.get(c.name),
                ))
        except DegenerateSignalError as exc:
            reason = (
                _MISSING_INPUT_REASON if c.name in _MISSING_INPUT_SLEEVES
                else f"{exc}{_DEGENERATE_SUFFIX}"
            )
            log.warning("proof_audit: %s degenerate -- %s", c.name, reason)
            rows.append(_failure_row(c, reason))
        except Exception as exc:
            reason = (
                f"{_MISSING_INPUT_REASON} ({exc})" if c.name in _MISSING_INPUT_SLEEVES
                else f"scoring failed: {exc}"
            )
            log.warning("proof_audit: %s failed (%s) -- marking INSUFFICIENT_DATA", c.name, exc)
            rows.append(_failure_row(c, reason))

    rows = _flag_duplicate_agent_scores(rows)

    write_scorecard(rows, out_path)
    log.info("proof_audit: wrote %d rows to %s", len(rows), out_path)
    return rows


def _dedupe_prices_by_calendar_day(price_df):
    """Collapse `prices_live` to one row per (symbol, calendar day).

    `prices_live` recurrently carries several intraday timestamps for the same
    trading day (e.g. 2022-12-21 00:00:00-05:00, 2022-12-21 19:00:00-05:00 and
    2022-12-21 20:00:00-05:00 -- all on the same calendar date; they do not
    straddle midnight), with *different* symbols populated on each -- not identical
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

    from ascent.data.store.parquet import has_data, load_parquet
    from ascent.data.normalize.prices import pivot_prices
    from ascent.features.build_features import FeatureBuilder

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

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
    run(features, prices, out_path=args.out)
