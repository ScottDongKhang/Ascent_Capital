#!/usr/bin/env python
"""
Measure the point-in-time (PIT) sector-classification gap in the canonical
walk-forward backtest.

Why this exists
----------------
`ascent/research/walk_forward_runner.py` builds a single `sector_map` dict
ONCE from the CURRENT `profiles` cache (each symbol's 2026 sector
classification) and reuses that same static map across every fold of the
2020-01-02 -> 2026-07-15 walk-forward backtest, including folds testing
2020/2021 data. The sector-constrained position cap (`max_per_sector`,
`sector_constrained_weighted()` in `ascent/portfolio/optimizer.py`) therefore
applies each company's 2026 sector classification retroactively to early
folds. Companies get reclassified over time (GICS revisions, business-mix
shifts, and especially mergers/acquisitions/spin-offs) — this is a
look-ahead-flavored bias in the sector-cap machinery, distinct from the
price/return look-ahead bias the project already guards against via
point-in-time joins (`ascent/data/store/point_in_time.py`).

This is a MEASUREMENT script, not a fix. A true PIT fix needs historical
sector-classification data with a `known_time` column, which this project
does not have (confirmed: the `profiles` cache carries no `known_time`
column). This script quantifies how bad the static-map assumption plausibly
is, using `ascent/data/universe.py::REMOVED_STOCKS` (260 S&P 500 removals,
2013-2026, each with a reason string) as the best available proxy for
"companies whose business/sector identity plausibly changed."

What this script measures
--------------------------
1. Of the 260 tracked removals, how many have a reason string indicating a
   business-identity change (acquisition, merger, spin-off) as opposed to a
   routine index-committee reshuffle ("Market capitalization change") or a
   failure ("... receivership").
2. Of those flagged symbols, how many still resolve to a sector in the
   CURRENT `profiles` cache (`sector_map`) — i.e. the static map has *an*
   opinion about a symbol whose corporate identity changed, which is exactly
   the situation where that opinion is least trustworthy for early folds.
3. A documented BLIND SPOT: this method can only see reclassification that
   coincided with a tracked S&P 500 removal event. A company that stayed in
   the index the whole time but silently changed its GICS sector (a pure
   business-mix shift, no corporate action) is invisible to this script.
   That case is NOT measurable from REMOVED_STOCKS alone and is called out
   explicitly rather than silently under-counted.
4. A rough estimate of exposure: using the simplified date-range-overlap
   method (per symbol `start_date`/`end_date` from
   `build_historical_universe(strict=True, sp500_only=True)` against an
   approximate walk-forward rebalance-date grid), how many of the walk-
   forward backtest's ~165 folds could plausibly have held at least one of
   the flagged symbols.

Usage
-----
    .venv/bin/python scripts/measure_sector_pit_gap.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascent.data.universe import (  # noqa: E402
    REMOVED_STOCKS,
    build_historical_universe,
)
from ascent.data.store.parquet import has_data, load_parquet  # noqa: E402

# ---------------------------------------------------------------------------
# Backtest window / fold grid this script approximates. These match the
# canonical walk-forward artifact documented in CURRENT_VERIFIED_NUMBERS.md
# (§1): OOS window 2020-01-02 -> 2026-07-15, 165 rolling folds, rebalanced
# every 10 trading days (cfg.backtest.rebalance_freq_days default).
# ---------------------------------------------------------------------------
WF_START = "2020-01-02"
WF_END = "2026-07-15"
REBAL_FREQ_TRADING_DAYS = 10

# ---------------------------------------------------------------------------
# Reason-string classification. Built by inspecting the actual reason text
# in REMOVED_STOCKS rather than guessing exact strings — most reasons are
# one of a handful of templated phrasings from the Wikipedia changes table.
# ---------------------------------------------------------------------------
_ACQUISITION_RE = re.compile(
    r"acquir|taken over|taken private|consortium|purchased", re.IGNORECASE
)
_SPINOFF_RE = re.compile(
    r"spun off|spin[- ]?off|spins off|separated into", re.IGNORECASE
)
_MERGER_RE = re.compile(r"merg", re.IGNORECASE)  # merge/merger/merged/merging/merges
_RECEIVERSHIP_RE = re.compile(r"receivership", re.IGNORECASE)
_MCAP_RE = re.compile(r"market cap", re.IGNORECASE)


def classify_reason(reason: str) -> str:
    """Classify one REMOVED_STOCKS reason string into a bucket."""
    is_acq = bool(_ACQUISITION_RE.search(reason))
    is_spin = bool(_SPINOFF_RE.search(reason))
    is_merger = bool(_MERGER_RE.search(reason))
    if is_acq or is_spin or is_merger:
        tags = []
        if is_acq:
            tags.append("acquisition")
        if is_spin:
            tags.append("spinoff")
        if is_merger:
            tags.append("merger")
        return "+".join(tags)
    if _RECEIVERSHIP_RE.search(reason):
        return "receivership (failure, not reclassification)"
    if _MCAP_RE.search(reason):
        return "index reshuffle (not a business-identity change)"
    return "other/unclassified"


BUSINESS_IDENTITY_BUCKETS = {"acquisition", "spinoff", "merger"}


def is_business_identity_change(bucket: str) -> bool:
    return any(tag in BUSINESS_IDENTITY_BUCKETS for tag in bucket.split("+"))


def load_current_sector_map() -> dict:
    """Reproduce the exact sector_map construction in walk_forward_runner.py
    (lines ~288-291): sector_map = dict(zip(profiles['symbol'], profiles['sector']))."""
    if not has_data("profiles"):
        print("[WARN] No 'profiles' cache found — sector_map will be empty.")
        return {}
    profiles = load_parquet("profiles")
    return dict(zip(profiles["symbol"], profiles["sector"]))


def approximate_rebalance_dates() -> pd.DatetimeIndex:
    """Approximate the walk-forward's rebalance-date grid: every
    REBAL_FREQ_TRADING_DAYS-th NYSE trading day (approximated here with
    pandas business days, which is what walk_forward_runner.py effectively
    operates over once weekends are excluded from the loaded price cache).
    This is a simplification — see docstring for why it's acceptable."""
    all_bdays = pd.bdate_range(start=WF_START, end=WF_END)
    return all_bdays[::REBAL_FREQ_TRADING_DAYS]


def main() -> None:
    print("=" * 78)
    print("  POINT-IN-TIME SECTOR-CLASSIFICATION GAP — MEASUREMENT")
    print("=" * 78)
    print()
    print(f"Backtest window approximated: {WF_START} -> {WF_END}")
    print(f"REMOVED_STOCKS entries tracked: {len(REMOVED_STOCKS)}")
    print()

    # --- Step 1: classify every removal reason ---------------------------
    classified = []
    for symbol, sector, removed_date, reason in REMOVED_STOCKS:
        bucket = classify_reason(reason)
        classified.append(
            {
                "symbol": symbol,
                "sector_at_removal": sector,
                "removed_date": removed_date,
                "reason": reason,
                "bucket": bucket,
                "business_identity_change": is_business_identity_change(bucket),
            }
        )
    df = pd.DataFrame(classified)

    bucket_counts = df["bucket"].apply(
        lambda b: "+".join(sorted(set(t for t in b.split("+") if t in BUSINESS_IDENTITY_BUCKETS)))
        if is_business_identity_change(b)
        else b
    )
    print("-" * 78)
    print("Step 1: Reason-string classification of all %d removals" % len(df))
    print("-" * 78)
    for bucket, count in bucket_counts.value_counts().items():
        print(f"  {count:4d}  {bucket}")
    print()

    flagged = df[df["business_identity_change"]].copy()
    n_flagged = len(flagged)
    print(
        f"==> {n_flagged} of {len(df)} reason-flagged removals are "
        f"corporate-action-driven business-identity changes\n"
        f"    (acquisition / merger / spin-off) where the static sector_map is "
        f"most likely wrong for early folds."
    )
    print()

    # --- Step 2: cross-reference against the CURRENT sector_map ----------
    print("-" * 78)
    print("Step 2: Cross-reference flagged symbols against the CURRENT sector_map")
    print("        (same construction as walk_forward_runner.py lines ~288-291)")
    print("-" * 78)
    sector_map = load_current_sector_map()
    print(f"Current sector_map size (from 'profiles' cache): {len(sector_map)} symbols")
    print()

    flagged["still_in_current_sector_map"] = flagged["symbol"].apply(
        lambda s: s in sector_map
    )
    flagged["current_sector_map_value"] = flagged["symbol"].apply(sector_map.get)

    n_still_resolves = int(flagged["still_in_current_sector_map"].sum())
    print(
        f"  {n_still_resolves} of {n_flagged} flagged (acquired/merged/spun-off) "
        f"symbols STILL resolve to a sector in the current profiles cache."
    )
    if n_still_resolves:
        print(
            "  This is the most suspicious subset: their old ticker still carries a "
            "current sector classification, even though the corporate action means "
            "the entity behind that ticker may no longer be the same business (or "
            "may not exist as a distinct listed entity at all). Any early fold that "
            "held this symbol used a sector label from an entity that had not yet "
            "undergone the change that makes today's label questionable."
        )
        print()
        print("  Flagged symbols still resolving in the current sector_map:")
        subset = flagged[flagged["still_in_current_sector_map"]][
            ["symbol", "sector_at_removal", "current_sector_map_value", "removed_date", "reason"]
        ]
        for _, row in subset.iterrows():
            print(
                f"    {row['symbol']:<8} removal-era sector={row['sector_at_removal']:<22} "
                f"current sector_map={row['current_sector_map_value']:<22} "
                f"removed={row['removed_date']}  ({row['reason']})"
            )
    print()

    # --- Step 3: documented blind spot ------------------------------------
    print("-" * 78)
    print("Step 3: Documented blind spot (NOT measurable from REMOVED_STOCKS)")
    print("-" * 78)
    print(
        "  REMOVED_STOCKS only captures symbols that LEFT the S&P 500 with a "
        "recorded reason. A company that stayed in the index continuously from "
        "2020 to today but had its GICS sector silently revised (a pure "
        "business-mix shift with no acquisition/merger/spin-off/removal event — "
        "e.g. a conglomerate re-weighting its revenue mix, or an index-provider "
        "GICS methodology revision) is INVISIBLE to this method. There is no "
        "removal event to flag it, and the profiles cache carries no historical "
        "sector snapshots or 'known_time' column to detect the change directly.\n"
        "  This means the true PIT sector gap in the walk-forward backtest is "
        "AT LEAST as large as the count in Step 1/Step 2 below, and plausibly "
        "larger by an unknown amount. Do not report the flagged count as a "
        "complete measurement of the gap — it is a lower bound."
    )
    print()

    # --- Step 4: fold exposure (simplified date-range overlap) -----------
    print("-" * 78)
    print("Step 4: Approximate fold exposure (simplified date-range overlap method)")
    print("-" * 78)
    print(
        "  Method used: SIMPLIFIED per-symbol start_date/end_date overlap against "
        "an approximate rebalance-date grid (every "
        f"{REBAL_FREQ_TRADING_DAYS} business days over {WF_START}..{WF_END}, "
        "matching cfg.backtest.rebalance_freq_days=10), rather than calling "
        "get_universe_on_date() per actual walk-forward fold date (which requires "
        "the live price cache to reconstruct exactly). This is the accepted "
        "simplification per the task scoping."
    )
    universe_df = build_historical_universe(strict=True, sp500_only=True)
    removed_universe = universe_df[universe_df["status"] == "removed"]

    rebal_dates = approximate_rebalance_dates()
    n_total_folds = len(rebal_dates)
    print(f"  Approximate rebalance dates in window: {n_total_folds}")

    flagged_symbols = set(flagged["symbol"])
    flagged_ranges = removed_universe[removed_universe["symbol"].isin(flagged_symbols)][
        ["symbol", "start_date", "end_date"]
    ]

    folds_with_exposure = 0
    fold_hit_details = []
    for d in rebal_dates:
        hits = flagged_ranges[
            (flagged_ranges["start_date"] <= d) & (flagged_ranges["end_date"] >= d)
        ]
        if len(hits) > 0:
            folds_with_exposure += 1
            fold_hit_details.append((d, sorted(hits["symbol"].tolist())))

    print(
        f"  {folds_with_exposure} of {n_total_folds} approximate folds "
        f"({folds_with_exposure / n_total_folds:.1%}) could plausibly have held "
        f"at least one of the {n_flagged} flagged symbols in the tradeable "
        f"universe on that rebalance date."
    )
    if not flagged_ranges.empty:
        n_symbols_matched = flagged_ranges["symbol"].nunique()
        n_symbols_missing = n_flagged - n_symbols_matched
        print(
            f"  ({n_symbols_matched} of {n_flagged} flagged symbols found in the "
            f"strict, sp500_only historical universe with a start/end date; "
            f"{n_symbols_missing} were excluded from this overlap check — most "
            f"likely because they lack a recorded addition date under "
            f"strict=True, e.g. removed before UNIVERSE_START or never in "
            f"SYMBOL_ADDITION_DATES.)"
        )
    print()

    # --- Summary -----------------------------------------------------------
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(
        f"  {n_flagged} of {len(df)} tracked S&P 500 removals (2013-2026) are "
        f"corporate-action-driven business-identity changes (acquisition / "
        f"merger / spin-off) — the clearest cases where the static, "
        f"current-day sector_map used by walk_forward_runner.py is most "
        f"likely wrong for early folds."
    )
    print(
        f"  {n_still_resolves} of those {n_flagged} still resolve to a sector "
        f"in the CURRENT profiles cache, meaning the static map silently "
        f"supplies a (probably-wrong) sector label for them in every fold, "
        f"including 2020/2021 folds years before the change."
    )
    print(
        f"  Roughly {folds_with_exposure} of ~{n_total_folds} walk-forward "
        f"folds ({folds_with_exposure / n_total_folds:.0%}) could plausibly "
        f"have held at least one flagged symbol — i.e. this is not a rare-edge-"
        f"case bias, it plausibly touches a meaningful share of the backtest."
    )
    print(
        "  BLIND SPOT (not counted above): symbols that never left the index "
        "but had a silent sector reclassification are invisible to this "
        "measurement. The true gap is a lower bound, not a complete count."
    )
    print(
        "  A real fix requires sourcing historical point-in-time sector "
        "classification data (with a known_time column) — out of scope for "
        "this measurement pass."
    )


if __name__ == "__main__":
    main()
