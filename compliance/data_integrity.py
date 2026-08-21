"""compliance/data_integrity.py

Data Integrity Officer -- Phase 5 of the compliance / middle-office layer.

Implements `docs/target_architecture/05_compliance_middle_office.md`'s Data
Integrity Officer role ("Is the data this whole pipeline runs on actually
clean?") and `docs/target_architecture/11_transformation_plan.md` Phase 5.

Standalone module: NOT wired into the live pipeline this task. That wiring
(a pre-flight gate before feature/alpha computation, per doc 05's Layer 3
description) is explicitly out of scope here -- matching how Task 2 built,
then separately wired, the Model Risk Reviewer (`ascent/risk/irm/`). This
task delivers a working, tested, standalone `check_cache()` only.

Reuse, not reimplementation
----------------------------
`scripts/reconcile_numbers.py`'s `data_integrity()` function already contains
the exact duplicate-row and phantom-row detection logic this officer needs --
the `_calendar_day_key`-keyed dedup and the non-midnight time-of-day check,
both written against the recurring `prices_live` corruption incident
documented in CLAUDE.md ("`prices_live` duplicate rows have recurred three
times"). This module calls that function rather than re-deriving the counts.

`data_integrity()` is hardcoded to a single module-level `PRICES` path
constant (`data_cache/prices_live.parquet`) instead of accepting a cache-name
parameter -- reconcile_numbers.py is being treated read-only per this task's
brief, so it is not refactored to take one. Instead, `check_cache()`
temporarily points `reconcile_numbers.PRICES` at the requested cache's
parquet file for the duration of a single call, then restores it. This still
runs the real function against real bytes on disk for whichever cache is
named; it does not copy or re-derive the duplicate/phantom-row counting
logic by hand.

Layer 4 contract (`docs/target_architecture/05`): `{cache: status,
duplicate_rows: int, phantom_rows: int, action: "pass"|"fallback_prior_day"|
"pass_with_warning"}`. `IntegrityResult` below is that contract as a
dataclass, plus a `severity` field for the HIGH/MEDIUM distinction the doc's
Layer 3 threshold draws, and a human-readable `detail`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Optional

from ascent.data.store.parquet import DATA_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- contract --

STATUS_CLEAN = "clean"
STATUS_DIRTY = "dirty"
STATUS_STALE = "stale"
STATUS_MISSING = "missing"
STATUS_ERROR = "error"

ACTION_PASS = "pass"
ACTION_FALLBACK = "fallback_prior_day"
ACTION_WARN = "pass_with_warning"

SEVERITY_NONE = "NONE"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"

# "cache staleness beyond the expected fetch cadence (e.g., no new row for
# >1 trading day where a fresh fetch should have occurred)" -- doc 05, Layer 3.
# Matches ascent/data/store/parquet.py::validate_cache's own default so the
# two staleness notions agree unless a caller overrides one of them.
DEFAULT_STALE_DAYS = 3


@dataclass
class IntegrityResult:
    """One cache's verdict, per doc 05's Layer 4 `data_integrity_<date>.json` contract."""

    cache: str
    status: str
    duplicate_rows: int
    phantom_rows: int
    action: str
    severity: str = SEVERITY_NONE
    detail: str = ""


# --------------------------------------------------------------- internals --

def _run_reconcile_data_integrity(cache_name: str) -> dict:
    """Call `scripts.reconcile_numbers.data_integrity()` against `cache_name`.

    `data_integrity()` reads its target path off the module-level `PRICES`
    constant rather than a parameter. We swap that constant to the requested
    cache's parquet path for the duration of this one call and restore it in
    a `finally`, so the *real* detection function runs against the *real*
    file -- this is reuse of that function, not a reimplementation of what it
    does.
    """
    import scripts.reconcile_numbers as reconcile_numbers

    original_prices = reconcile_numbers.PRICES
    try:
        reconcile_numbers.PRICES = DATA_DIR / f"{cache_name}.parquet"
        return reconcile_numbers.data_integrity()
    finally:
        reconcile_numbers.PRICES = original_prices


def _staleness_lag_days(date_max: Optional[str], as_of: Optional[_date] = None) -> Optional[int]:
    if not date_max:
        return None
    as_of = as_of or _date.today()
    try:
        cache_end = _date.fromisoformat(str(date_max)[:10])
    except ValueError:
        return None
    return (as_of - cache_end).days


# ------------------------------------------------------------------- API ---

def check_cache(
    cache_name: str,
    stale_days: int = DEFAULT_STALE_DAYS,
    as_of: Optional[_date] = None,
) -> IntegrityResult:
    """Run the Data Integrity Officer's pre-flight checks on one cache.

    Parameters
    ----------
    cache_name:
        Cache name as passed to `ascent.data.store.parquet.save_parquet` /
        `load_parquet` (e.g. ``"prices_live"``, ``"prices_macro"``) -- the
        file checked is ``data_cache/{cache_name}.parquet``.
    stale_days:
        Calendar-day lag beyond which a cache with no fresh-data integrity
        problem is still flagged MEDIUM / ``pass_with_warning``. Defaults to
        the same cadence `validate_cache()` uses.
    as_of:
        Reference date for staleness. Defaults to today; pass explicitly in
        tests so results don't depend on wall-clock date.

    Threshold, per doc 05's Layer 3 description of this role:
    - Any duplicate or phantom row detected -> ``action="fallback_prior_day"``,
      HIGH severity, printed and returned (no live fallback logic here --
      that's the Phase 3 / execution-layer concern doc 05 explicitly defers).
    - No duplicate/phantom rows but staleness beyond `stale_days` ->
      ``action="pass_with_warning"``, MEDIUM severity.
    - Otherwise -> ``action="pass"``, clean.
    """
    result = _run_reconcile_data_integrity(cache_name)

    if result.get("error"):
        status = STATUS_MISSING if "not found" in result["error"] else STATUS_ERROR
        verdict = IntegrityResult(
            cache=cache_name,
            status=status,
            duplicate_rows=0,
            phantom_rows=0,
            action=ACTION_FALLBACK,
            severity=SEVERITY_HIGH,
            detail=result["error"],
        )
        _log(verdict)
        return verdict

    dup_rows = int(result.get("dup_rows", 0))
    phantom_rows = int(result.get("non_midnight_rows", 0))

    if dup_rows > 0 or phantom_rows > 0:
        detail = (
            f"{dup_rows} duplicate (symbol, calendar-day) row(s), "
            f"{phantom_rows} non-midnight phantom row(s) in {cache_name}"
        )
        verdict = IntegrityResult(
            cache=cache_name,
            status=STATUS_DIRTY,
            duplicate_rows=dup_rows,
            phantom_rows=phantom_rows,
            action=ACTION_FALLBACK,
            severity=SEVERITY_HIGH,
            detail=detail,
        )
        _log(verdict)
        return verdict

    lag = _staleness_lag_days(result.get("date_max"), as_of=as_of)
    if lag is not None and lag > stale_days:
        detail = (
            f"{cache_name} latest date {result.get('date_max')} is {lag}d old "
            f"(cadence limit={stale_days}d)"
        )
        verdict = IntegrityResult(
            cache=cache_name,
            status=STATUS_STALE,
            duplicate_rows=0,
            phantom_rows=0,
            action=ACTION_WARN,
            severity=SEVERITY_MEDIUM,
            detail=detail,
        )
        _log(verdict)
        return verdict

    verdict = IntegrityResult(
        cache=cache_name,
        status=STATUS_CLEAN,
        duplicate_rows=0,
        phantom_rows=0,
        action=ACTION_PASS,
        severity=SEVERITY_NONE,
        detail=f"{cache_name}: clean, {result.get('rows', 0)} rows",
    )
    return verdict


def _log(verdict: IntegrityResult) -> None:
    line = (
        f"[DataIntegrityOfficer] {verdict.severity} cache={verdict.cache} "
        f"status={verdict.status} action={verdict.action} "
        f"duplicate_rows={verdict.duplicate_rows} phantom_rows={verdict.phantom_rows} "
        f"-- {verdict.detail}"
    )
    print(line)
    if verdict.severity == SEVERITY_HIGH:
        log.error(line)
    else:
        log.warning(line)


class DataIntegrityOfficer:
    """Thin object wrapper around `check_cache()` for call-site symmetry with
    `ReconciliationAnalyst` / `ModelRiskReviewer`-style classes elsewhere in
    the compliance layer (doc 05, Layer 5 concrete-implementation mapping).
    """

    def __init__(self, stale_days: int = DEFAULT_STALE_DAYS):
        self.stale_days = stale_days

    def check_cache(self, cache_name: str, as_of: Optional[_date] = None) -> IntegrityResult:
        return check_cache(cache_name, stale_days=self.stale_days, as_of=as_of)
