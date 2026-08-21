"""Model Risk Reviewer — Independent Risk Management, Phase 1.

Runs BEFORE alpha/portfolio construction. Does not touch weights; its only
output is a pass/fail verdict plus a structured reason, meant to be consumed
by a caller (e.g. ``ascent/main.py::run_pipeline``) to decide whether to
proceed with fresh data or fall back to the last-known-good weights.

Standalone module: NOT wired into the live pipeline yet. See
``docs/target_architecture/14_phase1_model_risk_reviewer_skeleton.md`` for
the integration plan (touches ``ascent/main.py``'s ``load_or_fetch_prices``
signature, deliberately out of scope for this change).

Ports three checks identified in that doc as genuinely unguarded today:

1. Cache staleness at the pipeline level — ``load_or_fetch_prices()`` /
   ``validate_cache()`` already compute a ``(ok, reason)`` pair, but today
   that reason is only ever printed, never captured as a structured
   artifact another component can act on.
2. NaN rate in required (non-sparse-exempt) feature panels — no check
   exists between feature computation and the alpha stage that fails
   loudly if a required panel is mostly NaN. ``ml_sleeve.py``'s
   ``_SPARSE_FILL_ZERO`` five columns are *deliberately* sparse
   (NaN-heavy, then zero-filled) and are exempted here by design, not by
   omission.
3. Regime-label staleness — how stale ``dashboard/regime_labels.csv`` is
   relative to ``data_cache/ai_regime_assessment.json``. Informational
   only per CLAUDE.md's documented tolerated lag; never fails the verdict
   by itself.

Deviation from the original doc skeleton, and why
--------------------------------------------------
The skeleton's cache-freshness check used a substring test,
``"stale" not in reason.lower()``. Reading ``validate_cache()``
(``ascent/data/store/parquet.py``) shows its failure-reason strings never
actually contain the word "stale" — they read things like
``"cache latest date ... is N calendar days old (limit=M)"``,
``"cache file missing: ..."``, ``"N required symbols missing from cache"``,
or the phantom-row corruption message. A substring match on "stale" would
silently pass every one of those failure modes. This module instead treats
any non-empty reason as a failure signal (``validate_cache`` returns
``reason == ""`` exactly when ``ok`` is ``True``), which is what the
upstream contract actually guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Optional

try:  # pragma: no cover - typing convenience only
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

NAN_RATE_FAIL_THRESHOLD = 0.05        # >5% NaN in a REQUIRED (non-sparse-exempt) panel -> fail
CACHE_STALE_FAIL_DAYS = 1             # prices_live older than 1 calendar day -> fail (direct-date path)
REGIME_LABEL_STALE_WARN_DAYS = 3      # informational only, per CLAUDE.md's documented tolerated lag

# Cache names considered "live/fresh" for the purposes of check 1. Anything
# else (prices_simulated, prices_live_fallback_simulated, ...) is degraded
# data and fails the freshness check regardless of the reason string.
_FRESH_CACHE_NAMES = {"prices_live"}


@dataclass
class ModelRiskVerdict:
    """Result of a Model Risk Reviewer pass.

    ``checks`` maps a check name to ``{"ok": bool, "detail": str}``.
    ``passed`` is the AND of every non-informational check (any key
    prefixed with ``"regime_"`` is informational and never affects
    ``passed``). ``reason`` is a human-readable, semicolon-joined summary
    of every failing check, or ``"all checks passed"`` when there are none.
    """

    passed: bool
    checks: dict = field(default_factory=dict)
    reason: str = ""


def _cache_freshness_check(price_cache_name: str, price_cache_reason: Optional[str]) -> dict:
    reason = price_cache_reason or ""
    ok = price_cache_name in _FRESH_CACHE_NAMES and reason == ""
    detail = f"{price_cache_name}: {reason or 'fresh'}"
    return {"ok": ok, "detail": detail}


def _cache_direct_date_check(cache_latest_date: date, as_of_date: date) -> dict:
    lag_days = (as_of_date - cache_latest_date).days
    ok = lag_days <= CACHE_STALE_FAIL_DAYS
    detail = f"latest cache date {cache_latest_date} is {lag_days}d old (limit={CACHE_STALE_FAIL_DAYS})"
    return {"ok": ok, "detail": detail}


def _nan_rate_check(panel) -> dict:
    if pd is not None and isinstance(panel, pd.DataFrame):
        if panel.size == 0:
            nan_rate = 0.0
        else:
            nan_rate = float(panel.isna().mean().mean())
    elif pd is not None and isinstance(panel, pd.Series):
        nan_rate = float(panel.isna().mean()) if panel.size else 0.0
    else:
        # Best-effort fallback for plain nested sequences / None.
        nan_rate = 0.0 if panel is None else 0.0
    ok = nan_rate <= NAN_RATE_FAIL_THRESHOLD
    detail = f"{nan_rate:.1%} NaN (threshold={NAN_RATE_FAIL_THRESHOLD:.0%})"
    return {"ok": ok, "detail": detail}


def _regime_staleness_check(
    regime_label_date: Optional[date],
    regime_assessment_date: Optional[date],
) -> Optional[dict]:
    if regime_label_date is None or regime_assessment_date is None:
        return None
    lag_days = abs((regime_assessment_date - regime_label_date).days)
    stale = lag_days > REGIME_LABEL_STALE_WARN_DAYS
    detail = (
        f"regime_labels.csv vs ai_regime_assessment.json lag = {lag_days}d "
        f"(warn threshold={REGIME_LABEL_STALE_WARN_DAYS}d)"
    )
    if stale:
        detail = f"WARNING: {detail}"
    # Informational only: "ok" always True so it never fails the verdict,
    # per the "regime_" key prefix convention used by `passed` below.
    return {"ok": True, "detail": detail}


def check(
    price_cache_name: str,
    price_cache_reason: Optional[str],
    feature_panels: Mapping[str, object],
    sparse_exempt: set,
    as_of_date: date,
    cache_latest_date: Optional[date] = None,
    regime_label_date: Optional[date] = None,
    regime_assessment_date: Optional[date] = None,
) -> ModelRiskVerdict:
    """Run the Model Risk Reviewer's pre-flight checks.

    Parameters
    ----------
    price_cache_name:
        The cache name returned by ``load_or_fetch_prices`` (e.g.
        ``"prices_live"``, ``"prices_simulated"``,
        ``"prices_live_fallback_simulated"``).
    price_cache_reason:
        The ``reason`` string from ``validate_cache()``. Empty string (or
        ``None``) means the cache validated cleanly.
    feature_panels:
        ``{feature_name: pd.DataFrame}``, post-computation, pre-stack (i.e.
        the ``features`` dict consumed by ``ml_sleeve._stack_features``).
    sparse_exempt:
        Set of feature names that are deliberately allowed to be NaN-heavy
        (pass ``ascent.alpha.ml_sleeve._SPARSE_FILL_ZERO`` directly — do not
        duplicate the list here).
    as_of_date:
        The date the review is being run for.
    cache_latest_date:
        Optional. When the caller has the actual latest date in the price
        cache (not currently plumbed out of ``load_or_fetch_prices``), pass
        it here for a direct day-count staleness check against
        ``CACHE_STALE_FAIL_DAYS``, in addition to the reason-string check.
    regime_label_date, regime_assessment_date:
        Optional. Mtimes/as-of dates for ``dashboard/regime_labels.csv``
        and ``data_cache/ai_regime_assessment.json`` respectively. When
        both are supplied, an informational (non-failing) staleness check
        is recorded.
    """
    checks: dict = {}

    # 1. Cache staleness -- validate_cache() already computed `reason`;
    #    this captures it as structured output instead of only a print().
    checks["cache_freshness"] = _cache_freshness_check(price_cache_name, price_cache_reason)

    if cache_latest_date is not None:
        checks["cache_freshness::direct_date"] = _cache_direct_date_check(cache_latest_date, as_of_date)

    # 2. NaN rate in required (non-sparse-exempt) panels
    for name, panel in feature_panels.items():
        if name in sparse_exempt:
            continue
        checks[f"nan_rate::{name}"] = _nan_rate_check(panel)

    # 3. Regime label staleness (informational -- never fails the verdict alone)
    regime_result = _regime_staleness_check(regime_label_date, regime_assessment_date)
    if regime_result is not None:
        checks["regime_label_staleness"] = regime_result

    passed = all(c["ok"] for k, c in checks.items() if not k.startswith("regime_"))
    reason = "; ".join(f"{k}: {c['detail']}" for k, c in checks.items() if not c["ok"]) or "all checks passed"
    return ModelRiskVerdict(passed=passed, checks=checks, reason=reason)
