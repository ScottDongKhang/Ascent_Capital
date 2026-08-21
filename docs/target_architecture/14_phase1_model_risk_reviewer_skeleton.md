# Phase 1 — Model Risk Reviewer: Ready-to-Implement Skeleton

Grounded directly in `ascent/main.py` and `ascent/alpha/ml_sleeve.py` as they
exist today. One correction to `01_risk_management.md`'s original framing: it
described the ML feature-name mismatch as an unguarded crash risk. **It's
already partially guarded** — worth knowing before "fixing" something that
isn't broken, and narrowing this role's actual job to what's genuinely
missing.

## What's already guarded (don't rebuild)

`ascent/alpha/ml_sleeve.py:450`:
```python
_feature_mismatch = _cached_features is not None and list(_cached_features) != list(available)
if _cached_model is not None and _cached_train_date is not None and not _feature_mismatch:
    ...  # fast path
# else falls through to full CPCV retrain — no crash, no silent bad prediction
```
This already prevents the shape-mismatch crash the CLAUDE.md gotcha warns
about — a mismatch forces a full retrain instead of predicting on stale
feature ordering. **The Model Risk Reviewer's job is not to add this check —
it's to make the fallback visible and countable** (right now it's a silent
`else` fallthrough with no log line distinguishing "cache fresh, fast path" /
"feature mismatch, forced retrain" / "cache stale, scheduled retrain" — all
three currently look the same from outside the function).

## What's genuinely unguarded

1. **Cache staleness at the pipeline level**, not just the ML sleeve.
   `ascent/main.py::load_or_fetch_prices()` (line 168-246) already computes a
   `(df, cache_name)` pair and a `reason` string from `validate_cache()` (line
   189-195) — but that reason is only ever printed (line 201), never captured
   as a structured artifact another component could act on.
2. **NaN rate in required feature panels** — no check exists anywhere between
   feature computation and the alpha stage that fails loudly if a required
   panel is mostly NaN (as opposed to `_SPARSE_FILL_ZERO`'s five *expected*-
   sparse columns, `ml_sleeve.py:291-297`, which are deliberately allowed to be
   NaN-heavy and zero-filled — a genuinely broken panel outside that named set
   would currently just silently reduce row count via `.dropna()` at
   `ml_sleeve.py:332`, with no volume/rate check on how much was dropped).
3. **Regime-label staleness cross-check** (`dashboard/regime_labels.csv` vs
   `data_cache/ai_regime_assessment.json`) — CLAUDE.md documents this as a
   known, tolerated lag, but nothing currently checks *how stale*, just that
   it can happen.

## Skeleton

New file: `ascent/risk/irm/model_risk_reviewer.py`

```python
"""Model Risk Reviewer — Independent Risk Management, Phase 1.

Runs BEFORE alpha/portfolio construction. Does not touch weights; its only
output is a pass/fail verdict plus a structured reason, consumed by the
caller (ascent/main.py::run_pipeline) to decide whether to proceed with
fresh data or fall back to the last-known-good weights.
"""

from dataclasses import dataclass, field
from datetime import date

NAN_RATE_FAIL_THRESHOLD = 0.05       # >5% NaN in a REQUIRED (non-sparse-exempt) panel -> fail
CACHE_STALE_FAIL_DAYS   = 1          # prices_live older than 1 trading day -> fail
REGIME_LABEL_STALE_WARN_DAYS = 3     # informational only, per CLAUDE.md's documented tolerated lag


@dataclass
class ModelRiskVerdict:
    passed: bool
    checks: dict = field(default_factory=dict)   # {check_name: {"ok": bool, "detail": str}}
    reason: str = ""


def check(
    price_cache_name: str,
    price_cache_reason: str,
    feature_panels: dict,          # {feature_name: pd.DataFrame}, post-computation, pre-stack
    sparse_exempt: set,            # pass ml_sleeve._SPARSE_FILL_ZERO directly -- don't duplicate the list
    as_of_date: date,
) -> ModelRiskVerdict:
    checks = {}

    # 1. Cache staleness -- validate_cache() already computed `reason`;
    #    this just captures it as structured output instead of only a print().
    cache_ok = price_cache_name == "prices_live" and "stale" not in (price_cache_reason or "").lower()
    checks["cache_freshness"] = {"ok": cache_ok, "detail": f"{price_cache_name}: {price_cache_reason}"}

    # 2. NaN rate in required (non-sparse-exempt) panels
    for name, panel in feature_panels.items():
        if name in sparse_exempt:
            continue
        nan_rate = float(panel.isna().mean().mean())
        ok = nan_rate <= NAN_RATE_FAIL_THRESHOLD
        checks[f"nan_rate::{name}"] = {"ok": ok, "detail": f"{nan_rate:.1%} NaN"}

    # 3. Regime label staleness (informational -- never fails the verdict alone)
    # left as a stub call site; wire to dashboard/regime_labels.csv +
    # data_cache/ai_regime_assessment.json mtimes at integration time.

    passed = all(c["ok"] for k, c in checks.items() if not k.startswith("regime_"))
    reason = "; ".join(f"{k}: {c['detail']}" for k, c in checks.items() if not c["ok"]) or "all checks passed"
    return ModelRiskVerdict(passed=passed, checks=checks, reason=reason)
```

## Exact insertion point

`ascent/main.py::run_pipeline()`, right after line 338
(`price_df, price_cache_name = load_or_fetch_prices(cfg, live)`) captures the
cache name, and again after the feature-computation step (find the call that
builds the `features` dict consumed by `ml_sleeve.py`'s `_stack_features` —
not yet re-verified this cycle, follow the `features = ...` assignment
between STEP 1 and STEP 2 in `run_pipeline`). Two check points, not one:

```python
# ascent/main.py, immediately after line 338:
price_df, price_cache_name = load_or_fetch_prices(cfg, live)
# --- Model Risk Reviewer, check 1: cache freshness ---
from ascent.risk.irm.model_risk_reviewer import check as _mrr_check
_cache_verdict = _mrr_check(
    price_cache_name=price_cache_name,
    price_cache_reason=reason,   # capture `reason` from validate_cache() call inside load_or_fetch_prices --
                                  # requires load_or_fetch_prices to return it, a small signature change
    feature_panels={}, sparse_exempt=set(), as_of_date=date.today(),
)
if not _cache_verdict.passed:
    print(f"[IRM] Model Risk Reviewer cache check failed: {_cache_verdict.reason}")
    # fall back to last-known-good weights -- reuse whatever existing
    # "hold current book" pattern run_all_agents.py already has elsewhere
```

A second call after feature computation (panel NaN-rate check) needs the
`features` dict, which is built later in `run_pipeline` — this cycle
confirmed `load_or_fetch_prices`'s signature and staleness-reason plumbing but
did not re-locate the exact `features = ...` line; that's a 10-minute grep for
whoever implements this, not a blocker to writing the module now.

## Note on `load_or_fetch_prices`'s signature change

Currently returns `(df, cache_name)` (`main.py:338` unpacking). Adding a third
return value (`reason`) is a small, low-risk signature change but touches
every call site — confirmed via grep this cycle there are exactly 2 call
sites in non-test code (`ascent/main.py`, and one other file not yet
re-checked). Verify both before changing the signature; do not assume test
files don't also unpack this tuple positionally.
