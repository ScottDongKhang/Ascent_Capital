# Phase 7 Quick Wins — Ready-to-Implement Skeletons

Major correction to `11_transformation_plan.md`'s framing this cycle: **items
1 and 2 (rolling IC monitor, sleeve-level cut rule) are not net-new — they
already exist, live, in production.** `10_investment_thesis_lifecycle.md`'s
finding that Ascent's code has "zero coverage" of Stages 7-8 was **wrong**,
specifically for the sleeve level (it's still true for the *strategy/agent*
level covered by `04`). This cycle's job shifted from "build" to "hardening
what's already there plus building the one piece that's genuinely missing
(item 3)."

## Item 1 & 2 — Already exist: `ascent/alpha/stack.py::_get_gated_weights()`

Confirmed live, called at `stack.py:194` inside `build_alpha_stack()` — every
single run of the alpha stack passes through this gate, not a proposed
addition.

**What it does today** (`stack.py:68-140`):
- Reads `logs/sleeve_ic_log.jsonl`, last `window=5` unique-date entries per
  sleeve.
- Computes rolling mean IC per sleeve over that window.
- **Zeroes any sleeve whose rolling mean IC < `IC_GATE_THRESHOLD` (-0.005,
  `stack.py:21`)** — this *is* the mechanical, pre-committed, sleeve-level
  cut rule Phase 7 item 2 asked for. It's not discretionary, it's not a
  committee review, it fires automatically inside `build_alpha_stack()`.
- Redistributes the freed weight proportionally among surviving sleeves
  (`stack.py:131-137`), falling back to unchanged weights if every sleeve is
  gated (never collapses to zero exposure).
- This *is* the rolling IC monitor Phase 7 item 1 asked for — it's just not
  exposed as a dashboard, only as an internal gating mechanism.

**What's genuinely missing, and what this cycle recommends building instead
of a parallel system:**

1. **The window is 5 trading days.** Per `10_investment_thesis_lifecycle.md`
   Stage 7 (Lo 2002's standard-error analysis), distinguishing genuine decay
   from noise at realistic Sharpe/IC levels needs *years*, not days, of data
   for statistical confidence. A 5-day window will fire on ordinary IC
   variance, not just genuine decay — this is very likely **too trigger-happy**
   as currently configured, not too slow. **Recommend**: add a second,
   longer-window (e.g. 63-day) rolling IC computation purely for
   *dashboarding/trend-visibility* (see below), and treat the existing 5-day
   gate as a fast circuit-breaker for acute breaks (a sleeve producing
   actively harmful signal right now), not a decay detector — these are two
   different jobs currently conflated into one threshold.
2. **No dashboard/visibility layer.** `_get_gated_weights()`'s effect is only
   visible via a `log.warning()` call (`stack.py:125-128`) — there's no
   equivalent of `dashboard/agent_skill_scores.json` at the sleeve level. Add
   `dashboard/sleeve_ic_scores.json`, written by a new small script
   `ascent/monitoring/sleeve_ic_tracker.py` mirroring
   `ascent/monitoring/skill_tracker.py`'s existing structure (same
   `{agent: {score, n_days, status, latest_date}}` shape, generalized to
   `{sleeve: {rolling_ic_5d, rolling_ic_63d, status, latest_date}}`) — this
   is a read-only reporting addition, zero risk to the live gating path.
3. **No cross-check against `IC_GATE_THRESHOLD` being tuned reactively.** The
   threshold's own comment (`stack.py:21`: "tightened from -0.010: fundamental
   IC=-0.008 recently evaded the gate") shows it has already been hand-tuned
   once, after the fact, to catch a specific past miss. Worth a short note in
   the module: this threshold is empirically tuned, not derived from a
   pre-registered statistical target — consistent with `10`'s finding that
   published, rigorous decay-threshold conventions don't really exist
   industry-wide either.

## Item 3 — Genuinely missing: rejected-hypothesis log

`ascent/research/self_improve.py::LOG_PATH` (`logs/self_improve_log.jsonl`,
line 32) already logs every variant tested per run (`log_entry["variants"] =
results`, line 306) with a `promoted: bool` flag (line 304) — **this is close
to a rejected-hypothesis log already**, but it's per-run (all variants from
one `run_self_improve()` call bundled together) rather than a queryable,
per-hypothesis registry, and it doesn't check "has this exact configuration
been tried and rejected before" — so nothing currently prevents re-testing
the same failed idea, which was `10` Stage 9's core finding about why this
matters.

**Skeleton**: new file `ascent/research/hypothesis_registry.py`

```python
"""Append-only registry of tested signal/variant hypotheses and their
verdicts -- prevents re-testing an already-falsified idea. Reads the
existing self_improve_log.jsonl as its data source rather than duplicating
what evaluate_variant() already logs; adds a lookup index on top."""

import json
from pathlib import Path
from hashlib import sha256

SELF_IMPROVE_LOG = Path("logs/self_improve_log.jsonl")   # existing, read-only here
REGISTRY_PATH     = Path("logs/hypothesis_registry.jsonl")  # new


def _config_hash(variant_config: dict) -> str:
    """Stable hash of a variant config so re-tests of the identical
    configuration are detectable regardless of key ordering."""
    return sha256(json.dumps(variant_config, sort_keys=True).encode()).hexdigest()[:12]


def was_previously_rejected(variant_config: dict) -> dict | None:
    """Returns the prior rejection record if this exact config was already
    tested and NOT promoted, else None. Call this BEFORE evaluate_variant()
    spends compute re-testing something already known to fail."""
    h = _config_hash(variant_config)
    if not REGISTRY_PATH.exists():
        return None
    for line in reversed(REGISTRY_PATH.read_text().splitlines()):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("config_hash") == h and not entry.get("promoted"):
            return entry
    return None


def record_verdict(variant_config: dict, variant_id: str, oos_sharpe: float,
                    edge: float, promoted: bool, reason: str = "") -> None:
    """Append one hypothesis verdict. Called once per variant from inside
    evaluate_variant()'s existing logging block (self_improve.py:296-309),
    not as a separate pass -- keep the two logs in sync by construction."""
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    entry = {
        "config_hash": _config_hash(variant_config),
        "variant_id":  variant_id,
        "config":      variant_config,
        "oos_sharpe":  oos_sharpe,
        "edge":        edge,
        "promoted":    promoted,
        "reason":      reason,
    }
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Exact insertion point**: `ascent/research/self_improve.py`, inside the loop
that produces `results` (the per-variant evaluation loop feeding
`evaluate_variant()`, around where `generate_variants()`'s output — line 80 —
is iterated before the `edge > MIN_SHARPE_EDGE` check at line 289). Two call
sites:
1. Before evaluating a variant: `if (prior := was_previously_rejected(variant_config)): skip and log "already tested, rejected on {prior['date']}"` — saves compute on exact repeats.
2. After the existing `log_entry` write at `self_improve.py:308-309`: add
   `record_verdict(...)` for each variant in `results`, using the same
   `edge`/`promoted` values already computed — no new evaluation logic, just
   an additional structured write alongside the existing one.

This is additive and low-risk: it doesn't change `evaluate_variant()`'s
decision logic, only adds a pre-check (skip known-bad repeats) and a
post-write (queryable registry) around the existing, unchanged evaluation.
