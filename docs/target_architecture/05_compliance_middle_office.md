# Compliance & Middle Office / Operations — Target Architecture

> **Correction (adversarial verification pass).** This document as originally
> written claimed no durable, tamper-evident audit trail exists in the codebase
> and proposed building `ascent/compliance/audit_trail.py` from scratch. That
> claim was false. A `compliance/` package **already exists at repo root**
> (commit `da3afb4`, "feat(plan7): compliance infrastructure," predating this
> document): `compliance/audit_trail.py` implements a **SHA-256 hash-chained,
> append-only** audit log (`AuditTrail`, `record_event()`), already covering
> `order_submitted`, `kill_switch_triggered`, `approval_granted/denied`,
> `debate_verdict`, and more (`VALID_EVENT_TYPES`), already writing to
> **`logs/audit_trail.jsonl`** — the exact path this document independently
> proposed — and already wired into `ascent/execution/eod_runner.py:37`
> (`from compliance.audit_trail import record_event as _audit`, called at line
> 352). Sibling modules `compliance/performance_report.py` (GIPS TWR),
> `compliance/risk_disclosure.py`, and `compliance/methodology_index.py` also
> already exist. **Everywhere below that describes the Audit Trail /
> Recordkeeping Officer as net-new, read it instead as: extend and wire the
> existing `compliance/audit_trail.py` into the additional call sites this
> document identifies (halt/override events, reconciliation writes,
> surveillance flags) — the hash-chained log itself does not need to be built.**
> The Reconciliation Analyst, Anomaly/Surveillance Analyst, and Data Integrity
> Officer roles below remain genuinely net-new; only the Audit Trail role's
> "what's missing" framing was wrong.

## Layer 1 — Department Mandate

**Mission**: independent recordkeeping and reconciliation, structurally separate
from the desk (`agents/us_equities_agent.py`, `ascent/execution/`) and from risk
(`ascent/risk/`, `orchestrator/central_intelligence.py`). "Independent" means this
department reads broker- and system-of-record truth and *never* computes the
numbers it's checking — it consumes `alpaca_broker.get_positions()`/`get_account()`
as ground truth against the internal weights/holdings log, the same way a real
fund's ops desk reconciles against the prime broker's statement rather than the
PM's blotter.

**What it exists to catch** (each with a real precedent already in this codebase):
1. **Broker/book divergence** — the internal `merged_weights` / holdings log says
   one thing, Alpaca's actual filled positions say another (partial fills, rejected
   orders, a crashed `run_eod_with_weights` mid-loop).
2. **Data integrity issues** — the `prices_live` duplicate-row incident (recurred
   three times per CLAUDE.md, most recently a phantom non-midnight row that
   silently disagreed with `_calendar_day_key`) is exactly the class of failure a
   Data Integrity function exists to catch *before* it reaches alpha/portfolio, not
   after a numbers audit finds it weeks later.
3. **Unauthorized or anomalous trades** — an order whose filled qty/price deviates
   materially from the debate-approved target weights (e.g., the
   `_apply_falsifier_trim` near-miss: a real-order-submitting write path built on
   unmeasured AI PM output that was caught only by manual review, per the CLAUDE.md
   integrity constraint #5 history).
4. **Missing or non-durable audit trail** — currently the only durable "who/what/
   when/why" record is `debate/adversarial_authority.py`'s `record_intervention()`,
   and it only covers debate-layer interventions, not order submission, halts, or
   overrides. Everything else lives in ephemeral `print()`/stdout and
   `logs/eod_log.jsonl`, which is an application log, not a recordkeeping system
   (no tamper-evidence, no retention policy, no independent writer).

**Reporting line**: this department reports *out-of-band* from both desk and risk
— it does not write to `active_alpha_config.json`, portfolio weights, or
execution, and its findings are consumed by the kill-switch/halt mechanism
(`ascent/execution/kill_switch.py`, `HALT_STATE_PATH`) rather than by the agents
that produced the numbers being checked. At this scale (single-strategy,
single-agent, paper trading, one human principal), "reports to" means: writes
machine-readable reports to `outputs/compliance/`, and any HALT-severity finding is
enforced structurally via `check_halt_state()` — nobody has to manually read a
report for the halt to take effect. Escalation to a human (the owner) happens via
the halt/override file, mirroring the existing halt-override precedent in
`run_all_agents.py`.

**Proportionality**: no restricted-list screening for material-nonpublic
information, no regulatory Form filings, no best-execution TCA against a NBBO
tape — those solve problems this fund doesn't have (no client assets, no
multi-broker routing, no material information walls). What's proportionate:
reconciliation, an immutable audit trail, and anomaly detection on the one thing
that actually moves capital — order submission.

## Layer 2 — Roles

| Role | Analogue | Core question |
|---|---|---|
| **Reconciliation Analyst** | Fund ops reconciliation | Does the broker's book match ours? |
| **Audit Trail / Recordkeeping Officer** | Compliance recordkeeping | Is every decision durably logged with who/what/when/why? |
| **Anomaly / Surveillance Analyst** | Trade surveillance | Does any order look inconsistent with what was approved? |
| **Data Integrity Officer** | Data ops / IT controls | Is the data this whole pipeline runs on actually clean? |

## Layer 3 — Per-Role Responsibilities and Decision Logic

**Reconciliation Analyst**
- Runs nightly, after `run_eod_with_weights` completes (fills have settled) but
  before the next day's pipeline starts.
- Compares `alpaca_broker.get_positions()` (qty, avg cost) + `get_account()` (cash,
  equity) against the internal holdings log (`_log_holdings()`'s output) for the
  same trading day.
- Thresholds: any single-position discrepancy **> 1 share** (fractional-share
  tolerance) or **> $50 notional**, whichever is looser, is logged as a
  discrepancy. A discrepancy **> $500 notional or > 5% of that position's target
  weight** on any single name halts next-day trading pending manual review (writes
  `HALT_STATE_PATH` with `requires_override: true`). **Three consecutive trading
  days with any unreconciled discrepancy** (regardless of size) escalates to a
  full-book halt — this catches slow-bleed drift (e.g., a silently-failing
  partial-fill handler) that any single day's threshold wouldn't trip.
- Cash reconciliation: `get_account()` cash vs. internal expected cash after known
  order flow — same thresholds, separate line item (cash breaks and position
  breaks have different root causes and must not be conflated in one report).

**Audit Trail / Recordkeeping Officer**
- Not a nightly batch job — a synchronous, append-only write triggered at the
  moment of each decision: order submission, halt trip, halt override, kill-switch
  trip/reset, debate verdict, AI PM position-change proposal (whether
  `applied=True` or `False`).
- Extends the `record_intervention()` precedent (already correctly append-only,
  idempotent on a natural key, and already logs `applied=False` proposals for
  auditability) to cover the categories it doesn't: raw order submissions and
  halt/override events currently only hit stdout and `logs/eod_log.jsonl`, neither
  of which is tamper-evident or has a retention guarantee.
- Decision logic: no discretion — every qualifying event gets a record with
  `{timestamp, actor (agent name / "operator"), action, inputs, outputs, reason}`.
  Idempotency key mirrors `record_intervention`'s `(date, symbol)` pattern,
  generalized to `(date, event_type, natural_key)` per event class, to survive the
  same re-entrant-call-site failure mode that has caused duplicate-write bugs
  elsewhere in the codebase.

**Anomaly / Surveillance Analyst**
- Runs immediately after order submission (same run, not nightly — catches
  same-day issues while the halt override is still cheap to apply, before the next
  day's decisions compound on top of an unnoticed bad fill).
- Flags: (a) filled qty/price deviating from the approved target weight by **> 2%
  of portfolio NAV** on a single name → auto-flag for review, **> 5%** →
  auto-halt; (b) **3+ consecutive broker order rejections** for the same symbol
  within one session → auto-flag (this is the surveillance analogue of
  `cancel_all_orders()`/`get_open_orders()` returning something unexpected); (c)
  any order present in the internal target-weight file but *absent* from Alpaca
  fills with no corresponding rejection reason logged (silent drop) → auto-halt,
  the same severity class as any silent-drop mini-rebalance incident.
- Explicitly out of scope at this scale: market-abuse/spoofing pattern detection
  (single-strategy paper book, no other market participants to manipulate against)
  and best-execution slippage review beyond simple deviation checks (no
  multi-venue routing to compare against).

**Data Integrity Officer**
- Runs at the start of the daily pipeline, before feature/alpha computation — a
  pre-flight gate, not a post-mortem.
- Checks, directly grounded in the recurring `prices_live` incident: duplicate-row
  count via `_calendar_day_key` dedup, non-midnight/"phantom row" count (the
  specific failure mode that survived two prior fixes), row-count/date-span sanity
  per cache (`prices_live`, `prices_macro`, `prices_international`,
  `prices_alternatives`), and presence of `feature_names` in the ML sleeve cache
  (guards the known "XGBoost crashes on shape mismatch" gotcha).
- Threshold: **any duplicate or phantom row detected** in a cache about to be read
  halts *that day's use of that cache* (falls back to prior-day cache, not
  full-book halt) and files a HIGH-severity report; **cache staleness beyond the
  expected fetch cadence** (e.g., no new row for >1 trading day where a fresh
  fetch should have occurred) is MEDIUM and logged but non-blocking, since some
  caches (macro/intl/alternatives) already have known slower fetch cadences.
- This role would have caught the 2026-08-15 phantom-row incident (a same-day
  duplicate at a non-midnight intraday timestamp with disjoint symbol coverage) at
  the source, rather than requiring the ad hoc audit that ultimately found it —
  that's the whole justification for the role existing as a standing daily check
  rather than a one-off cleanup.

## Layer 4 — Interfaces / Data Contracts

- **Reconciliation Analyst** — reads `alpaca_broker.get_positions()`,
  `alpaca_broker.get_account()`, and the internal holdings log (`_log_holdings()`
  output / `logs/eod_log.jsonl`). Emits `outputs/compliance/reconciliation_<date>.
  json`: `{status: "clean"\|"discrepancy"\|"halt", discrepancies: [{symbol,
  broker_qty, internal_qty, notional_delta, cash_delta}], consecutive_break_days:
  int}`. Consumed by `check_halt_state()`'s halt-writer.
- **Audit Trail / Recordkeeping Officer** — reads nothing computed; it's a write
  sink invoked inline by other modules (order submission in
  `ascent/execution/eod_runner.py`, halt logic in `run_all_agents.py`,
  `record_intervention()` call sites in `debate/`). Emits append-only
  `logs/audit_trail.jsonl` (parallel to, not replacing, `logs/eod_log.jsonl`), one
  durable record per event, never rewritten (contrast with
  `adversarial_authority.py`'s `_rewrite_interventions`, which *does* rewrite —
  audit trail entries must not be rewritable by design).
- **Anomaly / Surveillance Analyst** — reads the approved target-weight file
  (`merged_weights` / `active_alpha_config.json` snapshot at decision time) and
  `alpaca_broker.get_open_orders()` + fill confirmations from `submit_order()`
  return values. Emits `outputs/compliance/surveillance_<date>.json`: `{flags:
  [{symbol, type: "deviation"\|"rejection"\|"silent_drop", severity, detail}]}`.
  HALT-severity flags feed the same halt-writer as reconciliation.
- **Data Integrity Officer** — reads each `prices_*` parquet cache directly (via
  `load_parquet`) before any consumer touches it. Emits
  `outputs/compliance/data_integrity_<date>.json`: `{cache: status,
  duplicate_rows: int, phantom_rows: int, action: "pass"\|"fallback_prior_day"\|
  "pass_with_warning"}`. Consumed by the pipeline's data-loading step (a gate, not
  a downstream report) and by `scripts/reconcile_numbers.py`, which already
  surfaces duplicate/phantom-row counts and would become this officer's canonical
  metric source rather than a separate ad hoc script.

## Layer 5 — Concrete Implementation Mapping

New modules under a new `ascent/compliance/` package, mirroring existing package
conventions (`ascent/risk/`, `ascent/execution/`):

- `ascent/compliance/reconciliation.py` — `class ReconciliationAnalyst`, method
  `reconcile(run_date) -> ReconciliationReport`. Reads
  `ascent.execution.alpaca_broker.get_positions()`/`get_account()` directly (same
  import pattern as `eod_runner.py` already uses). Writes
  `outputs/compliance/reconciliation_<date>.json` and, on halt-severity, writes
  `HALT_STATE_PATH` using the same schema `check_halt_state()` already parses
  (`{halt_date, reason, requires_override, verdict_path}`) — no new halt mechanism
  needed, just a new writer into the existing one.
- `ascent/compliance/audit_trail.py` — `record_event(event_type, actor, inputs,
  outputs, reason, date_str, natural_key)`, structurally a generalization of
  `debate/adversarial_authority.py`'s `record_intervention`/`_append_intervention`
  (same append-only-JSONL, same idempotent-on-natural-key pattern) but writing to
  `logs/audit_trail.jsonl` and covering event types beyond interventions:
  `order_submitted`, `halt_triggered`, `halt_overridden`, `kill_switch_tripped`,
  `kill_switch_reset`.
- `ascent/compliance/surveillance.py` — `class SurveillanceAnalyst`, method
  `scan(approved_weights, fills, rejections) -> SurveillanceReport`.
- `ascent/compliance/data_integrity.py` — `class DataIntegrityOfficer`, method
  `check_cache(cache_name) -> IntegrityResult`, built directly on the
  duplicate/phantom-row logic already in `scripts/reconcile_numbers.py`'s
  data-integrity section rather than reimplementing it — that script becomes this
  officer's library, not a parallel tool.

**Where each plugs into `run_all_agents.py`'s daily sequence, and why:**

1. **Data Integrity Officer runs first**, before `validate_sector_data()` (line
   ~308) and before any agent reads a price cache — bad data must be caught
   before it propagates into alpha/portfolio, not audited after the fact.
2. **Audit Trail Officer's `record_event()` calls are inlined** at every existing
   decision point: inside `run_eod_with_weights()` at order submission (currently
   only `print`-logged), inside `check_halt_state()` on halt/override (lines
   360-402), and alongside every existing `record_intervention()` call in
   `debate/`. This is instrumentation, not a pipeline stage — it has no standalone
   slot in the sequence.
3. **Surveillance Analyst runs immediately after order submission** (after line
   1897's `run_eod_with_weights(..., dry_run=False, ...)` returns fills), same
   run — catching a bad fill same-day, while an override is still cheap, beats
   catching it the next morning after the book has already compounded on top of it.
4. **Reconciliation Analyst runs at the very start of the *next* day's pipeline**,
   before `check_halt_state()` (line 360) is evaluated for that day —
   reconciliation of yesterday's settled fills must complete and have a chance to
   write a halt *before* `check_halt_state()` decides whether today's
   `run_eod_with_weights` is allowed to fire. This mirrors why pre-thesis is
   documented as "consumed by the *next* run, not the current one" in CLAUDE.md:
   broker settlement isn't same-day-reliable (the existing "same-day Track B is
   unreliable" gotcha — Alpaca 1D bars settle late afternoon PT), so
   reconciliation against `get_portfolio_history()` (settled bars, the pattern
   CLAUDE.md already mandates over same-day equity deltas) can only run credibly
   the following morning.

Net effect: two gates bracket order submission (data integrity before,
surveillance right after) and one gate brackets the *day boundary* (reconciliation
before the next day's halt check), with the audit trail wired in as a cross-cutting
write at every decision point rather than a pipeline stage of its own.
