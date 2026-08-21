# Minimum-Viable-Cut Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the 4-6-weekend minimum-viable cut identified in
`docs/target_architecture/20_cost_timeline_reality_check.md`. Four of nine items in that cut
already shipped in commit `eb7dfdf` (governance bugfix, audit-trail wiring, rejected-hypothesis
log, Model Risk Reviewer built-but-unwired). This plan finishes the rest: wire MRR into the live
pipeline, fix the last Phase 0 housekeeping item, and build the two remaining net-new pieces
(data integrity wrapper, pre-trade compliance gate).

**Architecture:** Five independent-ish tasks against `docs/target_architecture/01`, `03`, `05`,
`11`, `12`, `14` (already-written blueprints/skeletons/checklists — read the cited one before
each task, it is the spec). Tasks 1-4 touch non-overlapping files and can be done in any order.
Task 5 (collapsing the two order-submission code paths) is the highest-risk item in the whole
cut — it touches live order-submission code — and is sequenced last, after the safer work has
built confidence in the branch's test discipline.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`, pytest. `scripts/reconcile_numbers.py` and
`ascent/execution/eod_runner.py`/`order_engine.py` are the key existing files several tasks
build on top of — read the relevant existing file before writing new code, never assume its
shape from the blueprint doc alone (the blueprints have already been shown to drift from
current source in prior verification passes — trust `git`/`grep` over the doc's prose).

## Global Constraints

- Always use `.venv/bin/python` for running tests/scripts.
- This is a solo-operator paper-trading system. No task in this plan may change what order gets
  submitted, at what size, or on what schedule — only add gates/checks/logging around the
  existing decision. If a task's own testing would require submitting a real order (even to
  Alpaca paper), stop and use mocked/injected data instead.
- TDD: write or extend tests alongside every change. Every task must leave `pytest` green on
  both its own new tests and every pre-existing test file it touches or that covers the code
  path it touches.
- Do not modify `ascent/portfolio/optimizer.py`'s water-fill cap, `ascent/execution/
  kill_switch.py`'s thresholds, or any existing gate's pass/fail thresholds — this plan only
  adds new gates alongside existing ones.
- Do not commit sensitive data, API keys, or `.env` contents. Do not push to any remote.
- Every new module gets a docstring stating what it does and citing which
  `docs/target_architecture/NN_*.md` file it implements, so the doc-to-code link survives.
- Nothing in this plan should be wired to have live veto/write power over order submission
  without an explicit shadow-mode flag defaulting to "log only, don't block" — per the project's
  own integrity constraint #5 precedent (advisory-until-proven), new gates start observational.

---

## Task 1: Fix the regime risk-multiplier silent try/except (Phase 0 item 4)

**Why:** `ascent/portfolio/optimizer.py:713-717` wraps a call to `regime_max_weight` in a bare
`except Exception: pass`, and the surrounding docstring still claims "max_weight is tightened
based on the current regime" — misleading if the call is actually failing. Per
`docs/target_architecture/12_phase0_execution_checklist.md` item 4 and
`docs/target_architecture/00_institutional_audit.md`'s "regime risk multiplier" finding
(`ascent/regime/integration.py`'s `regime_max_weight` may have been deleted as dead code — verify
current state, don't assume the doc's claim is still accurate).

**Files:** `ascent/portfolio/optimizer.py`, possibly `ascent/regime/integration.py` or
`ascent/regime/__init__.py` (read-only unless the fix requires touching them)

- [x] **Step 1: Diagnose what actually happens today** — DONE, commit d7fcd7c
- [x] **Step 2: Decide and implement based on what Step 1 found** — DONE, commit d7fcd7c
- [x] **Step 3: Test** — DONE, commit d7fcd7c

---

## Task 2: Wire Model Risk Reviewer into the live pipeline

**Why:** `ascent/risk/irm/model_risk_reviewer.py` (shipped in the baseline commit, 14 passing
tests) is a real, tested module but is standalone — not called from anywhere in the actual daily
pipeline. Per `docs/target_architecture/14_phase1_model_risk_reviewer_skeleton.md`'s "Exact
insertion point" section and `docs/target_architecture/01_risk_management.md` Layer 5.

**Files:** `ascent/main.py`, `ascent/risk/irm/model_risk_reviewer.py` (read-only — its interface
is already correct per the baseline commit's tests; do not change its signature unless you find
a genuine bug), possibly one new test file

- [x] **Step 1: Read the current `load_or_fetch_prices()` and `run_pipeline()`** — DONE, commit 7a506cc
- [x] **Step 2: Change `load_or_fetch_prices()` to return the reason** — DONE, commit 7a506cc
- [x] **Step 3: Call the Model Risk Reviewer at two points in `run_pipeline()`** — DONE, commit 7a506cc
- [x] **Step 4: Test** — DONE, commit 7a506cc, fix round 1 commit f54c145 (date-scoping bug)

---

## Task 3: Build the Data Integrity Officer (Phase 5)

**Why:** Per `docs/target_architecture/05_compliance_middle_office.md`'s Data Integrity Officer
role and `docs/target_architecture/11_transformation_plan.md` Phase 5 — "wraps
`scripts/reconcile_numbers.py`'s existing duplicate/phantom-row logic rather than reimplementing
it."

**Files:** existing top-level `compliance/` package (sibling to `compliance/audit_trail.py` —
controller ruling: do NOT create a new `ascent/compliance/` subpackage), new `data_integrity.py`
in that package, `scripts/reconcile_numbers.py` (read-only — reuse its functions, don't copy its
logic)

- [x] **Step 1: Read `scripts/reconcile_numbers.py` fully** — DONE, commit 370a811
- [x] **Step 2: Build `data_integrity.py`** — DONE, commit 370a811
- [x] **Step 3: Test** — DONE, commit 370a811

---

## Task 4: Build the Pre-Trade Compliance Checker (Phase 3)

**Why:** Per `docs/target_architecture/03_trading_execution.md`'s Pre-Trade Compliance Checker
role and `docs/target_architecture/16_phase2_and_phase3_skeletons.md`'s already-drafted skeleton
(read this fully — it has concrete code and a confirmed exact insertion point:
`ascent/execution/eod_runner.py` lines ~1013-1029, between the kill-switch check and
`cancel_all_orders()`). `LARGE_TRADE_THRESHOLD_PCT = 2.0` at `eod_runner.py:48` is confirmed to
have zero enforcement call sites anywhere in the file — this task gives it one.

**Files:** new `ascent/execution/compliance_gate.py`, `ascent/execution/eod_runner.py`
(additive change only, at the confirmed insertion point), `ascent/execution/order_engine.py`
(read-only — confirm the `Order` dataclass's actual fields before assuming `order.price` exists,
per the skeleton doc's own flagged open question)

- [x] **Step 1: Confirm the `Order` dataclass's actual fields** — DONE, commit 46761e8
- [x] **Step 2: Build `compliance_gate.py`** — DONE, commit 46761e8
- [x] **Step 3: Wire it into `eod_runner.py` at the confirmed insertion point (shadow-mode)** — DONE, commit 46761e8
- [x] **Step 4: Test** — DONE, commit 46761e8

---

## Task 5: Collapse the two order-submission paths (Phase 0 item 3) — highest risk, sequence last

**Why:** Per `docs/target_architecture/12_phase0_execution_checklist.md` item 3 and
`docs/target_architecture/03_trading_execution.md`: `run_eod()` (`eod_runner.py:103`) and
`run_eod_with_weights()` (`eod_runner.py:766`) independently reimplement the kill-switch-check →
order-loop → audit-log sequence with subtly different behavior (one swallows a non-
`KillSwitchTriggered` exception and continues, the other doesn't). Tasks 3-4 of this plan add new
gates that would otherwise need to be built twice if this collapse isn't done. This is
nonetheless the highest-risk task in the plan because it touches the actual order-submission
control flow both daily-run and multi-agent-merge paths depend on.

**Files:** `ascent/execution/eod_runner.py` only

- [ ] **Step 1: Read both functions completely, side by side**

Diff their behavior precisely: every place they diverge (the swallowed-exception branch is the
one already known; find any others). Write the divergences down as a short list before touching
any code — this list is what Step 2's tests must cover.

**IMPORTANT — carried forward from Task 4 (commit 46761e8):** Task 4 already inserted a
compliance-gate call (shadow-mode, log-only) into `run_eod_with_weights()`'s kill-switch→
cancel_all_orders region — the exact region this task's helper extraction covers. When you
extract the shared helper, you MUST preserve that compliance-gate call as part of the unified
helper. Do not treat the pre-Task-4 blueprint text as the current state of the function — read
the actual current code, which already has Task 4's call in it.

- [ ] **Step 2: Extract a shared internal helper**

Do not change either function's external signature (both are called from multiple sites in
`run_all_agents.py`). Extract the shared kill-switch-check → order-loop → audit-log sequence
(including Task 4's compliance-gate call, per Step 1's note above) into one internal helper
(e.g. `_execute_order_batch(...)`) that both `run_eod()` and `run_eod_with_weights()` call.
Resolve every behavioral divergence found in Step 1 as an explicit, deliberate choice (e.g.,
"always propagate non-KillSwitchTriggered exceptions, matching `run_eod()`'s stricter behavior,
since silently continuing past an unknown exception mid-order-loop is the riskier default") and
document the choice in a code comment at the point of resolution — do not let a divergence
silently resolve to "whichever happened to be written last."

- [ ] **Step 3: Test exhaustively**

This is the one task in the plan that should have MORE test coverage than its own diff, not
matching coverage. Before making the change, run the full existing TARGETED test suite covering
both functions (`grep -rl "run_eod\b\|run_eod_with_weights" tests/`) and record the baseline pass
count. After the change, the same tests must pass with identical behavior, PLUS new tests that
specifically exercise each divergence found in Step 1 (e.g., a test that forces a non-
KillSwitchTriggered exception mid-batch and asserts the now-unified behavior), PLUS a test
confirming Task 4's compliance-gate call still fires from both `run_eod()` and
`run_eod_with_weights()` post-refactor. Do not consider this task done until you can state, with
a specific test name, what covers each divergence identified in Step 1. Run only TARGETED tests
(see session-established pattern — full-suite runs hang on this branch).

- [ ] **Step 4: Do not enable this for live trading as part of this task**

This task's deliverable is a verified-correct refactor sitting in the working tree /
feature branch — it explicitly does not include flipping any live-trading switch, updating
`rebalance_calendar.csv`, or otherwise causing the next real rebalance to exercise this new code
path for the first time without a human explicitly choosing to. That decision belongs to the
project owner, not to this plan.
