# Ascent Capital — Project State Report

Compiled from 4 parallel agent scans (git history, live operational state,
shipped-code health, docs synthesis), each grounded in direct file/command
output, not memory. Repo: `/Users/scott/IdeaProjects/ascent-capital`,
branch `main`, working tree clean.

---

## 1. Current Operational State — read this first

**The system is not currently running live, and has not been for 25 days.**
This is a distinct, newer problem from the earlier documented decision to
hold live trading — it looks like the scheduler simply isn't loaded, not a
deliberate stop or a code failure.

- **Last actual pipeline run**: 2026-07-27 (`logs/eod_log.jsonl`'s last
  entry — itself a `catch_up` run recovering from a prior 19-day outage).
  25 days ago from today.
- **Scheduler status**: the daily/heartbeat launchd jobs
  (`com.ascentcapital.eod.plist`, `com.ascentcapital.heartbeat.plist`) exist
  on disk in `~/Library/LaunchAgents/` but are **not loaded** —
  `launchctl list com.ascentcapital.eod` returns "Could not find service in
  domain." Only unrelated services (`com.ascent.mirofish`,
  `com.ascent.litellm`) are currently running.
- **Monitoring is also stale**: `logs/liveness.json` was last regenerated
  2026-08-12 (already reading `CRITICAL` then) and hasn't updated since —
  even the alerting artifact isn't being refreshed. At least one more
  scheduled rebalance (2026-08-19) has since passed with no log entry.
- **Nothing is broken or emergency-halted**: no tripped kill switch (no
  state file exists — defaults to untripped), no active halt-state file,
  git tree clean.
- **Book performance as of last update** (`CURRENT_VERIFIED_NUMBERS.md`,
  generated 2026-07-28): $104,640.21 equity, +3.79% since 2026-04-01, but
  **book vs SPY: −8.98%**. AI PM authority: Level 1 ("Analyst"), 5% weight,
  stuck 19 days. AI-layer counterfactual advantage not statistically
  significant (B−A★ = −5.92pp/70d, t=−1.24; D−A★ = −3.04pp/47d, t=−0.94).

**Action implied, not taken**: reload the two launchd jobs if resuming
scheduled runs is intended, or confirm the pause is still deliberate.

---

## 2. What Changed This Session

16 commits, `8952198..main` (55 files, +9744/−169 lines). Two are routine
automated dashboard-chore commits, not analyzed further.

**Documentation** (28 files, ~4950 lines): `docs/target_architecture/00`
through `26` + `README.md` — a full institutional-firm audit (risk
management, alpha research, trading/execution, CIO/capital allocation,
compliance/middle-office, judgment/governance, staffing/regulatory
benchmarks, transformation plan, empirical alpha/beta audits) — plus the
SDD implementation plan that executed pieces of it
(`docs/superpowers/plans/2026-08-20-min-viable-cut-completion.md`).

**New capabilities** — wiring status confirmed by grep, not assumed:

| Module | What it does | Wired into live path? |
|---|---|---|
| `ascent/risk/irm/model_risk_reviewer.py` | Pre-flight cache-staleness + NaN-rate check | **Yes** — two checkpoints in `ascent/main.py::run_pipeline()`, shadow-mode (log-only, never blocks) |
| `ascent/execution/compliance_gate.py` | Pre-trade restricted-symbol / large-order / buying-power checks | **Yes** — `eod_runner.py::_execute_order_batch()`, shadow-mode (logs `[ComplianceGate][SHADOW] Would reject...`, never filters `orders`) |
| `compliance/data_integrity.py` | Duplicate/phantom-row cache checker, wraps `reconcile_numbers.py` | **No** — standalone, only called from its own tests |
| `ascent/research/hypothesis_registry.py` | Prevents re-testing already-falsified signal configs | **Yes** — wired into `ascent/research/self_improve.py`'s research loop (not the daily trading pipeline) |

**Modified existing behavior**:
- `ascent/portfolio/optimizer.py` — removed a dead, silently-broken regime
  risk-multiplier call and its misleading docstring.
- `ascent/execution/eod_runner.py` — collapsed `run_eod()` and
  `run_eod_with_weights()`'s duplicated order-submission logic into one
  shared helper, resolving 5 documented divergences with no change to what
  gets submitted, at what size, or on what schedule.
- `ascent/research/walk_forward_runner.py` — now persists per-fold daily
  returns to `outputs/wf_results/`; fixed an unconditional import of a
  deleted module that was silently aborting runs.
- `ascent/strategy/earned_authority.py` — added a stuck-buffer diagnostic.
- `compliance/audit_trail.py` + `run_all_agents.py` — halt/override events
  now hit the durable hash-chained audit trail.
- `docs/REPO_MAP.md`, `CLAUDE.md` — two stale-claim corrections.

**No commit in this range flips a kill switch, changes the alpha sleeve
set, or writes to live weights** — consistent with integrity constraint #5.

---

## 3. Shipped-Code Health (verified now, not just at merge time)

**84/84 tests passing across 7 targeted files, 0 failures.** `scripts/
verify_docs.py`: 25/25 checks green, zero doc drift. All 4 new modules
import cleanly.

Every "file doesn't exist yet" state was traced to its handling code and
confirmed to be an intentional, tested branch — not an unguarded assumption:
- `data_cache/large_trade_approvals.json` (absent) → fails closed to an
  empty approval list, every large order gets shadow-rejected as designed.
- `logs/hypothesis_registry.jsonl` (absent) → `None` return, correctly read
  as "no prior rejection," created lazily on first real verdict.
- `logs/sleeve_ic_log.jsonl` (absent) → `{}` return, pre-existing behavior,
  unrelated to this session's changes.

**Nothing needs attention before the next pipeline run.**

---

## 4. Bottom Line

**Does this operate like an institutional hedge fund, structurally?** No,
and this is settled. Real portfolio-construction math (MVO, vol-targeting,
kill-switch, sector caps) but no independent risk function — every cap
lives inside the same optimizer that sizes the book, and the only override
is a binary all-stop, not a scalpel. No independent compliance/
reconciliation function. Single-strategy systematic book with
governance-shaped scaffolding, not a multi-desk institutional platform.

**Is there real, measured trading edge?** This was the open question and
it's now closed, negatively, with directly computed evidence. The two live
sleeves (meanrev, statarb) are textbook reversal signals, beta 0.947 to
SPY. A real 165-fold walk-forward run, with the actual daily return series
persisted, gives a computed beta-hedged Sharpe of **≈ −0.10**. Recovered
historical IC data initially suggested dormant trend/insider sleeves beat
the live pair 2-7x — **this was re-checked and reversed for `trend`**
(`docs/target_architecture/27_trend_insider_reconciliation.md`): the
formal, larger-sample proof audit found trend has a statistically
significant *negative* IC (an anti-signal, correctly cut), and the
recovered-log finding that looked good was a statistical illusion — 21
"observations" that were really one expanding-window running average
sampled near its tail. Insider remains a genuine open question (positive,
not yet significant), not a promotion candidate on current evidence. What's
live is, on current evidence, beta dressed as a governance-heavy strategy.

**Single highest-priority next step:** don't build more governance/
compliance infrastructure around the current sleeve pair — it would
rigorously validate a signal already indistinguishable from beta. Pause
and re-underwrite (already formalized as an IC memo): hold current weights
flat, no new capital. Do not chase `trend` — it's a confirmed anti-signal.
For `insider`, the real next step is collecting more insider-transaction
data and rerunning the formal proof audit, not promoting on existing
evidence.

**Planning vs. shipped code:** this flipped mid-session. It started as
"26 planning documents, zero shipped code" — but a follow-on implementation
pass actually shipped and merged real code, and the single most important
deliverable (the empirical alpha answer) is now committed, tested code,
not a projection.

**Most urgent unresolved item, independent of all of the above:** the
scheduler being unloaded for 25 days is a separate, more immediate
question than the strategy-quality question — worth resolving (reload or
confirm-and-document the pause) regardless of what happens with the alpha
question.
