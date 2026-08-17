# Ascent Capital — Research Compiler Redesign (PAT-derived)

**Date:** 2026-07-31
**Revision:** 2 — rewritten after four-reviewer adversarial audit
**Status:** **NOT approved.** Revision 1 was approved section-by-section in conversation; the
audit overturned material parts of it. §7 (demotion) and §9.1 (whether the compiler is built
at all) require owner re-decision before anything proceeds.

**Scope:** Replace the *analysis* half of `ascent/monitoring/` with a compiled research layer.
Impose canonical series naming. Build a unified benchmark suite.

---

## 0. What the audit overturned

Revision 1 was reviewed by four independent agents: an evidence auditor (claims vs artifacts),
a PAT-fidelity reviewer, a feasibility reviewer, and a red team. Their findings are folded in
below. The material reversals, recorded here so this document does not quietly launder them:

| Rev-1 claim | Status after audit |
|---|---|
| "Nothing that submits an order changes. Blast radius bounded by construction." | **FALSE.** `ascent/monitoring/` contains the rebalance trigger and the pre-execution safety gate. See §2.2. |
| "At 0%, Track D becomes a clean, unconfounded measure. Demoting improves measurement." | **FALSE.** `blend()` never mutates `ai_portfolio`; Track D snapshots the pure pre-blend vector at any `ai_weight`. Demoting deletes Track B's information instead. See §7.2. |
| "All 29 monitoring modules are a question asked once and frozen." | **PARTLY FALSE.** Roughly 11–14 are gates, schedulers, or live-IO, not analyses. See §5. |
| "The catalog is a lens, not a database." | **MISLEADING.** 6–8 structurally distinct adapter families, 4–6 weeks. See §6. |
| "Declared schemas would have caught ~10 historical bugs." | **~5.** The other five live in code this spec declares out of scope, and were already fixed by ordinary means. See §8.2. |
| `tests/` is 124 files; `verify_docs.py` has 24 assertions | **WRONG.** 147 and 25. Both were copied from `docs/REPO_MAP.md` rather than measured — the exact failure mode this spec exists to eliminate, committed inside the spec. |
| Two LLM→book write paths | **THREE.** A third `apply_judge_position_change` call exists on the discovery path at `run_all_agents.py:2979`. `CLAUDE.md` constraint #5 already documents it. |

---

## 1. Provenance

Derived from Bridgewater's public "Pocket Analyst Tool" (PAT) talk at Interrupt 2026
(Brendan McManus, Michael Ran, Santi Weight) plus published AIA Labs material:

- <https://www.bridgewater.com/aia-labs/how-bridgewaters-aia-labs-built-pat-the-ai-pocket-analyst-tool>
- <https://www.zenml.io/llmops-database/building-pat-an-ai-analyst-for-investment-research-at-scale>
- <https://www.bridgewater.com/aia-labs>
- <https://www.hedgeweek.com/bridgewater-launches-2bn-machine-learning-driven-fund/>

Bridgewater runs two separate tracks: PAT (research copilot, hundreds of investors, does not
trade) and AIA Labs (small autonomous fund, ~$2bn, Greg Jensen).

**Caveat the red team raised, and it is fair:** their reasons for that split — access control
across 300 investors, headcount, separate client capital and governance — do not apply to a
solo builder. This document tacitly conceded that by dropping the chat agent and per-user
scoping (§4.2). **The Bridgewater split is therefore demoted from "the single most important
finding" to supporting context.** The recommendations here must stand on their own evidence,
and where they cannot, they are marked as not standing.

Per the ZenML critique: PAT's benchmarks are internal, unvalidated externally; no failure
rates disclosed; learning governance unaddressed. Copy the architecture, not the confidence.

---

## 2. The problem, restated honestly

### 2.1 What is genuinely wrong

`ascent/monitoring/` is 29 modules, 5,751 lines (verified). **Five artifacts answer one
question**: `ai_pm_counterfactual.py` (475), `counterfactual_tracker.py` (177),
`counterfactual_rebuild.py` (231), `scripts/backfill_counterfactual.py` (212),
`scripts/rebuild_counterfactual_log.py` (133). Each time the answer came out wrong, a new
module was written instead of the analysis being regenerated.

That is a real, evidenced pathology. **Note precisely what it is evidence of: missing
deduplication discipline — no canonical name for a question — not proof that deterministic
hand-written Python is the wrong tool.** §9 keeps those two conclusions separate.

### 2.2 The blast-radius claim was false

`run_all_agents.py` holds **26 references** into `ascent.monitoring`, and some sit in the
execution control path:

```python
from ascent.monitoring.rebalance_trigger import is_triggered, check_ic_decay_trigger
if is_triggered():
    is_rebalance = True        # flips real Alpaca order submission on
```

`pre_rebalance_checklist.run_checklist()` returns `blocking=True` on kill-switch and API-key
failure and gates whether execution proceeds.

So `ascent/monitoring/` is not a reporting layer. It contains the trigger and the safety gate.
Any claim of containment must be **enforced**, not asserted by directory convention (§4.4).

### 2.3 The learning-rate argument (survives)

Per `CURRENT_VERIFIED_NUMBERS.md` (2026-07-28): 2 scored Phase-2 decisions, effective n closer
to 2 than 70, `t = −1.24` on B−A★. Biweekly calendar. Judgment confidence takes years to
accrue, while analytical correctness could accrue per query. That asymmetry is real and is the
strongest surviving motivation for this work.

---

## 3. What this does not do

- **It does not produce alpha.** Sharpe 0.41, WFE −0.65, book trailing SPY by −8.98pp
  (`CURRENT_VERIFIED_NUMBERS.md` §1, §2). Strategy problems, separate workstream, unscoped.
- **The throughput→alpha link is asserted, not measured.** Nobody has timed how long a research
  question takes today versus post-compiler. Since `ascent/research/` (walk-forward, CPCV,
  factor discovery) is explicitly unchanged, the compiler mainly accelerates *post-hoc*
  monitoring and attribution — one step further from finding new alpha than rev 1 implied.
- It does not claim the AI layer's judgment is bad; per `CURRENT_VERIFIED_NUMBERS.md` §3 that
  judgment has never been cleanly measured.
- It does not claim PAT's reported figures (4x codegen speed, 95% determinism) generalize.

---

## 4. Architecture

```
ascent/analyst/
  catalog/     series registry: name, schema, provenance, as-of, search      [PAT 1, 9]
  plan/        AnalysisPlan: tasks, schemas, dependency DAG, open questions  [PAT 2, 3]
  codegen/     parallel sub-agents, one per task, dependency-schema scoped   [PAT 4]
  verify/      static analysis, DAG layering, forced schema validation       [PAT 5]
  runtime/     framework-owned execution + cache injection                   [PAT 6]
  review/      self-critique pass bound to the grounding rule                [PAT 7]
  report/      provenance-bearing output                                     [PAT 8]
  teach/       fail-first benchmark -> context/harness fix -> suite -> PR    [PAT 10]
  context/     step-by-step workflow guides per analysis type
```

### 4.1 The task schema (rev 1's largest hole)

PAT's determinism is a *consequence* of plan rigor. Rev 1 asserted the consequence and never
designed the cause; `plan/` got one line while the catalog got an eight-field table. A task
declares:

| Field | Purpose |
|---|---|
| `task_id` | Stable address; the cache key's root. |
| `description` | Natural-language statement of what to compute. |
| `inputs` | Catalog series names **plus their resolved schemas**, not just names. |
| `depends_on` | Upstream `task_id`s **plus their declared output schemas** — a sub-agent must see the shape it consumes without seeing the code that made it. |
| `output_name` | Canonical series name if the result is registered (PAT stage 9). |
| `output_schema` | Columns, dtypes, `index_kind`, expected row semantics. |
| `postconditions` | The invariants of §10.2, attached to the task that must satisfy them. |
| `cache_key` | Deterministic function of `task_id` + input series versions + code hash. Governs `runtime/` reuse and idempotent re-runs. |

**Test of adequacy:** two independent codegen agents given the same task must produce
semantically equivalent code. If they do not, the task schema is underspecified — that is the
diagnostic, not the model's fault.

### 4.2 Deliberate deviations from PAT

| PAT element | Decision | Rationale |
|---|---|---|
| Chat agent (LangGraph) | Dropped as a component | Solo operator who codes; the Claude Code session is the chat surface. |
| Clarifying-question phase | **Kept, moved into the planner** | "The plan is the analysis." The planner emits open questions and refuses to commit until answered. |
| Clarifying-question *quality* investment | **Deferred, explicitly** | PAT invested heavily in teaching what makes a good research question. Rev 1 dropped this silently. It is a real gap; see §10.5. |
| Per-user context/tool scoping | Dropped | Access control for 300 investors. Does not apply. |
| LangGraph for the deterministic half | Not used | Santi: the coding agent is "just Python code, no agentic orchestration" — that is what makes it deterministic. `CLAUDE.md` also mandates the single wrapper in `ascent/llm/client.py`. |
| Run cancellation | **Open loose end** | PAT's chat agent handled cancelling in-flight runs. Nothing here does. Matters once a bad plan can start an expensive parallel codegen sweep. |

### 4.3 Component contracts (rev 1 left these hollow)

- **`codegen/`** — uses `ascent/llm/client.py` only. Must honour the Claude 5 rules in
  `CLAUDE.md`: `extract_text` never `content[0].text`, no `temperature`/`top_p`/`top_k`, no
  `thinking={...}`, keep thinking enabled in tool loops, echo thinking blocks back unchanged.
  One sub-agent per task, scoped to its own spec plus dependency schemas. Partial-failure
  policy: a failed task fails its dependents, never silently emits an empty frame.
  **Cost and latency per analysis are unestimated** — see §10.6.
- **`verify/`** — plain Python, no agent orchestration. Static-analyses generated code to build
  the DAG, layers it, and validates each output against its declared `output_schema` and
  `postconditions`. Bounded repair attempts, then hard failure. Agents cannot skip it because
  they never invoke it.
- **`runtime/`** — the framework executes; the agent never invokes its own code. **Reuses the
  existing `data_cache/` parquet layer and `as_of_join`/`as_of_merge` rather than inventing a
  second cache.** Given `prices_live` duplicates have recurred twice, a new cache layer is a
  liability, not a feature.
- **`review/`** — bound to `CLAUDE.md`'s grounding rule: every number in the output must trace
  to a named artifact read during the run. That rule is the natural specification for this
  stage; rev 1 never connected them.
- **`report/`** — renders. **`catalog/` owns stage-9 write-back**, not `report/`. Rev 1 left
  that ambiguous.

### 4.4 Containment must be enforced

Rev 1 asserted "nothing writes to execution" by directory convention. This document's own
central argument is that *you cannot claim governance you have never observed working* — that
skepticism applies to this claim too.

**Requirement:** a check in `verify/` and in the benchmark suite asserting that no
codegen-produced module imports from `ascent.execution`, `ascent.portfolio`, or any broker
path, and that generated code performs no writes outside a designated output area. This is the
same class of guard as `verify_docs.py`'s existing assertions.

---

## 5. `ascent/monitoring/` splits into two tracks

Rev 1's "each module is a question with a golden answer" is a category error for roughly half
of them.

**Track A — analyses (migrate).** Pure question→answer over catalog inputs. Roughly 15–18
modules, including `ai_pm_counterfactual`, `attribution`, `weekly_debrief`, `live_vs_backtest`,
`alpha_wedge_tracker`, `skill_tracker`, `signal_health`, `regime_trajectory`.

**Track B — gates, orchestration, live IO (do NOT migrate).** Named carve-outs:

| Module | Why it cannot be a generated analysis |
|---|---|
| `pre_rebalance_checklist.py` | Live safety gate. Reads `os.environ`, the kill-switch file, and market state at call time; returns `blocking=True`. The correct output changes every call — no golden answer exists. |
| `rebalance_trigger.py` | Sets `is_rebalance = True`. It is the execution switch. |
| `weekend_runner.py` | Job scheduler driving ML retraining, factor discovery, self-improve, causal graphs. Retiring it means re-architecting a scheduler. |
| `alert_system.py` | Sends an external liveness ping *so that failure of the channel is detectable*. Comparing it to a historical output is a category error. |
| `forward_pnl_tracker.py` | Live external fetches plus stateful idempotent writes. |

Track B stays hand-written, deterministic, and tested. It may *consume* catalog series; it is
not generated.

---

## 6. The catalog

**Honest scoping.** Rev 1 called it "a lens, not a database." It does not move bytes, but it is
**not small**: 41 distinct `logs/*.jsonl` schemas, dozens of `data_cache/*.parquet` files whose
provenance semantics live in prose, per-date JSON and CSV of differing shapes in
`outputs/debate_log/` and `outputs/wf_results/`, mixed formats in `dashboard/`, and
single-current-state files like `execution/merged_weights.json`.

**That is 6–8 structurally distinct adapter families, each hand-written. 4–6 weeks.** It is the
critical path; everything downstream consumes its schemas.

**The unit is a named series. One name, one series, forever.**

| Field | Why, in this system |
|---|---|
| `name` | Canonical address. `counterfactual.track_b` exists once, not five times. |
| `description`, `tags`, `units` | Search surface. |
| `index_kind` | `market_trading_day` \| `calendar_day` \| `utc_timestamp`. Kills the Track B off-by-one and the Juneteenth row. |
| `value_schema` | Columns and dtypes. |
| `provenance` | **Typed, not a filename.** Today `prices_live` vs `prices_live_fallback_simulated` lives in a string every reader must remember. Typed, an analysis declares "refuse simulated" and the runtime enforces it. |
| `coverage` | e.g. "every NYSE session in range" — the 45-rows-for-78-days bug, caught at generation. |
| `as_of` | Point-in-time semantics so `as_of_join`/`as_of_merge` fire automatically. Those helpers are currently **inert — zero callers outside their own definition and a `verify_docs.py` check that only asserts they are defined.** |
| `lineage` | For derived series: the plan and task that produced it. |
| `adapter` | How to read it from the existing file. |

**Coverage limit, stated honestly:** the catalog serves PAT stage 9 and the *structured* half of
stage 1. Unstructured document search and web search are undesigned. The registration order's
item 5 below is a placeholder, not a design.

**Registration order:**
1. The five counterfactual series (`track_a★`, `track_b`, `track_c`, `track_d`, `nav`)
2. `prices_live` + macro, provenance-typed
3. `eod_log`, `holdings_log`, `ai_pm_decision_log`, `verdicts`
4. Regime, IC, skill scores, weights
5. Unstructured corpus — **design pending**

---

## 7. The AI PM authority decision — argument withdrawn

### 7.1 Current write paths (corrected: three, not two)

| Path | Site | Bound |
|---|---|---|
| `authority_blend(...)` | `run_all_agents.py:1524` | `ai_weight` = 5% |
| `apply_judge_position_change(...)` — scheduled path | `run_all_agents.py:1913` | one change, ≤1.0pp |
| `apply_judge_position_change(...)` — **discovery path** | `run_all_agents.py:2979` | same bound |

### 7.2 Why rev 1's argument fails

Rev 1 argued that demoting to Level 0 *improves measurement* because Track D becomes
unconfounded. **This is false.** `blend()` (`ascent/strategy/earned_authority.py:298`) builds a
new dict and never mutates `ai_portfolio`. At `run_all_agents.py:1533`,
`snapshot_ai_pm(today, dict(ai_pm_result.portfolio))` captures the pure pre-blend vector at any
`ai_weight`. **Track D is already clean at 5%.**

What `ai_weight → 0` actually does: `blend()` short-circuits (`if budget <= 0.0: return
dict(quant_portfolio)`), so **Track B becomes identical to A★ by construction** and stops
carrying information. Demotion removes an observation; it does not clean one.

### 7.3 The evidence is one fix-cycle stale

The instrumentation findings are real and remain recorded (§7.4). But the override-derivation
fix landed **2026-07-28**, and `CURRENT_VERIFIED_NUMBERS.md` states it made
`n_decisions_evaluated` reachable "for the first time." The last rebalance was 2026-07-22; the
next is **2026-08-05**. **No rebalance has run since the fix.** Empty buffers today are expected,
not proof of a broken mechanism.

`CURRENT_VERIFIED_NUMBERS.md` §5 item 4 says explicitly: *"do NOT change the 5% authority budget
on this evidence; re-ask after 8–10 cleanly scored rebalances."* Rev 1 recommended the opposite,
contradicting the project's own source of truth on this exact decision.

### 7.4 What the investigation did establish (unchanged)

- `earned_authority.json`: `track_d_returns: []`, `track_astar_returns: []`, `days_at_level: 19`,
  `days_stuck: 19` — 19 update cycles, zero buffer appends.
- Promotion is gated on `len(d_buf) >= win` (win=21), so the gate dict is never constructed —
  gates were never *evaluated*, not merely failed.
- `logs/ai_pm_decision_log.jsonl`: 9 rows, 8 dated 2026-06-10, `fallback` flipping between
  duplicates. Same failure family as the counterfactual log: append with no idempotency key.
- Demotion fires on a single day (−5pp → down one level, −10pp → Level 0).
- `ai_pm_perf_feedback.py`: *"`min_decisions: n >= 5` is a hard promotion gate while demotion
  needs only one bad day. The ladder could only ratchet downward."*

### 7.5 Revised recommendation

**Hold at 5%. Do not demote.** Let 2026-08-05 run under the fixed pipeline and observe whether
the buffers fill and a real `overrides_applied` entry appears. That is a cheap, decisive test
that falsifies or confirms the premise within one rebalance.

**Do now, regardless:** add the idempotency key on `(date, phase)` to the decision log, and add
a **liveness benchmark** on `earned_authority` that fails when N update cycles pass with zero
buffer appends. The generalized lesson survives intact: *before trusting a governance mechanism,
benchmark that the mechanism is alive.*

**Owner decision required.** This section reverses an approved rev-1 decision. It is also a
different risk class from the rest of this document — a change to live paper-trading behaviour,
not a research-layer refactor — and should be approved separately regardless of which way it goes.

---

## 8. Benchmarks

### 8.1 Three suites converge

| Today | Defends | Runner |
|---|---|---|
| `scripts/verify_docs.py` — **25** checks | claims in `CLAUDE.md` | bespoke |
| `scripts/reconcile_numbers.py --check` | figures in `CURRENT_VERIFIED_NUMBERS.md` | bespoke |
| `tests/` — **147** `test_*.py` files | code behavior | pytest |

An invariant is an invariant. `verify_docs.py` is the embryo.

### 8.2 The bug-class claim, corrected

**In scope and schema-shaped (~5).** These would plausibly have been caught:

| Bug | Invariant |
|---|---|
| 45 rows for 78 trading days | `coverage: every NYSE session in range` |
| Track B keyed one day late | `index_kind: market_trading_day` |
| P&L row on Juneteenth | holiday check on that index kind |
| `prices_live` duplicates (recurred twice) | uniqueness on `(symbol, _calendar_day_key)` |
| Decision log, 8 rows for one date | idempotency key on `(date, phase)` |
| Track A ≡ A★ | `must_differ_from: <baseline>` |

**Out of scope, and already fixed by ordinary means.** `reduce_size` renormalization lives in
`ascent/execution/`; the Sortino ÷ √252 bug in `ascent/research/wf_framework/metrics.py`. §3
declares both areas untouched. Both were fixed with a code change plus a known-value regression
test — no catalog, no compiler. **They are not evidence for this redesign and are removed from
the motivation.** The walk-forward cache-key collision is likewise a logic error no schema catches.

### 8.3 Track-A migration benchmarks

Each Track-A monitoring module is a question with a known output; the compiler must reproduce it
before that module retires. Track-B modules (§5) are excluded.

### 8.4 The public-output determinism gate

`scripts/generate_performance_page.py:847` imports directly from
`ascent/monitoring/ai_pm_counterfactual.py` to build the **public GitHub Pages** output. Putting
nondeterministic codegen upstream of that artifact — already burned once by the −7.82pp episode —
is the sharpest risk in this design, and §1 concedes PAT's 95% determinism figure is unvalidated
outside Bridgewater.

**Hard requirement:** no generated analysis may feed the public dashboard until it has
demonstrated byte-identical output across repeated independent generations, measured, at a
threshold set in §10.1. Until then the public page keeps reading hand-written Track-B code.

---

## 9. Build order — vertical slice first

Rev 1's order (catalog → benchmarks → compiler → migrate → demote) back-loads all validation:
1.5–2 months before any signal that the approach works. Replaced.

### 9.1 Phase 0 — the cheap fix that may be sufficient (days)

**Impose the naming discipline with no compiler at all.** One canonical series name per question;
register the five counterfactual series; delete or alias the redundant artifacts. Add the
decision-log idempotency key and the `earned_authority` liveness benchmark (§7.5).

This directly fixes the *evidenced* pathology of §2.1 — five artifacts, one question — without
LLM codegen, without adapters for 41 log schemas, without a 4–6 month build.

**Gate:** if Phase 0 substantially relieves the pain, the compiler may not be worth building.
That is a real possible outcome and this document should not pretend otherwise.

### 9.2 Phase 1 — one hand-built vertical slice (1–2 weeks)

One adapter (the counterfactual series), one **hardcoded** plan, and the §11 anchor question run
end to end **by hand** — no planner LLM, no codegen, no verify framework.

**Gate:** does the machinery save enough labour on this one real question to justify generalizing?
If not, stop here. This is the earliest honest checkpoint and it costs two weeks instead of two
months.

### 9.3 Phase 2+ — generalize, only if the gates pass

Catalog adapter families → `plan/` → `codegen/` + `verify/` → `runtime/` → Track-A migration.

### 9.4 Sizing

| Component | Solo estimate |
|---|---|
| `catalog/` | 4–6 weeks |
| `plan/` | 2–3 weeks |
| `codegen/` | 3–4 weeks |
| `verify/` | 1–2 weeks |
| `runtime/` | 1–2 weeks |
| `review/` | 1 week + ongoing tuning |
| `report/` | 3–5 days |
| `teach/` | 2–3 weeks, then perpetual |
| `context/` | ongoing, unbounded |

**Working v1: 10–14 weeks FTE ≈ 4–6 months calendar**, split against an internship.

### 9.5 The honest failure mode

Scope creep disguised as "a lens." The most likely bad ending is not a technical dead end: it is
the catalog phase quietly consuming a semester while `run_all_agents.py` still has to work every
day, leaving the project shelved half-migrated with two incomplete monitoring systems side by
side. Phases 0 and 1 exist specifically to make that outcome cheap to detect and cheap to exit.

---

## 10. Open decisions

1. **Determinism bar** — must be set before §8.4 can be satisfied; measurable only once Phase 2 exists.
2. **Benchmark thresholds** for any future authority change.
3. **Unstructured/web search design** (§6 coverage limit) — undesigned.
4. **Whether `factor_discovery` becomes compiler-driven** — deferred.
5. **Clarifying-question quality** — PAT invested heavily here; no benchmark tier exists (§4.2).
6. **Cost and latency of parallel sub-agents** under Claude 5 constraints — unestimated. No
   prototype has run against a single real ascent artifact.
7. **The alpha workstream** — unscoped, and §3 notes the throughput→alpha link is unmeasured.

---

## 11. v1 anchor question

**"Across the last N judge interventions, did the position changes actually help?"** — verdicts,
`outcome_tracker`, versus the counterfactual. Chosen because it is archaeology currently done by
hand and gotten wrong, it needs exactly the §6 tier-1 series, and it exercises every stage.

---

## 12. What survives the audit

| Holds | Does not |
|---|---|
| Canonical naming discipline — fixes the evidenced problem | "A lens, not a database" |
| ~5 schema-shaped bug classes | The other five, out of scope and already fixed |
| The teach loop and unified benchmark suite | "Bounded blast radius" |
| The learning-rate diagnosis | "Demote to Level 0 improves measurement" |
| Track-A/Track-B migration split (new) | "All 29 modules are frozen questions" |
| The liveness-benchmark principle | The Bridgewater org-structure analogy as primary justification |
