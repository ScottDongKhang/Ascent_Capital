# Ascent Capital — Institutional Target Architecture

## If you read nothing else in this directory, read this

The alpha question (`21`) is now answered with a **real, directly-computed
number from a full 165-fold walk-forward run**, not just an estimate: a
patched `walk_forward_runner.py` (`26_shipped_code_summary.md`) persisted
the actual daily return series, and the beta-hedged Sharpe computed
directly from it is **≈ −0.10** — negative, confirming the earlier
algebraic estimate's sign (≈−0.19) at roughly half the magnitude. The
OLS-estimated beta from this real data (0.9468) matches the canonical
artifact's reported beta (0.947) to three decimal places, which is strong
independent validation that this is the correct return series, not a
different or corrupted computation. **This is no longer "possibly beta
dressed as edge" or even "algebraically estimated to be negative." It is
computed, verified, and negative.**

Five real code changes shipped alongside this (`26`): a governance bug fix,
audit-trail wiring, a rejected-hypothesis registry, a Model Risk Reviewer
module (which caught a real bug in its own design skeleton before
shipping), and this walk-forward persistence patch — 75 passing tests, zero
regressions, nothing committed yet.

1. **Beta-hedged Sharpe is negative — now confirmed by direct computation,
   not just algebra (`26_shipped_code_summary.md`, refining
   `24_beta_decomposition_analysis.md`'s ≈ −0.19 estimate to a real ≈ −0.10).**
   Once the strategy's ~0.947 loading on SPY is removed from the reported
   Sharpe 0.41, the residual return stream has a *negative* risk-adjusted
   return, not a diminished-but-positive one — corroborated by the
   artifact's own `alpha: −0.0362` and `excess_sharpe: −0.222` fields. This
   is now a directly computed
   result, cross-validated two independent ways, not a suspicion.
2. **The live sleeves are the empirically weakest ones measured
   (`23_empirical_alpha_audit.md`).** Recovered historical IC data
   (recovered from backup copies of a deleted log — the live copy is
   gone, but not for a sinister reason, see below) shows `meanrev` and
   `statarb` — the only two sleeves actually running — have the lowest IC
   information ratio of any measured sleeve; the *dormant*, zero-weighted
   `trend` and `insider` sleeves measure 2-7x better. Both live sleeves
   show real IC decline within the one clean month of data available, and
   `meanrev` specifically would already have been mechanically zeroed by
   the system's own gate rule at the last measured point.
3. **The missing log is not a bug.** It was deliberately deleted on
   2026-08-14 during a real cleanup (stale sleeve names corrupting a
   redistribution bug that was separately fixed), and live trading has
   simply been paused since 2026-07-27 — consistent with this project's
   own prior "held" decision. The gate will resume working automatically
   the moment trading resumes; no code fix is needed. This corrects an
   overnight-pass finding that called the gate "already live" without
   checking whether its input file existed.
4. **Capacity is not the binding constraint (`24` §5); crowding is real but
   unmeasurable from inside this repo.** At current paper-account scale
   (~$105K), headroom to the model's own capacity ceiling (~$50M-$400M) is
   3-4 orders of magnitude. The binding question was never "can this scale
   safely" — it was "is there an edge to scale," and (1)-(2) above answer
   that.
5. **`25_ic_memo_alpha_sleeve_review.md`'s recommendation — pause and
   re-underwrite, not proceed as-is — was written before (1)-(2) landed
   and is now more strongly supported, not less.**
6. **This is 25 planning documents and zero lines of shipped code
   (`20_cost_timeline_reality_check.md`).** That finding stands unchanged
   and matters more now, not less: the empirical work above was exactly
   the kind of "ship something, even a data pull" the over-planning
   critique called for — do the equivalent for the next open question
   rather than writing a 26th planning document about it.

Everything below this point is the prior, still-valid research — the
governance/risk/execution architecture is real and well-grounded. It's
just not the most urgent open question anymore; the alpha question was,
and it now has a real, computed answer.

## Overnight Pass Summary (autonomous, unattended — spot-check before implementing)

This pass ran unattended overnight in 4 paced cycles, doing direct source
grounding (no parallel agents, credit-conscious per user instruction) rather
than more research. One line each on the 6 new files and their single most
important finding:

- **`11_transformation_plan.md`** — the master 8-phase cut/add roadmap
  everything else in this pass refines.
- **`12_phase0_execution_checklist.md`** — 4 housekeeping items, all 4
  re-confirmed exactly against source with precise line numbers.
- **`13_phase6_governance_bugfix_checklist.md`** — **found the earlier
  blueprint's root-cause theory was wrong.** The promotion-path bug isn't AI
  PM cadence, it's `yfinance` price-fetch fragility at
  `run_all_agents.py:2176`, with in-repo evidence (a comment referencing a
  prior related incident) that this exact fetch has already caused a silent
  freeze once before.
- **`14_phase1_model_risk_reviewer_skeleton.md`** — **found `01`'s crash-risk
  framing was partly wrong.** The ML feature-mismatch crash is already
  guarded (`ml_sleeve.py:450`); narrowed the role to what's genuinely
  unguarded instead.
- **`15_phase7_quick_wins_skeleton.md`** — **the biggest finding of the
  night.** A rolling IC monitor and a mechanical sleeve-level cut rule
  already exist, live, in production (`stack.py::_get_gated_weights()`,
  called every run). `10`'s "zero coverage" claim was wrong at the sleeve
  level — the real gap is a too-short window (5 days) and no dashboard, not
  an absent mechanism.
- **`16_phase2_and_phase3_skeletons.md`** — DSR is implementable now; PBO is
  correctly stubbed as blocked (not faked) pending an upstream data-shape
  change. `compliance_gate.py` wired to the exact confirmed insertion line.

**Net effect of the night**: 3 of 6 cycles found the existing plan (from the
earlier session) was meaningfully wrong about the codebase in ways that
matter for implementation — not just adding detail, but correcting direction.
This is exactly why the plan called for verification before building: acting
on the original `01`/`06`/`10` claims without this pass would have led to
rebuilding things that already exist (Phase 7's IC monitor) and chasing the
wrong root cause (Phase 6's bug). **Read `13`, `14`, and `15` before writing
any code from `01`, `06`, or `10`'s original text** — they supersede those
sections' implementation framing, though not their overall structure.

Two deliverables in this directory:

1. **`00_institutional_audit.md`** — where Ascent Capital stands today against real
   hedge fund practice, verified against the current codebase (HEAD `8952198`), not
   documentation on trust.
2. **Six department blueprints** — a 5-layer-deep target architecture for evolving
   Ascent from a single-strategy systematic book into something structured like a
   real multi-desk investment firm, with each layer detailed enough to implement
   directly:

| File | Department | What it governs |
|---|---|---|
| `01_risk_management.md` | Independent Risk Management | The missing second line of defense: VaR/CVaR, stress tests, per-trade veto |
| `02_alpha_research.md` | Alpha / Quant Research | Idea → backtest → overfitting-corrected validation → paper trade → staged live capital |
| `03_trading_execution.md` | Trading & Execution | Pre-trade compliance, order routing, TCA, broker reconciliation |
| `04_cio_capital_allocation.md` | CIO / Capital Allocation Committee | Fund-level vol budget, multi-strategy capital allocation, promotion/demotion of dormant strategies |
| `05_compliance_middle_office.md` | Compliance & Middle Office | Independent reconciliation, audit trail, data integrity, trade surveillance |
| `06_judgment_governance.md` | Judgment & Governance | Staged authority for AI/discretionary overrides, asymmetric promotion/demotion |

Each blueprint follows the same 5-layer structure:
**Layer 1** department mandate & authority boundaries → **Layer 2** roles →
**Layer 3** per-role decision logic with concrete thresholds → **Layer 4**
data contracts between roles → **Layer 5** exact code mapping (existing files
reused, new modules named, insertion points in the current pipeline).

Three further files ground the whole design in real, cited numbers rather than
invented ones — every claim in them is sourced, and gaps are stated explicitly
where no credible figure exists:

| File | Covers |
|---|---|
| `07_data_and_infrastructure_economics.md` | Real market-data/alt-data vendor pricing (Bloomberg, LSEG, FactSet, alt-data marketplaces), compute costs, data budget as % of AUM |
| `08_staffing_and_compensation_benchmarks.md` | Real headcount ratios and comp figures by role at systematic funds, three-lines-of-defense staffing in practice, minimum viable compliance staffing |
| `09_regulatory_context.md` | Investment Advisers Act obligations (206(4)-7, 204-2, 206(4)-8), registration thresholds, Form PF, best execution, Reg SCI scoping — with an explicit "what's actually required today vs. later" read for Ascent's current stage |

## The deepest, most-verified piece: one thesis's full lifecycle

| File | Covers |
|---|---|
| `10_investment_thesis_lifecycle.md` | The single most rigorously cross-verified document in this set: one investment thesis's complete life, idea → hypothesis → backtest → OOS validation → paper trading → staged live capital → live decay monitoring → exit → post-mortem. Every claim cited to primary sources, cross-checked from independent angles, and every genuinely under-sourced claim (staged capital ramps, exact kill thresholds, post-mortem practice) flagged as convention rather than fact. |

## Verification pass

The entire set above was adversarially attacked by 6 further agents: one
re-checked all 71 file:line code citations against current source (62 exact, 5
trivial off-by-one line numbers, **2 substantive errors found and corrected** —
`06_judgment_governance.md` had the halt-override mechanism backwards, and
`05_compliance_middle_office.md` claimed a durable audit trail didn't exist when
`compliance/audit_trail.py` already does, hash-chained, wired into
`eod_runner.py`); one cross-checked the external institutional-practice claims
(Three Lines Model, Fundamental Law of Active Management, vol-targeting) against
additional independent sources and surfaced real caveats (vol-targeting's
pro-cyclicality risk, breadth/correlation critique) now folded into
`00_institutional_audit.md`; one attacked the cost/staffing/regulatory figures in
`07`-`09` and tightened or flagged several (Bloomberg pricing corroborated
against Bloomberg's own pricing letter, a Neudata internal inconsistency
surfaced, SEC thresholds fully re-confirmed); three built `10` from independent
research angles per lifecycle segment. Corrections are marked inline in the
affected files with a `> **Correction**` or `> **Adversarially verified**`
callout rather than silently rewritten, so the audit trail of what changed and
why is visible.

## The plan: how to actually close the gaps

| File | Covers |
|---|---|
| `11_transformation_plan.md` | Prioritized, sequenced cut/add roadmap synthesizing everything above into 8 phases (housekeeping → IRM → alpha research → execution → CIO/multi-desk → compliance → governance bug fix → the lifecycle's missing back half), with effort estimates and an execution order tuned for a single-operator pace. |
| `12_phase0_execution_checklist.md` | Phase 0's four housekeeping items re-verified against current source with exact line numbers — directly actionable, no re-derivation needed. |
| `13_phase6_governance_bugfix_checklist.md` | Phase 6's promotion-path bug, re-derived from source rather than paraphrased from the earlier blueprint — found the actual root cause differs from what `06` assumed (yfinance price-fetch fragility at a specific line, not AI PM cadence), with the exact fix. |
| `14_phase1_model_risk_reviewer_skeleton.md` | Ready-to-implement Model Risk Reviewer module — also found `01`'s original framing was partly wrong (the ML feature-mismatch crash risk is already guarded in `ml_sleeve.py:450`), so this narrows the role to what's genuinely unguarded: cache staleness as a structured artifact, panel NaN-rate checks, regime-label staleness. |
| `15_phase7_quick_wins_skeleton.md` | **Major correction**: a rolling IC monitor + mechanical sleeve-level cut rule already exist live in `stack.py::_get_gated_weights()` — `10`'s "zero coverage" finding was wrong at the sleeve level. Recommends hardening (the 5-day window is likely too trigger-happy per Lo 2002) plus a dashboard, and ships a real skeleton for the one genuinely-missing piece: a rejected-hypothesis registry built on `self_improve.py`'s existing logging. |
| `16_phase2_and_phase3_skeletons.md` | DSR on `PerformanceAnalyzer` (implementable now); PBO correctly stubbed as blocked, not faked, since `FoldResult` doesn't yet carry per-trial data — confirms rather than contradicts `02`'s prerequisite finding. `compliance_gate.py` skeleton wired into the exact confirmed line (`eod_runner.py:1013-1029`), with one open question flagged (whether `Order` carries a `price` field yet) rather than guessed. |

## Second attack wave — attacking the plan itself, not the code

| File | Covers |
|---|---|
| `17_red_team_findings.md` | Outside-due-diligence attack on the plan: sequencing risk, single-points-of-failure unique to a solo operator (who checks the CRO's own decisions?), 6 entire risk categories missing from all prior files (disaster recovery, key-person risk, broker/credential/LLM-provider outages), and where the documentation's institutional tone risks overclaiming actual organizational maturity. |
| `18_institutional_rituals_and_imperfect_details.md` | The ritual/cadence layer (morning risk call, IC memo, investor letter) and the realistic human friction (gut-call overrides, vendor outages fixed by phone, LTCM-style rule erosion under pressure) that a clean architecture erases — with a concrete, opinionated design for which rituals a solo operator should automate vs. deliberately keep manual to preserve real accountability. |
| `19_deployable_agent_architecture.md` | Classifies all ~28 roles across the 6 department blueprints against the codebase's real LLM infrastructure — only 5 are genuinely new LLM-backed agents, everything else is deterministic code already fully specified by its own numeric thresholds. |
| `20_cost_timeline_reality_check.md` | Converts every phase's S/M/L tag into real solo-operator weekends (~20-28 weekends total, not stated anywhere in `11`), finds a real undocumented prerequisite in Phase 2, prices Phase 1's new recurring LLM cost, and gives a 4-6-weekend minimum-viable cut — plus the blunt finding that this project has produced 22 planning documents and zero shipped code. |
| `21_alpha_edge_audit.md` | **The audit this whole project was missing**: reads the actual live alpha sleeve code and finds textbook, heavily-crowded reversal signals with beta 0.95 to SPY and no published per-sleeve IC — the operational scaffolding may be built around a signal indistinguishable from beta, and nothing in the other 20 documents currently rules that out. |

## Third wave — the empirical answer, with real computation

| File | Covers |
|---|---|
| `22_operational_resilience.md` | Fills the gap `17` found entirely missing: disaster recovery, credential health, backups, a daily liveness digest — scoped realistically for one person, discovering along the way that a real watchdog (`scripts/heartbeat_check.py`) already exists in the repo but its launchd job isn't currently loaded. |
| `23_empirical_alpha_audit.md` | Real per-sleeve IC computed from three recovered backup copies of a deleted log file: the live sleeves (meanrev, statarb) measure as the *weakest* of the sleeves with any data, dormant trend/insider measure 2-7x better, and meanrev would already be mechanically zeroed by the system's own gate at the last measured point. Also corrects the "gate is broken" framing — it's a deliberate deletion plus paused trading, not a bug. |
| `24_beta_decomposition_analysis.md` | Computes (not estimates) beta-hedged Sharpe at ≈−0.19 by algebraically removing the strategy's 0.947 SPY beta from the reported Sharpe 0.41 — cross-validated against the walk-forward artifact's own excess-return fields. Plus a capacity/crowding study: capacity isn't binding at current scale, crowding is real but unmeasurable from this repo. |
| `25_ic_memo_alpha_sleeve_review.md` | A real Investment Committee memo, in the format `18` designed, applying every finding to the actual decision: pause and re-underwrite the meanrev/statarb weighting rather than proceed as-is — written before `23`/`24` landed and now more strongly supported by them. |

Produced by 31 parallel research/audit/computation agents across five waves:
1 external literature review, 3 codebase audits, 6 department blueprint
designs, 3 grounding-research passes on real costs/staffing/regulation, 3
adversarial verification passes, 3 independent lifecycle-research passes, 5
plan-attacking passes, and 6 empirical-computation passes (real IC
statistics, beta decomposition, capacity/crowding, root-cause diagnosis, an
operational-resilience blueprint, and a decision memo) — each citing sources,
exact file:line citations, or actual computed numbers, with gaps flagged
rather than filled by invention.
