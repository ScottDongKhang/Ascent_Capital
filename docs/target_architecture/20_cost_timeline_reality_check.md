# Cost & Timeline Reality Check — This Is a Solo, Part-Time Project

## 1. Phase-by-phase effort, converted to real time

S/M/L in `11_transformation_plan.md` describes engineering complexity, not
calendar time for one person working part-time around an already-running
system. Converting with "S = 1 focused weekend, M = 2-3 weekends over a
month, L = a multi-month effort with real re-familiarization cost each
return":

| Phase | Plan's tag | Realistic effort | Note |
|---|---|---|---|
| 0 — Housekeeping | S ×4 | **1 weekend total** | Item 3 (collapsing order paths) touches the live path — needs a paper-trading verification cycle, not just review |
| 1 — IRM shadow mode | M-L (MRR alone = S) | **MRR alone: 1 weekend. Full IRM: 4-6 weekends over 2 months** | The skeleton doc already found the crash risk it targets is partially guarded — even the "S" piece is smaller than advertised once the code is actually read first |
| 2 — DSR/PBO | M | **2-3 weekends, but only after an unlisted prerequisite** | The plan calls the optimizer signature change "small, low-risk" — the skeleton doc contradicts this: `FoldResult` structurally can't carry per-trial distributions today, so this is a real walk-forward internals refactor |
| 3 — Compliance gate + reconciliation | M | **Gate: 1-2 weekends. Reconciliation: 2-3 weekends** (the plan's own text calls this "genuinely novel infrastructure") | Two different sizes bundled under one "M" |
| 4 — CIO / second strategy | L | **8-12 weekends, non-contiguous** | Correctly sequenced last; undersells the *decision* work (real correlation computation, a promotion threshold, closing `alternatives_agent`'s measurability gap first — unscoped anywhere) |
| 5 — Compliance remainder | S-M | **1-2 weekends** | Cheapest substantive phase — audit trail already ships in Phase 0, data integrity wraps an existing script |
| 6 — Governance bug fix | S | **Half a day, possibly less** | The dedicated checklist already narrows this to one call site with an exact diff derived — close to the cheapest item in the whole plan |
| 7 — Lifecycle back half | S ×3 | **Item 3: 1 evening. Items 1-2: already done, 0 hours** | The biggest estimate correction in the plan — items 1-2 are already live in `stack.py`, found by the plan's own later audit |

**Total honest estimate, all 8 phases, solo/part-time: ~20-28 weekends over
6-9 months** — not a sequenced backlog a full-time team would read it as.
`11_transformation_plan.md` never states this number anywhere.

## 2. True critical path vs. the stated order

- **Genuinely blocking**: Phase 0 item 3 (collapse order paths) blocks
  Phase 3 exactly as the plan says — building a new gate against two
  divergent order paths means building it twice.
- **Falsely sequential, actually independent**: Phase 6, Phase 7 item 3, and
  Phase 1's Model Risk Reviewer have zero dependency on each other or on
  Phase 0. The plan's order reads as a pipeline; for a solo operator the real
  constraint is "what can I finish in one sitting without re-deriving
  context" — Phase 6 (one call site, exact diff already written) answers
  that better than its position-6 ranking suggests.
- **Understated prerequisite**: the DSR/PBO module is *not* self-contained
  as claimed ("testable independently of the rest of the pipeline") — its
  own skeleton doc shows `ParameterOptimizer` must change first.
- **Correctly deferred**: Phase 4 — the only phase requiring
  irreversible-feeling capital-allocation judgment plus an unscoped
  prerequisite measurement fix.

## 3. Recurring LLM cost — the plan's real blind spot

Per `ascent/llm/client.py` pricing: Opus 5 = $5/$25 per MTok (in/out);
Sonnet 5 = $3/$15 per MTok (verified against `ascent/llm/client.py`; Sonnet 5
technically carries a $2/$10 introductory rate through 2026-08-31, but the
codebase deliberately prices at the higher standard rate so cost estimates
never understate the bill — the $15-45/month figure below is therefore a
conservative upper bound). The system already runs both on every scheduled
rebalance day. **Neither Phase 1 nor Phase 6 is described anywhere as an LLM
cost line item, but Phase 1's CRO/IRM roles are LLM-backed by design.**

Order-of-magnitude for a daily IRM pass (5 roles, Sonnet-tier per `19`'s
breadth-vs-judgment split): ~8-15K input + ~2-5K output tokens per role call
≈ $0.05-0.15/role ≈ **$0.50-1.50 per full IRM pass**. Daily cadence: **~$15-45/
month**. Weekly cadence (matching the existing rebalance calendar, the more
realistic fit): **~$2-6/month**.

Not large in absolute terms, but new, recurring, and currently unbudgeted
anywhere in the plan — worth naming explicitly given this documentation
effort has itself already consumed real usage.

## 4. Minimum-viable "ready to run" cut — 4-6 weekends

**In scope:**
- Phase 0, all 4 items (1 weekend)
- Phase 6, the bug fix (0.5 weekend)
- Phase 7 item 3 only, rejected-hypothesis log (0.5 weekend)
- Phase 1's Model Risk Reviewer alone, shadow mode (1 weekend — directly
  targets the `prices_live` corruption class that's recurred three times)
- Phase 5's `data_integrity.py` wrapper (0.5-1 weekend)
- Phase 3's compliance gate only, not reconciliation (1 weekend)

**Explicitly aspirational / long-term, not in the near-term cut:**
- Phase 2 (DSR/PBO) — real prerequisite work makes this multi-weekend
- Phase 3's reconciliation — genuinely novel, budget separately
- Phase 1's remaining IRM roles
- Phase 4 in its entirety
- Phase 7 items 1-2 — moot, already shipped

## 5. The over-planning risk, stated plainly

Sixteen-plus planning documents (00-19, plus README), 3,300+ lines, and zero
lines of Phase 0-7 implementation code exist as of this writing. Two
documents (14, 15) already found the plan's own effort estimates and problem
statements were wrong once someone actually read the current source. **The
marginal planning document is now finding bugs in earlier planning documents
faster than it is finding new work to plan** — the exact shape of
over-planning as procrastination. Every hour spent re-deriving what a
skeleton corrected about an earlier estimate is an hour not spent shipping
the one-line CLAUDE.md fix that's been ready since document 12.

**Concrete recommendation**: after this synthesis round, stop writing
planning documents. Pick the Phase 0 four-item weekend — the cheapest, most
fully-specified, zero-ambiguity unit in the entire set — and set a hard ship
date of the next available weekend, with the explicit rule that no new
planning doc gets written until that PR merges. If the instinct to write
another document shows up before then, redirect it into the actual diff for
Phase 0 item 2, already specified down to the line number in
`12_phase0_execution_checklist.md`.
