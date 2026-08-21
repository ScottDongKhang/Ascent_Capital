# Due Diligence Report — Attacking the Plan, Not the Code

Scope: all 17 prior files. A skeptical outside due-diligence consultant's
attack on the plan itself.

## 1. Sequencing risk

**CRITICAL** — Phase 4 (the only phase that adds breadth, the plan's own
headline justification via Grinold-Kahn IR ≈ IC×√breadth) is sequenced last.
Defensible risk management, but it means "transformation plan" mostly
transforms governance for 7 phases while the core structural limitation
(single-strategy, breadth-thin) goes untouched.

**HIGH** — real duplication risk: Phase 1's Market Risk Analyst and Phase 4's
Strategy Allocation Analyst both independently propose generalizing
`correlation_guard.py`'s dead `len(agent_weights)>=2` gate — once intra-book,
once inter-strategy — ~3 phases apart, no shared design. Low cost today, real
drift risk by Phase 4.

**MEDIUM** — Phase 0's "all four items are S, mostly deleting false
confidence" framing buries that item 3 (collapsing the two order-submission
paths) is the one item that touches the live order path directly — it needs
a paper-trading verification cycle, not just a code review.

**LOW** — Phase 6's root-cause revision (yfinance fragility, not AI PM
cadence) wasn't propagated back into Phase 1's own justification, which cites
a different, already-partially-fixed failure mode as its motivation.

## 2. Single points of failure — solo operator

**CRITICAL** — no file anywhere describes who or what checks the CRO/IRM's
own decisions. Every "independent" role is code written, reviewed, and
operated by the same single person who writes the alpha, the portfolio
construction, and the AI PM prompts. Real independence requires a different
person with different incentives; here it reduces to "a separate Python
module" — real code-level separation, zero organizational independence. The
same blind spot can be wrong about the model AND the check on the model.

**HIGH** — the CIO Capital Allocation Committee uses real multi-role
governance language ("final sign-off authority," "the only role that can
override") for what is, in practice, one person's config file plus an LLM
call. No mechanism catches correlated LLM failure — if Opus/Sonnet has a
systematic blind spot reasoning about correlation or risk budget, every
"role" inherits it simultaneously, unlike human committees whose members
bring genuinely different priors.

**HIGH** — no coverage plan. A solo operator sleeps, travels, gets sick. No
file addresses who/what handles a halt-requiring event when the operator is
unavailable for hours or days — the book can sit halted indefinitely with no
SLA, on-call, or dead-man's-switch design.

**MEDIUM** — the Gatekeeper/Validation Statistician gates are pitched as
removing "the sleeve's own advocate from the cut decision" — but the same
operator built the sleeve, wrote the gate thresholds, AND can edit the
thresholds. Nothing requires a second sign-off or a change-log specifically
for threshold edits (except the CRO's "limit register... versioned"
language in `01`, the one place this is handled reasonably).

## 3. What's missing entirely — not covered in any of the 21 files

**CRITICAL** — no disaster recovery / business continuity plan anywhere.
What happens if the machine dies mid-run with open orders in flight? Is
`data_cache/`, `logs/`, `earned_authority.json`, `kill_switch_state.json`
backed up off-site? No documented recovery procedure.

**HIGH** — key-person risk is researched as an abstract industry concept
(`08`) but never turned back on Ascent's own single-point-of-failure design.
No policy for "operator unavailable for an extended period."

**HIGH** — single-broker (Alpaca) concentration risk is *named* but never
mitigated — logged as a standing finding and shelved. No discussion of an
Alpaca outage during a rebalance, account freeze, second-broker contingency,
or manual fallback.

**MEDIUM** — no model/config versioning or rollback plan for the LLM layer
itself — what happens on Anthropic model deprecation, behavior change, or an
outage during Phase 2 AI PM synthesis, beyond the existing missing-prethesis
fallback (a data-availability fallback, not a provider-outage contingency).

**MEDIUM** — no incident response process for *operational* (not signal)
failures — a bad fill, duplicate order, a halt that should have fired and
didn't, a credential leak. No severity taxonomy, no required write-up.

**MEDIUM** — no secrets/credential-expiry monitoring anywhere in the doc set,
despite this being one of the single most probable real-world failure modes
for a solo-operator system (and one similar failure — MiroFish/OpenRouter
402 on low credits — has already happened once, per CLAUDE.md).

## 4. Overclaiming vs. actual state

**HIGH** — the five-layer blueprint format ("Layer 1 — Department Mandate,"
"Escalates to") is well-executed but its cumulative institutional aesthetic
across 6 department docs could be mistaken for organizational maturity
rather than code structure, including by the operator's own future self.
Thorough documentation ≠ "this behaves like a fund" — worth stating
explicitly and prominently, not just in the audit's closing line.

**MEDIUM** — "adversarially verified" framing risks over-trusting itself.
Verification here means other LLM agent passes, not an independent human
reviewer or an actual test-suite run. Each pass has found real errors in the
prior pass (2 substantive errors, a wrong root-cause theory, a wrong "zero
coverage" claim) — that pattern suggests convergence hasn't been reached, not
that the current state is "verified" in the reader-facing sense.

**LOW** — "Reversibility is structural, not procedural" (`06`) is stated as
an accomplished property of the whole architecture but is really an
aspiration for the *proposed* new gates — a bad hard-coded threshold change
isn't structurally reversible, only procedurally so.

## 5. Realistic solo-operator failure modes

**CRITICAL** — silent API key expiry/rejection (Alpaca or Anthropic) is
addressed nowhere, despite being one of the single most probable real-world
failure modes for exactly this kind of system. None of the IRM/compliance
roles are scoped to catch auth/billing failures specifically, only
data-quality failures.

**HIGH** — broker outage mid-rebalance is not gamed out; neither is an
unhandled exception mid-order-loop (the existing divergent-exception-handling
issue Phase 0 item 3 is meant to fix makes this worse, and the transformation
plan doesn't specify the new unified error-handling contract for
partial-batch failure).

**HIGH** — LLM provider outage *during* Phase 2 synthesis (not just a missing
prethesis) is unaddressed — does the system retry, fall back to quant-only
weights, or halt? Sits directly on the critical path of every rebalance.

**MEDIUM** — laptop/server downtime, the literal single point of physical
failure, is never addressed beyond a passing mention of `catch_up` runs
after past outages — an artifact of past incidents, not a designed
contingency.

**LOW** — risk coverage is real but heavily skewed toward "clean" financial
risk (VaR/CVaR/DSR/PBO) versus near-total silence on operational risk —
the throughline connecting every finding above.

---

**Bottom line**: documentation quality is genuinely unusual — cited,
adversarially checked, honest in tone about its own gaps. But due diligence
separates "well-written architecture spec" from "operationally resilient
fund," and on that axis the plan is lopsided: sophisticated statistical/risk
design layered onto a single machine, single broker, single LLM vendor,
single human, with no named answer for how that stack actually breaks in
production. The CIO Committee's and IRM's "independence" is real at the
code-separation level and illusory at the organizational level.
