# The Institutional Gap — Ascent Capital vs. Real Fund Practice

Audit performed by 4 independent research agents (no shared context): one external
literature review, three reading the live codebase directly (not documentation on
trust). Repo state: HEAD `8952198`. Goal under test: not beating the S&P 500, but
maximizing return per unit of volatility.

> **Adversarially verified.** This document was subsequently attacked by two
> independent verification passes: one re-checked every code citation against
> current source (result: 62/71 exact, 5 trivial off-by-one line numbers, 0
> substantive errors in this file specifically — the 2 substantive errors found
> were in other documents in this set, corrected there), one cross-checked the
> external institutional-practice claims against additional sources. The
> corrections from the second pass are folded in below, marked inline.

## Headline verdict

**On the objective, it's real.** Live-date position sizing is a genuine mean-variance
optimization (`α − λ·variance − turnover`, `mvo_optimizer.py:4`), gross exposure is
scaled by a causal realized-vol-targeting overlay on by default at a 15% annualized
target (`exposure.py:81-130`, `settings.py:142-169`), and every sleeve-weight/config
selection in the research layer picks by Sharpe, never by SPY-relative return
(`self_improve.py:281`, `wf_framework/optimizer.py:50`). SPY appears in the code
almost exclusively as a risk-regime feature, not a target.

**On firm structure, it isn't yet.** A real fund's defining feature is an
*independent* risk function that can reject or shrink one specific position without
asking the desk that sized it. Ascent has no such function — every cap lives inside
the same optimizer that builds the book, and the one mechanism that can override the
book (the debate judge's `halt_and_review`) is a binary all-stop, not a scalpel. It
is currently a single-strategy systematic book (`us_equities` only) with a
research/journaling layer bolted on, not a multi-desk platform.

## Scorecard

| Dimension | Institutional standard | Ascent Capital, verified in code | Verdict |
|---|---|---|---|
| **Objective function** | Absolute-return funds target a fixed vol band (8-15%) and optimize Sharpe; benchmark tracking is a separate mandate (Grinold & Kahn; Sharpe 1994). | MVO objective is α−variance−turnover, no benchmark term. Vol-target overlay live by default at 15%, scales gross exposure. Sleeve selection maximizes OOS Sharpe. `mvo_optimizer.py:4`, `exposure.py:81` | **Present** |
| **Portfolio construction** | Mean-variance, risk parity, or Black-Litterman with hard sector/factor caps enforced pre-trade. | Real cvxpy MVO with covariance from a live factor model; water-fill max-weight cap; sector cap with explicit <80%-coverage fallback that never collapses to one name. `optimizer.py:153`, `optimizer.py:691` | **Present** |
| **Circuit breakers** | Staged, non-discretionary drawdown triggers (multi-manager platforms reportedly cut capital ~5% DD, terminate ~7.5% DD — figure consistently repeated across multiple independent practitioner/insider accounts, but no primary or firm-disclosed source exists; treat the qualitative pattern as well-corroborated and the exact percentages as convention-level, not audited fact) that remove judgment at the moment it's least trustworthy. | Real, staged: 8% soft warn, 15% hard stop, blocks order submission, persists trip state, requires explicit clearing. `kill_switch.py:130` | **Present** |
| **Independent risk function** | Three Lines Model: a risk desk separate from the PM, empowered to reject or shrink a *specific* position without PM sign-off (IIA 2020). | No such function. All caps live inside the same optimizer that sizes the book. The only override that reaches live behavior is a binary halt-everything, not a per-trade reject. `central_intelligence.py:826-833` gate unreachable (single agent). | **Absent** |
| **Cross-strategy risk (correlation)** | Concentration/correlation limits enforced across the whole platform's books, not one silo. | Code exists (`correlation_guard.py:80`) and is genuinely correct, but its orchestrator gate requires ≥2 live capital-allocating agents; only one runs. Structurally unreachable today, not broken. | **Unreachable** |
| **Discretionary override discipline** | Any human/AI override of a systematic signal gets logged as a counterfactual (applied vs. not) so its track record can be measured later. | Done well: judge position-change, earned-authority blend, and falsifier trim are all deleted as live-write paths, but all three still compute and log every candidate via `record_intervention(applied=False)`, feeding a scoring ladder. | **Present** |
| **Alpha validation pipeline** | Idea → backtest → OOS/walk-forward → paper trading → staged live capital, with overfitting correction (deflated Sharpe, PBO) before trusting a result (Bailey & López de Prado). | Walk-forward exists and is Sharpe/Calmar-driven (`wf_framework/optimizer.py:33`), but the self-improvement promotion loop is gated off (`SELF_MODIFY_ENABLED = False`). No deflated-Sharpe/PBO correction found anywhere. | **Partial** |
| **Multi-desk / capital allocation committee** | CIO + risk + allocator set capital across multiple independent strategies; no single PM owns the whole book. | Orchestrator is built for this (`central_intelligence.py`) but only `us_equities` is a capital-allocating agent today; macro/international/alternatives exist as full pipelines but run only optionally, for context, via an AI PM tool call. | **Partial** |
| **Compliance / middle office** | Reconciliation of P&L, positions, and collateral independent of the desk (classic post-Madoff control point). | No such independent reconciliation function found in any of the four audit passes. | **Absent** |

## Real governance vs. theater

**Live and reachable:**
- Water-fill max-weight cap — enforced in every construction pass, `optimizer.py:153`
- Sector constraint + <80%-coverage fallback — `optimizer.py:691`, never collapses to one name
- 200MA+VIX exposure cut & vol-target scale — applied to every agent's weights, every run, `exposure.py`
- Kill switch (8% warn / 15% hard stop) — blocks order submission, persists trip state, `kill_switch.py:130`
- Judge's `halt_and_review` — real, live, human-override required to clear, `debate_runner.py:398-411`

**Computed, logged, never acted on:**
- Debate judge's position change — function deleted, zero call sites remain
- AI PM earned-authority blend — function doesn't exist in current code at all
- Falsifier trim — computes the trim, logs "WOULD TRIM (ADVISORY)", submits nothing
- Cross-agent correlation guard — orchestrator gate needs ≥2 agents; only 1 runs
- Regime risk multiplier — no consumer anywhere; wrapped in a bare try/except that silently no-ops

## What the outside literature says a fund actually is

**Three Lines Model** (IIA, 2020 update of 2013 original). First line (PMs/traders)
owns risk day-to-day with no unilateral veto over their own limits. Second line (an
independent risk function, reporting outside the PM chain) sets limits and can
override the first line without their sign-off. Third line (internal audit) checks
that the first two actually work. The entire point of "independent" is that risk can
act *without asking the desk it's overseeing* — this is the piece Ascent doesn't have
yet. **Verified correction**: the IIA's own model text uses softer language than a
flat override right — it charges the second line with "expertise, support,
monitoring, **and challenge**," not an explicit unilateral-veto mandate. Independent
governance commentary (Norman Marks; GRCReport) notes real-world second lines often
fall short of true independence because they sit close to, and are compensated by,
the same profit centers they oversee. Read the standard above as the intended design
target, not a guarantee any firm claiming "three lines" actually achieves —
including, eventually, Ascent's own IRM department (see `01_risk_management.md`),
which should not assume shadow-mode logging alone proves independence once promoted
to live veto power.

**Fundamental Law of Active Management** (Grinold & Kahn, 1994/2000): IR ≈ IC ×
√breadth — the standard argument for running many small, weakly-correlated bets
rather than one large one. Ascent's 2-sleeve, single-agent structure is breadth-thin
by this measure; the dormant macro/international/alternatives agents would widen
breadth if promoted to capital-allocating status. **Verified caveat**: this is a
real and well-documented critique in the literature (Clarke/de Silva/Thorley, "How
to calculate breadth"; the "Fundamental Law of Active Management: Redux"
literature), not a minor footnote — breadth has no rigorous, uncontested
measurement, and IC and breadth are **not independent**: widening breadth by
relaxing signal thresholds typically *reduces* IC, and correlated bets don't count
as full independent breadth even if they're nominally separate "agents." Ascent's
own macro/international/alternatives agents are systematic-equity-adjacent
strategies plausibly correlated with `us_equities` through overlapping macro
factors — promoting them would only deliver the claimed IR benefit to the extent
their correlation genuinely stays low, which `04_cio_capital_allocation.md`'s
correlation-gated promotion criteria already anticipates, but is worth stating
explicitly rather than assuming breadth is free.

**Deflated Sharpe Ratio / Probability of Backtest Overfitting** (Bailey & López de
Prado, 2014/2017). A Sharpe ratio picked as "best" across many trials needs
correction for the number of trials run and for return skew/kurtosis before it can be
trusted. No such correction was found anywhere in Ascent's walk-forward or
self-improvement code — Sharpe is used directly as the selection metric. **Verified
caveat**: independent practitioner commentary suggests DSR/PBO are more often
discussed and cited than mechanically enforced as a hard promotion gate even at
sophisticated shops — "most practitioners walk away with the same advice: use more
out-of-sample data, watch your Sharpe ratio deflation, apply the Bonferroni
correction," without necessarily automating it. This doesn't change the finding
(Ascent has no correction at all, automated or manual), but it means the gap is
less anomalous relative to real-world practice than "no institutional fund would
skip this" would imply — it's a real gap against the *documented standard*, a
smaller gap against *typical practice*.

**Vol-targeting as construction, not reporting** (Qian 2005; Bridgewater All
Weather, 1996–). Risk parity's operating principle: scale leverage to hold
realized/forecast volatility near a fixed target, decoupling the return objective
from any benchmark entirely. This is the one dimension where Ascent's code most
closely mirrors the institutional pattern — the overlay is mechanically identical in
spirit, just with SPY (not the book's own realized vol) as the default reference
series. **Verified caveat, and a real risk worth naming since Ascent runs this
live**: vol-targeting is not a free lunch. It is documented in the literature as
structurally **pro-cyclical** — a vol spike (typically a selloff) mechanically
triggers a gross-exposure cut, which can mean selling into the crash and amplifying
it (comparisons to 1987 portfolio-insurance dynamics appear in this literature;
see also Moreira & Muir's NBER work on volatility-managed portfolios, and
MSCI/IASG commentary on risk-parity stress behavior in 2020). Risk parity
strategies underperformed relative to their diversification premise in COVID 2020
when cross-asset correlations broke down. Ascent's own overlay referencing SPY's
vol (not the book's own) is a specific instance of this same mechanism and
inherits the same pro-cyclicality risk — worth a deliberate design review, not
just a footnote about which series is the reference.

**Citation-fidelity note.** An adversarial line-by-line check of all 71 file:line
citations across this document and the six department blueprints found 62 exact,
5 trivial off-by-one line numbers (the cited claim is correct, the line pointer is
off by 1-2 lines — e.g. `mvo_optimizer.py:4` should read `:5`, a docstring blank
line), and 2 substantive errors, both since corrected in their respective files:
`06_judgment_governance.md`'s halt-override claim, and `05_compliance_middle_office.
md`'s "no audit trail exists" claim. See those files for the corrections.

## Notable, unprompted findings

**CLAUDE.md itself is stale in two places.** Its opening claims Phase 2 "requires"
calling all four agents. Current code (`ai_pm_agent.py:972-975`) makes
macro/international/alternatives optional, context-only — changed deliberately in
commit `be7d870`, doc never updated. Separately, the falsifier-trim removal is dated
2026-08-14 in CLAUDE.md but 2026-08-15 in the actual removal comments.

**A judge write-path CLAUDE.md doesn't name.** Constraint #5 lists three
advisory-only mechanisms. It's accurate as far as it goes, but the judge's
`reduce_size` macro verdict does de-gross the live book via `eod_runner.py:476-520` —
distinct from the enumerated `apply_judge_position_change`, so the letter of the
constraint holds, but "the judge is purely inert" overstates it.

**The vol-target reference is SPY, not the book itself.** Production default is
`vol_reference="spy"` (`settings.py:169`) — the de-risking trigger fires off market
volatility, not this book's own realized volatility. A self-referential `"strategy"`
mode exists in code but isn't the default. Worth a deliberate decision either way,
not an oversight to leave silent.

## Sources consulted (external research pass)

1. Grinold, R. & Kahn, R., *Active Portfolio Management*, McGraw-Hill (1994/2000)
2. Bailey, D. & López de Prado, M., "The Deflated Sharpe Ratio," *Journal of Portfolio Management* 40(5), 2014
3. Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q., "The Probability of Backtest Overfitting," *Journal of Computational Finance* 20(4), 2017
4. Sharpe, W.F., "The Sharpe Ratio," *Journal of Portfolio Management* 21(1), 1994
5. Chan, E.P., *Quantitative Trading*, Wiley
6. Institute of Internal Auditors, "The IIA's Three Lines Model" (2020)
7. Qian, E., "Risk Parity Portfolios" (2005); Bridgewater Associates, "The All Weather Strategy"

Internal claims verified against repo HEAD `8952198` by direct source read, not
against CLAUDE.md or memory on trust — discrepancies between the two are called out
explicitly above rather than silently repeated.
