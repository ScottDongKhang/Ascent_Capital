# The Ritual and Friction Layer

Real funds run on a fixed clock and are full of imperfect human friction that
clean architectures erase. This document fills that gap and designs the
solo-operator adaptation.

## 1. The daily/weekly/monthly cadence of a real fund

- **~7:00 AM pre-market prep** — PM reviews overnight news, prior day's
  positions.
- **~8:00 AM morning meeting / risk briefing** — a 15-30 minute verbal
  huddle: what contributed/detracted yesterday, what's on watch today
  (earnings, macro releases). Not a document — a huddle.
- **~10:30 AM exposure reassessment** — after the open's initial volatility
  settles, confirm no position carries excessive weight and hedges are in
  place.
- **Intraday flash report** — ad hoc, triggered by an unusual move, not
  scheduled. Closer to a ping than a report.
- **After close, P&L attribution** — not just "what did we make" but *why*:
  genuine edge capture vs. position drift vs. hedging slippage.
- **Weekly investment committee (IC) meeting** — theses debated and sized,
  using a written IC memo as the pre-read.
- **Monthly risk committee meeting** — aggregate exposures, limit breaches,
  model performance review.
- **Monthly/quarterly investor letter.**
- **Annual compliance review** (e.g. Form ADV update) and risk-framework
  re-certification.

**The throughline**: cadence exists independent of whether anything material
happened. A fund holds its Monday IC meeting even in a quiet week — the
ritual itself, forced periodic re-justification of every position, is the
control, not the news that triggers it.

## 2. Real memo structures

**IC memo**: executive summary, thesis (3-5 discrete, quantified drivers, not
vague conviction), sizing rationale tied to a risk budget, risk factors
ranked by severity with explicit mitigants (deal-breakers separated from
manageable), exit plan with scenarios and timeline. The discipline that
matters isn't the template — it's that the memo is written *before* sizing,
forcing the thesis into a form checkable later against what actually
happened (the same pre-registration discipline `10`'s Stage 2 already
covers).

**Monthly investor letter**: performance vs. benchmark, named
contributors/detractors (not just aggregate return), market
outlook/positioning, and — in the better letters — explicit discussion of
what assumptions the thesis depends on, rather than a prediction dressed as
a forecast. The genre convention is candor about *process* over false
precision about outcomes.

## 3. Real, imperfect friction clean architectures erase

- **The "gut call" ledger** — PMs override models constantly; disciplined
  shops log the override *separately* so its track record can be scored
  later. Ascent already applies this pattern to its judge/earned-authority/
  falsifier mechanisms — real funds apply it to a human's unprincipled gut
  call too, not just another algorithm's proposal.
- **Vendor data outages resolved by phone, not code** — a stale feed at 6:45
  AM gets fixed by someone calling the vendor's support line. The "fix" is a
  human relationship, undocumented in any architecture diagram.
- **The junior analyst re-running the broken script** — mundane failures (a
  cron job silently didn't fire, a CSV format changed) get manually
  re-triggered by whoever notices first, often without a postmortem, because
  it "worked."
- **Month-end reconciliation breaks** — routinely take days to chase down; a
  process running past ~10 business days is treated industry-wide as a red
  flag.
- **Key-person dependency** — key-person departures produce average investor
  redemptions around 18%; quant-specific model risk contributed to ~15% of
  hedge-fund closures in 2022. The person who understands why the alpha
  model does what it does is frequently one person, and "what happens when
  they're on vacation and it breaks" is a recurring real failure mode, not
  a hypothetical.
- **Risk committees overriding their own rules under pressure** is a known
  failure pattern — LTCM is the canonical illustration: a firm with real
  risk systems in place still degraded discipline under the pressure of
  mounting losses. A committee with a hard rule can still find a reason in
  the room why "this time is different" — exactly why mechanical shops
  (Millennium-style drawdown ladders) strip discretion out of the trigger
  itself rather than trusting a committee to hold the line under pressure.

## 4. Design proposal for a solo operator + AI agents

The core principle: **a ritual at a real fund does two things — (a) surfaces
information, (b) creates an accountability checkpoint where a specific
human's judgment is on record.** Automation can fully replace (a). It must
not fully replace (b), or the ritual becomes theater.

**Fully automate (agent-generated, no human gate):**
- **Daily flash report** — scheduled agent reads `logs/eod_log.jsonl`, the
  day's `verdict_<date>.json`, kill-switch/correlation-guard state; produces
  a short summary (what moved, what fired, what's on watch). Pure
  information surfacing — no judgment lost by automating it.
- **Weekly IC-memo draft** — agent compiles the week's theses, sizing
  changes, and debate verdicts into IC-memo format. Drafting is mechanical
  compilation of already-logged reasoning.

**Deliberately keep human-in-the-loop, and say why:**
- **The weekly memo needs a human sign-off before it's "the fund's"
  position**, not just an agent's draft. A real IC memo means something
  because a specific person's name is on the sign-off and their judgment is
  on record if it's wrong. An agent that drafts and signs its own memo has
  removed exactly the accountability the ritual exists to create. The
  operator should sign as-is or annotate disagreement — that annotation *is*
  the ritual's real output, more than the memo text.
- **The monthly self-performance-review should be hand-written or
  hand-edited, not agent-generated end to end** — a monthly letter's value
  is candor about process and named contributors/detractors, which an agent
  optimizing for a clean report will tend to smooth over unless a human
  forces the uncomfortable parts back in.
- **Any manual override needs the same counterfactual logging as
  `record_intervention(applied=False)`.** If the operator ever manually
  overrides a system-produced weight, that override needs the same
  advisory-only, logged, scored treatment CLAUDE.md's constraint #5 already
  applies to the judge/earned-authority/falsifier mechanisms — otherwise a
  human override is exactly the same unmeasured live-write risk the
  falsifier trim was, just with a person instead of an LLM behind it.
- **No implicit ability to raise a mechanical limit.** The LTCM lesson isn't
  "have rules" — Ascent already has the 8%/15% kill switch — it's that a
  rule only holds under pressure if what can override it requires an
  explicit, logged, *named* action, not a code path that quietly widens a
  threshold. Any future "risk committee" analog (even a single Claude agent)
  should have zero implicit ability to raise its own limit; only the
  operator, explicitly, logged, can do that.
- **Log "self-healing" events too.** When an agent auto-retries a failed
  fetch or falls back to a cached price, that event should land in a log the
  operator actually reads periodically — not because the retry needs
  approval, but because a pattern of silent retries is the solo-operator
  equivalent of the junior analyst quietly re-running a broken script for
  the third month running — a signal something structural needs fixing,
  easily lost if every instance self-resolves invisibly.

The underlying design rule: **automate the surfacing of information
completely; keep the moment a specific person's judgment is committed to
record deliberately manual**, because that commitment — not the document —
is what a ritual is actually for.
