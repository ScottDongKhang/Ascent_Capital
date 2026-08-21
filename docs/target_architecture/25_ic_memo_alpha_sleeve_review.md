# IC Memo — Alpha Sleeve Review: meanrev / statarb at 50/50

> **Post-write update.** This memo's "pause and re-underwrite"
> recommendation was written before the two-week triage it calls for was
> actually run. It has since been run — see `23_empirical_alpha_audit.md`
> and `24_beta_decomposition_analysis.md`. The results make this memo's
> recommendation *more* clearly correct, not less: beta-hedged Sharpe
> computed at ≈−0.19 (not merely lower than 0.41 — negative), and recovered
> historical IC data shows meanrev/statarb are the empirically *weakest*
> measured sleeves in the stack, both declining, with meanrev already
> below the system's own gate threshold at the last measured point. This
> is no longer a memo arguing for caution under uncertainty — the
> uncertainty has been substantially resolved, in the direction this memo
> anticipated.

**To**: Ascent Capital Investment Committee (operator)
**From**: Ascent Capital Investment Committee (operator)
**Re**: Should the two live sleeves continue running at the current 50/50
weighting?
**Date pre-registered**: before any further capital ramp or DSR/PBO
infrastructure spend (21 §5)

---

## Executive summary

I recommend **pause and re-underwrite**: hold current live weights flat
(no increase, no new capital), suspend further DSR/PBO/capital-ramp build
spend, and run the two-week triage in 21 §5 before this position gets
re-sized in either direction. The edge has never been measured at the
sleeve level, and what has been measured — 0.95 beta, 0.41 Sharpe with no
WFE/DSR/PBO correction — is consistent with priced beta wearing a
governance costume (21 §4). That is not evidence of zero edge; it is
evidence I don't know, and I have been sizing as if I did.

## Thesis / current state

What IS the edge, stated precisely: `meanrev_alpha` is a weighted blend of
four textbook oversold/overbought indicators — inverted 5-day momentum
(35%), inverted 20-day z-score (35%), inverted 14-day RSI (15%), inverted
Bollinger %B (15%) (21 §1). `statarb_alpha` is sector-relative residual
reversal across 5/10/21-day momentum, vol-scaled and liquidity-dampened
(21 §1). Stripped of code polish, both are the same trade: buy recent
losers, sell recent winners, relative to sector, sized by volatility and
liquidity — the textbook short-term reversal factor (Jegadeesh 1990), the
mirror image of the momentum factor `10` already cites (Jegadeesh &
Titman 1993). RSI(14) and Bollinger %B are not proprietary; they are the
two most commonly taught indicators in retail technical analysis (21 §1).

Weighting is `DEFAULT_ALPHA_WEIGHTS = {"meanrev": 0.50, "statarb": 0.50}`
(21 §1, `stack.py:16-19`). Measured portfolio-level results: beta 0.95 to
SPY, Sharpe 0.41, win rate 52.3%, trailing SPY by -3.62pp, no WFE figure
computable, no DSR, no PBO (21 §3-4). Per-sleeve IC for meanrev or statarb
individually: **not published anywhere** — `CURRENT_VERIFIED_NUMBERS.md`
reports only the blended two-sleeve Sharpe and win rate (21 §3). The
gating machinery to compute and act on per-sleeve IC already exists
(`_get_gated_weights`, `IC_GATE_THRESHOLD = -0.005`, reading
`logs/sleeve_ic_log.jsonl`; 21 §3, 15) — it has simply never been pointed
at these two sleeves. The doc set's own citation, McLean & Pontiff (2016),
found returns 26% lower OOS and 58% lower post-publication across 97
published factors, with mechanical/easily-replicated factors like reversal
decaying fastest from crowding (10 Stage 7, cited in 21 §2). Short-term
reversal is 36 years public. Nothing in the code claims a reason this
implementation should retain edge after that much daylight (21 §2).

## Sizing rationale

50/50 is not defensible on the evidence I actually have, for a specific
reason: it was set without ever separating the two sleeves' contributions,
so I cannot say whether either one is pulling its weight or whether both
are simply riding the same 0.95-beta wave in different technical clothing.
A weighting decision requires knowing what you're weighting. Right now I
know the blend's aggregate Sharpe and beta, not the sleeves' individual
IC, correlation to each other, or capacity.

I am **not** recommending a re-weighting today (e.g., 70/30, or adding a
third sleeve) — that would repeat the same mistake in a different ratio,
sizing on vibes instead of measurement. The correct sizing action right
now is temporal, not proportional: freeze the 50/50 split as-is, do not
scale gross exposure up, and make the *next* sizing decision — whatever it
turns out to be — contingent on the per-sleeve IC pull that 21 §5 already
scopes as a two-week, no-new-research task using machinery that exists.

## Risk factors, ranked by severity

1. **Beta-dressed-as-alpha (deal-breaker if confirmed).** 0.95 beta with
   0.41 Sharpe and no overfitting correction is statistically
   indistinguishable from "priced beta plus noise" (21 §3). If the
   rolling-beta-adjusted backtest in 21 §5.2 shows most of the 0.41 Sharpe
   evaporates once beta is stripped out, this is not a risk to manage —
   it is confirmation there is no sleeve-level edge to size at all, and
   the position should go to zero, not get trimmed. Mitigant: the
   market-neutral variant of the walk-forward backtest (21 §5.2) resolves
   this directly and is scoped as a two-week task, not a research program.
2. **Crowding/decay risk (deal-breaker on a long horizon, manageable
   short-term).** Textbook, mechanical, 36-years-public reversal factor,
   exactly the category McLean & Pontiff flag as decaying fastest (21 §2,
   10 Stage 7). Even if today's IC comes back positive, a mechanical
   factor this generic has a shrinking, not stable, half-life. Mitigant:
   short-term — nothing, because it is intrinsic to the factor's nature.
   Long-term — the redirection in 21 §5.4 toward already-scaffolded,
   judgment-intensive sleeves (insider transactions, short interest,
   options flow — currently gated to 0% weight) that sit further from the
   crowded end of the spectrum.
3. **Unmeasured-IC risk (manageable, and cheap to fix).** This is the
   condition this memo exists to correct, not a standing structural risk
   — the fix is a data pull against machinery that already exists
   (`_get_gated_weights`, `sleeve_ic_log.jsonl`; 21 §3). Mitigant: the
   two-week triage in 21 §5, items 1-2.
4. **Governance-outpacing-edge risk (manageable, reputational/process,
   not P&L).** Twenty-plus documents built Three Lines risk models, DSR/
   PBO gates, staged capital ramps, and kill-switch ladders around a pair
   of sleeves that were never separately audited (21 §4). Not dangerous to
   capital directly, but it means every "the system is rigorous" claim
   this doc set makes has been unearned at the one layer that matters
   most. Mitigant: this memo, plus 21 §5's explicit instruction to state
   the sleeve pair's status plainly in CLAUDE.md and
   CURRENT_VERIFIED_NUMBERS.md rather than implying validated proprietary
   alpha under an elaborate governance stack.
5. **Single-mechanism-of-correction risk (manageable).** The only
   automatic guard against a decayed sleeve is `IC_GATE_THRESHOLD =
   -0.005`, and it has never fired for these two sleeves because it has
   never been fed real per-sleeve IC. A threshold that has never been
   tested against live data it was built for is not proven protection.
   Mitigant: once §5.1 populates `sleeve_ic_log.jsonl` with real numbers,
   confirm the gate actually fires in a controlled backtest before trusting
   it live.

## Exit plan

This is not a new position, so "exit" means de-weighting or zeroing a
sleeve already live, tied to the same DSR/PBO/sleeve-IC gates already
designed in `02` and `15`:

- **Now → +2 weeks**: run 21 §5 items 1-2 — publish per-sleeve IC for
  meanrev and statarb separately, and run the rolling-beta-adjusted
  walk-forward to isolate residual reversal edge from beta exposure. No
  weight change during this window; this is measurement, not trading.
- **If per-sleeve IC comes back at or below `IC_GATE_THRESHOLD` (-0.005)
  for either sleeve, sustained over the existing gate's rolling window**:
  that sleeve zeroes automatically via `_get_gated_weights` — this is not
  a new rule, it is the existing mechanism finally being fed real data.
  No committee vote needed; the gate is mechanical by design (18 §4, the
  LTCM lesson — a threshold that requires discretionary sign-off to fire
  is not a threshold).
- **If beta-adjusted Sharpe collapses toward zero for the blended pair**
  (i.e., most of 0.41 was beta, not reversal edge): reduce combined
  meanrev+statarb gross weight toward a beta-neutral or near-zero
  allocation within 2 weeks of that result, and redirect freed capital
  toward the DSR/PBO-gated pipeline once a genuine candidate sleeve
  (insider/short-interest/options-flow, currently 0%-weighted per 21 §5.4)
  clears Gate 1 (DSR-p < 0.05 AND PBO < 20% AND WFE ≥ threshold, per `02`).
  Do not raise the reversal pair's weight to fill the gap in the interim.
- **If per-sleeve IC is measured and comes back positive and
  statistically distinguishable from beta** (the outcome this memo does
  not predict, but must leave room for): 50/50 is provisionally
  reaffirmed, but only pending the crowding-adjusted honesty check (21
  §5.2) and a stated capacity ceiling — a positive IC today doesn't waive
  the crowding-decay risk ranked #2 above.
- **Hard stop**: if the two-week window in 21 §5 slips past 4 weeks
  without per-sleeve IC being published, that slippage is itself a
  decision — it means live capital is running on an unmeasured signal by
  choice, not oversight, and I log that explicitly as a manual override
  under 18 §4's rule (same counterfactual logging as
  `record_intervention(applied=False)`), not silently.

## Final recommendation

**Pause and re-underwrite** — not proceed as-is, not proceed with an
immediate re-weighting. Proceeding as-is means continuing to size a
governance-heavy stack around two textbook reversal formulas that have
never had their sleeve-level edge separated from beta, which the doc
set's own citations (McLean & Pontiff) say is exactly the profile most
likely to be crowded to nothing. Changing the split today, in either
direction, without per-sleeve IC would just be a different unmeasured
guess. The two-week triage in 21 §5 is cheap, uses machinery that already
exists, and turns "I believe this works" into "I measured whether this
works" — which is the entire point of running an IC memo process at all.
