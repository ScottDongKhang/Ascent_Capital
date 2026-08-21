# Alpha Edge Audit — The Missing Due Diligence

Everything else in this document set is governance, risk, and process built
around an alpha signal that had never been separately audited. This is that
audit, reading `ascent/alpha/meanrev.py` and `ascent/alpha/statarb.py`
directly.

## 1. What is Ascent's actual current edge, precisely?

Both live sleeves (`DEFAULT_ALPHA_WEIGHTS = {"meanrev": 0.50, "statarb":
0.50}`, `stack.py:16-19`) are short-horizon price-reversal signals built from
off-the-shelf technical features:

- **`meanrev_alpha`** (`meanrev.py:10-51`): a weighted blend of four
  textbook oversold/overbought indicators — inverted 5-day momentum (35%),
  inverted 20-day z-score (35%), inverted 14-day RSI (15%), inverted
  Bollinger %B (15%) — cross-sectionally z-scored, clipped to ±3. No
  universe filter, no sector control, no vol adjustment.
- **`statarb_alpha`** (`statarb.py:1-188`): sector-relative residual
  reversal across 5/10/21-day momentum horizons (50/30/20 weight),
  vol-scaled by lagged 21-day realized vol, liquidity-dampened by
  dollar-volume rank, with a distressed-name filter (`mom_252d < -65%`
  zeroed) applied downstream in `stack.py:429-438`.

Stripped of code polish: **buy recent losers, sell recent winners, relative
to sector, sized by volatility and liquidity.** The textbook short-term
reversal factor (Jegadeesh 1990), reversal being momentum's mirror image
from the same literature era `10` already cites for momentum (Jegadeesh &
Titman 1993). RSI(14) and Bollinger %B are not proprietary — they're the two
most commonly taught indicators in any retail technical-analysis course.
There is no fundamental data, no alt data, no microstructure signal, no
proprietary universe construction, and no economic rationale offered
anywhere in the code beyond "reversion happens."

## 2. Is this edge well-differentiated, or generic?

Generic — and the doc set's own citations say so. `10`'s Stage 7 cites
McLean & Pontiff (2016): across 97 published variables, returns are 26%
lower OOS and 58% lower post-publication; the same document notes
"mechanical, easily-replicated factors (momentum, reversal) decay faster
from crowding than judgment-intensive factors" per the 2026 crowding
follow-up. Short-term reversal is about as mechanical and easily-replicated
as a factor gets. Nothing in `stack.py`/`meanrev.py`/`statarb.py` claims a
reason this implementation should have residual edge after 36 years of
Jegadeesh (1990) being public. The sector-relative residualization in
`statarb.py` is a legitimate refinement — but a well-known one, not a novel
insight.

## 3. What a real edge audit finds absent

- **Per-sleeve measured IC**: the machinery exists — `stack.py`'s
  `_get_gated_weights` (lines 68-145) reads `logs/sleeve_ic_log.jsonl` and
  gates a sleeve to zero below `IC_GATE_THRESHOLD = -0.005`. But
  `CURRENT_VERIFIED_NUMBERS.md` reports **zero IC figures for meanrev or
  statarb specifically, anywhere** — only blended-portfolio Sharpe (0.41)
  and win rate (52.3%) for the whole 2-sleeve stack. CLAUDE.md's own
  precedent (the fundamental sleeve disabled after measured negative IC;
  macro_agent's IC=+0.0204 at p=0.061 being the *only* sleeve-specific IC
  cited anywhere) shows this measurement discipline exists elsewhere and
  simply hasn't been applied to the two live sleeves.
- **Capacity/crowding, never discussed for these two sleeves specifically**
  — despite Almgren-Chriss and the crowding literature both being cited
  elsewhere in this same document set.
- **Differentiation from a retail implementation: none demonstrated** — no
  ablation exists comparing sector-relative reversal against plain reversal,
  net of costs.
- **The walk-forward number is soft evidence.** Sharpe 0.41, **no WFE
  figure** ("not computable for this artifact"), no DSR, no PBO. A prior
  superseded run's WFE was -0.65 (documented overfit). Sharpe 0.41 for a
  two-factor reversal blend on a 6.5-year window with beta 0.95 to SPY is
  statistically indistinguishable from "priced beta plus noise" without
  overfitting correction — and hasn't been tested against the doc set's own
  cited bar (Harvey/Liu/Zhu's t>3.0 for a new factor to be taken seriously).

## 4. The uncomfortable question, answered plainly

**Yes, this doc set has it backwards.** Twenty-plus documents build Three
Lines risk models, compliance middle-office functions, DSR/PBO promotion
gates, staged capital ramps, kill-switch ladders, and a five-layer alpha
*governance* department — but the alpha itself is two off-the-shelf
technical reversal formulas with no per-sleeve IC reported, no crowding
analysis, no capacity study, no differentiation claim beyond sector-relative
residualization. A real allocator opens due diligence with "walk me through
your edge" and does not proceed to operations until that holds up. Here the
operational scaffolding is extensive and genuinely rigorous in places — but
the edge underneath it is unexamined by the same standard. Perfect
governance around a possibly-crowded-to-zero factor pair doesn't produce a
fund — in the audit's own framing, it produces an expensive way to lose
money slowly with excellent paperwork. Beta 0.95, trailing SPY by -3.62pp,
Sharpe 0.41 with no overfitting correction is at minimum consistent with
"this is beta with a governance costume," and nothing in the doc set rules
that out, because nothing measures it.

## 5. Concrete recommendation for Phase 2

Before building DSR/PBO infrastructure for signals that don't exist yet,
spend the first two weeks of Phase 2 on triage of what's already live:

1. **Compute and publish per-sleeve IC for meanrev and statarb, separately,
   now** — the mechanism already exists; this is a data pull, not research.
2. **Run the crowding-adjusted honesty check**: given McLean & Pontiff's
   findings already cited in `10`, estimate what fraction of the observed
   0.41 Sharpe is plausibly residual beta/regime exposure (beta is 0.95!)
   versus genuine cross-sectional reversal edge. A rolling-beta-adjusted or
   market-neutral variant of the walk-forward backtest would separate this
   cleanly and hasn't been run.
3. **Treat the 2-sleeve reversal pair as the base/beta-like layer it likely
   is**, and say so plainly in CLAUDE.md and CURRENT_VERIFIED_NUMBERS.md,
   rather than implicitly presenting it as validated proprietary alpha
   underneath an elaborate governance stack.
4. **Redirect genuine edge-seeking research toward what `07` already
   scoped**: the repo already has `altdata_alpha.py`, `insider.py`,
   `short_interest.py`, `options_flow.py`, `earnings_tone.py`,
   `narrative_alpha.py` scaffolded but gated to 0% weight. Before spending
   on external data, exhaust what's already coded but unmeasured — several
   of these (insider transactions, short interest, options flow) sit closer
   to the "judgment-intensive, harder-to-arbitrage" end of the crowding
   spectrum the doc set itself flags as more durable than mechanical
   reversal.
5. **Only after (1)-(2) show whether there's a real signal worth
   protecting** does the DSR/PBO/capital-ramp machinery earn its build
   cost — right now it would validate a factor pair already publicly known
   to be crowded, solving the wrong problem first.
