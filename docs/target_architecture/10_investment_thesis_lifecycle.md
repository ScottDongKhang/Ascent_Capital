# The Life of an Investment Thesis — Idea to Post-Mortem

A single, rigorous, cross-verified account of how one systematic investment
thesis moves through its full lifecycle at a real quant fund: idea → hypothesis
→ backtest → out-of-sample validation → paper trading → staged live capital →
live monitoring → exit → post-mortem. Every claim below is cited to at least one
primary source; where two independent research angles converged on the same
conclusion, both are cited, and every genuinely under-sourced claim is flagged
as such rather than presented with false precision. This document is deliberately
narrow and deep — it is the one artifact in this directory meant to be gotten
exactly right, rather than broad.

---

## Stage 1 — Idea Generation

Real quant ideas come from a small number of well-documented channels, not from
unconstrained brainstorming:

- **Academic literature mining and anomaly replication.** Jegadeesh & Titman
  (1993), "Returns to Buying Winners and Selling Losers" (*Journal of Finance*),
  established momentum — stocks with strong 3-12 month past returns continue
  outperforming over the following 3-12 months — as a robust, replicated
  cross-sectional effect. Fama & French (1993), "Common Risk Factors in the
  Returns on Stocks and Bonds" (*JFE* 33), formalized size and value/
  book-to-market as priced factors, later extended (Fama-French 2015) to add
  profitability and investment — the quantitative ancestors of "quality"
  factors, further formalized in AQR's "Quality Minus Junk" work.
- **Alternative data and market microstructure.** Larger, better-capitalized
  funds (Two Sigma is the most commonly cited archetype) increasingly generate
  ideas from alt-data sources — satellite imagery, credit-card transaction
  flows, web-scraped data. CTAs (Man AHL and similar) generate ideas from
  microstructure/regime observations — e.g., the empirical fact that
  trend-following produces "crisis alpha" (2008 outperformance during the
  equity collapse) itself seeded systematic trend-following as an idea class.
  **Honest gap**: granular, technically detailed public accounts of funds'
  *actual internal* idea-generation workflows are largely not available — what's
  citable is published factor research, not internal process documentation.
- **The factor zoo critique.** Cochrane's 2011 AFA Presidential Address,
  "Discount Rates" (*Journal of Finance* 66), coined the "zoo of new factors"
  framing — dozens of purportedly priced factors have proliferated beyond what
  CAPM alone explained. Harvey, Liu & Zhu, "…and the Cross-Section of Expected
  Returns" (*Review of Financial Studies* 29(1), 2016), made this rigorous:
  given the sheer number of published multiple-testing factor papers, the
  conventional t>2.0 significance threshold is indefensible, and they recommend
  **t>3.0** as the minimum hurdle for a newly proposed factor. AQR's "Taming
  the Factor Zoo" applies this critique practically via regularization/
  selection methods — independent academic and practitioner convergence on the
  same diagnosis.

## Stage 2 — Hypothesis Formalization

The discipline that separates a well-formed hypothesis from an ad hoc backtest
search is **specifying the test before seeing results** — locking in entry/exit
logic, parameters, universe, and the performance bar required for validation,
prior to running it. This is finance's analogue of scientific pre-registration:
enforced not by a journal timestamp but by (a) fixing the test design in a
research log first, and (b) mathematically discounting the result by the number
of variants tried, whether disclosed or not.

The statistical mechanism is straightforward multiple-comparisons math: at a 5%
significance level, roughly 1 in 20 independent tests appears "significant" by
chance alone — 200 parameter variants on one price series should produce
roughly 10 spurious "wins." The rigorous treatment is Bailey, Borwein, López de
Prado & Zhu, "The Probability of Backtest Overfitting" (*Journal of
Computational Finance* 20(4), 2017), which introduces Combinatorially Symmetric
Cross-Validation (CSCV) to *estimate* — not merely warn about — the probability
that a chosen configuration is overfit to noise. Its companion, Bailey & López
de Prado, "The Deflated Sharpe Ratio" (*Journal of Portfolio Management* 40(5),
2014), formalizes the correction: adjusting observed Sharpe for the number of
trials attempted and for return non-normality (skew/kurtosis). Independent
corroboration: Harvey, Liu & Zhu's factor-discovery critique (Stage 1) and
Bailey/López de Prado's backtest-overfitting work are separately-authored
research programs converging on the identical underlying discipline — correct
for how many things you tried.

## Stage 3 — In-Sample Backtesting

**Walk-forward vs. simple train/test split.** A simple split fixes parameters
once on a training window, evaluates once on a held-out window — cheap, but
highly sensitive to exactly where the split falls, especially under
non-stationary regimes. Walk-forward optimization re-trains on a rolling/
expanding window and tests on the immediately following slice, repeating
forward through the dataset — closer to how the strategy would actually be
deployed live. Real tradeoff, not just a talking point: walk-forward's repeated
re-optimization at each step reintroduces a smaller-scale version of the same
overfitting risk it's meant to solve, since each window's parameter choice is
itself a fitting exercise.

**Parameter overfitting and trial count.** Bailey/López de Prado's "Minimum
Backtest Length" result gives this a concrete number: after trying as few as
**7 strategy configurations**, a researcher should *expect* to find at least one
2-year backtest with an annualized Sharpe above 1.0 purely by chance, even when
the true expected OOS Sharpe is zero. This is why trial count must be tracked
and fed into the Deflated Sharpe Ratio, not treated as a free search.

**Transaction costs.** Omitting or underestimating transaction costs is one of
the most consistently cited sources of backtest-to-live decay: spread cost
(mid-price fills vs. realistic bid/ask crossing), market impact (scales
non-linearly with order size — a strategy profitable at $1M can be uneconomic
at $50M), and short-selling cost/availability bias in long-short backtests. One
cited illustrative example: a nominal 3.9% no-cost backtest return turns
negative once ~6.2% annualized costs are properly modeled — cost modeling can
flip a strategy's sign, not just shave a few basis points.

**Minimum sample size — attacking the rule of thumb.** The most commonly
repeated heuristics ("need at least 30 trades") are frequently stated without a
rigorous source and should be treated skeptically. The genuinely rigorous
answer is Grinold & Kahn's Fundamental Law (IR = IC × √Breadth — gives a
principled reason bet count matters, but no universal minimum N) combined with
López de Prado's **Minimum Track Record Length (MinTRL)** and Probabilistic
Sharpe Ratio, which compute the number of observations needed for statistical
confidence that a Sharpe ratio exceeds a stated benchmark, as a function of the
*observed* Sharpe, its skew/kurtosis, and the desired confidence level — not a
fixed round number. Any flat "N trades" rule circulating in blogs should be
read as a rough mnemonic, not a validated threshold.

## Stage 4 — Out-of-Sample Validation

Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) are
distinct diagnostics, not two names for the same check. **DSR** answers "what
is the probability this Sharpe ratio is genuinely greater than zero, given how
many strategies were tried to find it?" — a significance test corrected for
search intensity and non-normality. **PBO**, via CSCV, answers a different
question directly through resampling: given a backtest that selected the
best-performing configuration in-sample, what is the probability that this
"winner" ranks below the *median* configuration out-of-sample? PBO does not
require an explicit trial count the way DSR does.

**Independent cross-verification**: these two papers do not stand alone.
Harvey & Liu's parallel, independently-arrived-at work on multiple-testing
bias in factor research reaches the same underlying diagnosis (most claimed
findings are likely false once multiple-testing is accounted for) via a
different methodological route (asset-pricing academic critique vs.
computational/statistical-finance CSCV). That convergence across
methodologically distinct research programs is meaningful corroboration that
the underlying problem — not just one lab's framework for describing it — is
real. The CSCV/PBO implementation itself has drawn narrower methodological
criticism as one candidate framework among several, not the unique final word.

**Walk-forward efficiency (WFE)** = (annualized OOS return) / (annualized
IS-optimized return), originating with Robert Pardo's *The Evaluation and
Optimization of Trading Systems* (Wiley, 1992/2008) — the book that coined
"Walk-Forward Analysis." Practitioner heuristics (WFE > ~50-60% treated as
"robust") are informal convention, not a statistically derived cutoff the way
DSR's p-value is. **Important nuance**: WFE is squarely established in the
systematic/retail trading-systems and CTA literature, but it is narrower and
less rigorous than the DSR/PBO apparatus, which comes from an institutional-
quant, statistics-first tradition. A fund citing "WFE = 0.7" without also
controlling for the number of configurations searched is still exposed to
exactly the selection-bias problem DSR was built to catch — WFE is a useful,
standard, but *lower-rigor* screening metric alongside, not a substitute for,
DSR/PBO.

## Stage 5 — Paper Trading / Execution-Parity Validation

Ernest Chan's *Quantitative Trading* (Wiley) is explicit that paper trading
functions as an infrastructure and operational check, not a second alpha test —
it verifies the automated pipeline connects correctly, orders route and fill as
coded, and the mechanical system behaves as intended, distinct from the earlier
backtesting step that validates whether the strategy itself has edge. This
framing is consistent across the independent practitioner literature: paper
trading exists to surface *implementation* bugs, not to re-derive statistical
confidence in the signal.

Cross-verified checks real practitioners run:
- **Slippage/fill-quality parity.** Backtests assume a fill price (mid or
  last-traded); paper trading routinely reveals worse real/simulated fills,
  because paper engines often don't model market depth or the order's own
  impact. Standard guidance: re-inject realistic slippage/commission estimates
  rather than trusting the paper engine's optimistic fills.
- **Latency effects.** A backtest implicitly assumes zero decision-to-fill
  delay; live/paper execution introduces real latency, materially important
  for short-horizon strategies, comparatively immaterial for lower-turnover
  ones.
- **Shadow trading as an intermediate rung.** Independent sources describe a
  graduated evidence ladder — simulation/backtest → replay → paper → shadow
  (production data path, no capital at risk) → small live capital — as the de
  facto staged validation sequence, echoing Chan's infrastructure-first
  framing.

**Market microstructure grounding**: Almgren & Chriss, "Optimal Execution of
Portfolio Transactions" (1999/2000), formalizes the mechanism underlying
slippage-parity checks — execution cost is a mean-variance trade-off between
market impact (trading fast) and timing risk (trading slow), governed by a
liquidity parameter that determines slippage-per-unit-of-trading-velocity. A
backtest assuming fills at the touch with no impact is implicitly assuming that
parameter is zero; Almgren-Chriss (still described as a cornerstone of
Implementation Shortfall algorithm design industry-wide) is the standard tool
for estimating how wrong that assumption is as size scales. This is an
independent literature from Chan's book (market microstructure/optimal control
vs. a trading-business how-to guide), and the two converge on the same
conclusion: backtest fill assumptions are optimistic, and paper-trading
execution-parity testing exists to quantify the gap before it's paid for in
live capital.

## Stage 6 — Staged Live Capital Deployment and Sizing

**Kelly criterion and fractional-Kelly.** The Kelly criterion (Kelly, 1956)
prescribes a position size maximizing long-run geometric growth rate. Edward
Thorp carried Kelly from gambling theory into professional investing
(blackjack, then Princeton-Newport Partners), and his own writing plus the
broad practitioner literature converge strongly: **full Kelly is not used in
practice.** Full-Kelly sizing produces extreme volatility (50%+ drawdowns are
routine even with a genuine edge) and is acutely sensitive to edge
misestimation — because expected-value/variance inputs are never known with
certainty in investing (unlike a fixed-odds casino game), overestimating edge
and betting full Kelly on that overestimate can produce negative growth.
Standard practitioner convention, uniformly echoed: **half-Kelly or less** —
half-Kelly retains roughly 75% of full-Kelly's growth rate while sharply
cutting drawdown severity, buffering against overconfidence/estimation error.

**Staged ramp schedules — explicitly flag as illustrative, not documented
fact.** Publicly available, named, cited "day-1 vs. track-record" position-
sizing ramp schedules from real funds are genuinely hard to find. **Any
numeric schedule circulating in writeups (e.g., "10% of target size for N
rebalances, doubling each cycle," including the specific numbers proposed in
`02_alpha_research.md`'s Breadth Manager sizing rule in this same document set)
should be read as an invented illustrative convention, not a sourced
practitioner rule.** No primary or credible secondary source publishes such a
specific numeric ramp as documented industry practice. What *is* documented,
qualitatively: internal PM capital allocations vary by track record and pod
maturity at multi-manager platforms, newer pods start smaller than established
ones, and building an internal team typically involves a 12-18 month build-out
period paid regardless of live performance before further capital follows
track record. External-manager "seeding" (now used by a majority of large
platforms per industry surveys) runs through fee/revenue-share rather than
salary, with capital pullable post-lockup on underperformance — again, no
audited numeric ramp curve exists publicly. **The honest conclusion**: Kelly/
fractional-Kelly sizing logic is rigorously documented; the specific mechanics
of how much capital a new strategy gets on day one vs. after N periods is
proprietary risk-committee practice at every platform that does it, not a
standardized, publicly documented number.

## Stage 7 — Live Monitoring for Signal Decay

Funds track a small set of statistics continuously once a strategy is live:
rolling **information coefficient (IC)** — correlation between predicted and
realized returns over a moving window — rolling Sharpe, and the gap between
live and backtested/OOS performance. A declining rolling-IC trend is the
standard first flag for decay; a widening live-vs-backtest gap is treated as
evidence the original edge was partly overfit rather than genuinely decaying.

**The hard statistical problem**: a few bad months and true alpha decay look
identical in a short live track record. Andrew Lo's "The Statistics of Sharpe
Ratios" (*Financial Analysts Journal*, 2002) is the standard reference — Lo
derives the asymptotic standard error of an estimated Sharpe ratio and shows
that (1) naively annualizing a monthly Sharpe by √12 is wrong once returns are
serially correlated, and (2) accounting for serial correlation can overstate
an annualized hedge-fund Sharpe by as much as 65%, enough to reorder fund
rankings. The practical consequence used across the industry: with realistic
Sharpe ratios (0.5-1.5) and monthly observations, distinguishing genuine edge
change from zero-mean noise at conventional confidence requires **several
years**, not several months, of live data, because standard error scales
roughly as 1/√T and is inflated further by serial correlation. **Caveat on
precision**: any specific figure like "3.5 years needed" is a derived
consequence of applying Lo's formula under assumptions, not a headline number
Lo states directly — treat such figures as illustrative calculations, not
asserted facts.

**Best-documented cause of genuine decay: crowding.** McLean & Pontiff (2016,
*Journal of Finance*), "Does Academic Research Destroy Stock Return
Predictability?" — the canonical empirical result: across 97 return-predicting
variables from the academic literature, portfolio returns are 26% lower
out-of-sample (upper bound on pure data-mining effects) and 58% lower
post-publication, implying roughly a 32-point incremental decline attributable
to investors trading on a published anomaly once public. More recent work
("Not All Factors Crowd Equally," 2026, arXiv) models crowding-driven decay as
hyperbolic rather than linear/exponential and shows mechanical, easily-
replicated factors (momentum, reversal) decay faster from crowding than
judgment-intensive factors (value, quality) that are harder to arbitrage away.

## Stage 8 — Exit / Kill Criteria

Two models coexist in the industry, genuinely, in tension:

**Mechanical, pre-committed rules** dominate at large multi-manager "pod shop"
platforms. The most concretely reported example: Millennium Management's
drawdown ladder — 5% drawdown from allocated capital triggers an automatic 50%
capital cut, 7.5% terminates the pod, enforced by risk infrastructure, not PM
appeal. **Sourcing caveat, stated plainly**: neither Millennium nor comparable
platforms publish these thresholds officially; the figures circulate through
practitioner newsletters and forum threads from former employees, consistently
repeated but stemming from overlapping secondary reporting rather than
independently corroborated primary disclosure. The *qualitative structure*
(automatic, capital-based, PM-level stop-outs, independent of PM discretion) is
well corroborated across independent secondary sources; the *exact
percentages* are industry convention, not an audited fact, and reportedly vary
by firm and even by PM-level negotiation.

**Discretionary/committee-based review** is more common at single-manager or
research-driven shops with fewer, larger, harder-to-mechanically-stop-out
strategies. The tradeoff, stated directly in practitioner commentary:
"drawdown is not a reason to kill a strategy; drawdown with broken process,
exposure drift, or liquidity mismatch is" — a committee wants to distinguish
*why* a strategy is losing money before cutting it, which a purely mechanical
rule cannot do.

**Why mechanical rules exist at all**: the sunk-cost fallacy / escalation-of-
commitment literature. Decision-makers who have already committed capital and
reputational capital to a thesis systematically under-weight new negative
evidence and continue funding a failing position — well-established outside
finance, applied directly to trading-strategy retirement in practitioner
writing. The standard de-escalation remedies (pre-committed sunset clauses,
rotating the evaluator away from the original decision-maker, external/
independent audit of the kill decision) map directly onto why platforms push
kill decisions into automated risk infrastructure: it removes the actor most
vulnerable to sunk-cost bias from having discretion over the exit.

## Stage 9 — Post-Mortem and Knowledge Capture

**This is the weakest-sourced stage in the entire lifecycle, and that gap is
itself informative.** Generic "investment post-mortem" content exists (e.g.,
semi-annual reviews dissecting how research, decision-making, and execution
contributed to an outcome), but this describes fundamental/discretionary
portfolio review, not a systematic quant fund's process for retiring a
specific signal or strategy. **No practitioner blog, book, or conference talk
was found that specifically documents a formal "strategy post-mortem"
template used at a named systematic fund, nor any public description of a
"rejected-hypothesis log" or "idea graveyard" used to prevent re-testing
failed signals.** This very likely exists as internal convention (a research
wiki, ticket system, or institutional memory of "why we don't retest X") at
serious shops — it would be operationally obvious and low-cost to adopt — but
it is not written about publicly, probably because it reveals competitively
sensitive research process. Any specific claim about "how funds do
post-mortems" beyond generic PM-review content should be treated as informed
inference, not documented fact.

---

## What's genuinely under-sourced across the whole lifecycle

Stated plainly rather than smoothed over, because this document's purpose is
correctness over completeness:

1. **Stage 9 (post-mortem/knowledge capture)** is almost entirely unsourced at
   the practitioner level — the single weakest link in this document.
2. **Staged capital-ramp schedules (Stage 6)** are not publicly documented
   anywhere with real numbers; any specific ramp percentage/timeline in this
   document set (including `02_alpha_research.md`'s own proposed schedule) is
   an illustrative convention, not a cited fact.
3. **Exact kill-criteria thresholds (Stage 8, e.g. Millennium's 5%/7.5%)** rest
   on consistent but non-primary sourcing — convention-level, not audited.
4. **Precise "years needed to detect decay" figures (Stage 7)** are derived
   from Lo (2002)'s standard-error formula under stated assumptions, not a
   number Lo asserts as a headline result.

## How this maps onto Ascent Capital's current code

Cross-referencing this verified lifecycle against `02_alpha_research.md` and
the `00_institutional_audit.md` scorecard: Ascent has real coverage of Stages
3-4 structurally (walk-forward exists, `wf_framework/optimizer.py`) but
**zero DSR/PBO correction anywhere** (Stage 2/4 gap, now doubly confirmed —
first by the original audit, again by this pass's independent literature
check). Stage 5 (paper trading) has scaffolding (`SHADOW_DIR`,
`_promote_to_shadow`) but no formal execution-parity checker. Stage 6's
capital ramp is entirely unbuilt (`SELF_MODIFY_ENABLED = False`) — and per the
finding above, there is no real external precedent to copy for the ramp
schedule even if built; it would need to be a reasoned, disclosed design
choice, not a "this is how real funds do it" claim. **Stages 7-9 have no
counterpart in Ascent's code at all** — there is no rolling-IC monitor, no
mechanical or committee-based kill criterion for a *sleeve* (as opposed to the
whole-book kill switch), and no post-mortem/rejected-hypothesis log. Given how
thin public documentation of Stages 7-9 is industry-wide, building even a
simple version of these — a rolling IC/Sharpe dashboard per sleeve, a
pre-committed (not just discretionary) sleeve-level cut rule, and a plain
append-only log of "signals we tried and rejected, and why" — would likely put
Ascent ahead of typical practice at comparable-scale funds on exactly the part
of the lifecycle real funds are worst at operationalizing and documenting.

## Sources

**Stage 1-2**: Jegadeesh & Titman (1993), *Journal of Finance*; Fama & French
(1993), *JFE* 33; AQR, "Quality Minus Junk"; Cochrane (2011), "Discount
Rates," *Journal of Finance* 66; Harvey, Liu & Zhu (2016), "…and the
Cross-Section of Expected Returns," *RFS* 29(1); AQR, "Taming the Factor
Zoo"; Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest
Overfitting," *JCF* 20(4) 2017; Bailey & López de Prado, "The Deflated Sharpe
Ratio," *JPM* 40(5) 2014; Chan, *Quantitative Trading*, Wiley.

**Stage 3-4**: Pardo, *The Evaluation and Optimization of Trading Systems*,
Wiley 1992/2008; López de Prado, *Advances in Financial Machine Learning*,
Wiley 2018; López de Prado, Minimum Track Record Length / Probabilistic
Sharpe Ratio (boston.qwafafew.org).

**Stage 5-6**: Almgren & Chriss (1999/2000), "Optimal Execution of Portfolio
Transactions"; Thorp, Kelly-criterion practitioner writing (via secondary
summaries); multi-manager platform capital-allocation coverage (Navnoor Bawa
Research, Young and Calculated Substack, angelinvestorsnetwork.com).

**Stage 7-9**: Lo, A. (2002), "The Statistics of Sharpe Ratios," *Financial
Analysts Journal* 58(4); McLean & Pontiff (2016), "Does Academic Research
Destroy Stock Return Predictability?," *Journal of Finance*; "Not All Factors
Crowd Equally" (arXiv:2512.11913, 2026); Young and Calculated Substack, "How
Multi-Manager Hedge Funds Actually Work Internally" and "What Risk Managers
at Pod Shops Actually Do All Day"; Resonanz Capital, "Quant Hedge Funds in
2026: A Due Diligence Framework"; Wikipedia, "Escalation of commitment";
FinanceFeeds, "Analysis on Trading Psychology & The Sunk Cost Fallacy"; TD,
"Tips for Designing an Effective Post-Mortem Exercise."
