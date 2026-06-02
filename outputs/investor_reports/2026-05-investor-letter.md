ASCENT CAPITAL
May 2026 Investor Letter
Strategy: Concentrated Cross-Sectional Momentum
Benchmark: S&P 500 | Paper Trading | Simulated vs Live Prices

---

I. THE NUMBER

| | May 2026 | ITD (Apr 6 – May 31) |
|---|---|---|
| Fund (gross) | +4.89% | +12.93% |
| S&P 500 | +5.26% | +15.35% |
| Net Alpha | −0.37% | −2.41% |

| Metric | Fund | Benchmark |
|---|---|---|
| Max Drawdown (month / ITD) | −5.76% | −1.93% |
| Annualized Vol (month) | 30.9% | 9.8% |
| Annualized Vol (ITD) | 23.2% | 11.0% |
| Beta to SPY (ITD, OLS) | 0.96 | — |
| Walk-Forward OOS Sharpe (Jan 2020–Apr 2026) | 0.52 | — |
| Live Sharpe | Not calculable (39 sessions) | — |
| Calmar | Not meaningful at this sample length | — |

The fund delivered 93 cents of gross return for every dollar the index earned in May, while running volatility 3.2 times wider — a risk-adjusted outcome that deserves no celebration and no apology, only clarity. The ITD picture is the same: −2.41% net alpha against a vol ratio that makes the comparison structurally unfavorable to the fund, and 39 sessions is not a sample that resolves anything.

---

II. WHAT DROVE RETURNS

VICR contributed +1.93% to May's gross return, entering the month at a 7.0% position weight. The stock extended a momentum trajectory that had already produced a reconstructed return of +714% from signal inception, which places it firmly in the territory where the thesis and the factor are nearly indistinguishable — price strength attracts price strength, and separating "the business is rerating" from "the signal is self-reinforcing" is genuinely difficult at this stage of a run. The unanswered question is whether VICR's momentum score still reflects an underreaction to fundamentals or whether it has transitioned into distribution-top behavior of the kind the rebalance flagged explicitly on May 27; the flag was noted, the position was held, and that decision has not yet been vindicated or refuted.

*Weight at entry (month open): 7.0% → 7.0% at close | Contribution: +1.93% | Signal trajectory: parabolic momentum divergence flagged May 27*

EWY, the South Korea ETF, contributed +1.89% against a 10.0% beginning-of-period weight trimmed to 9.3% at the May 27 rebalance. This was factor performance, not thesis performance — the position exists because South Korean equities carried a strong cross-sectional momentum signal relative to the universe, not because of a particular view on the Korean won, semiconductor export cycles, or geopolitical positioning in the region. The trim was correct procedurally: combined EM cluster weight across EWY, EWT, and EEM at 17.5% was flagged as behaving like a single factor under stress, and reducing that concentration before it becomes a problem is exactly the kind of process step that looks conservative until it is necessary. The uncertainty that remains is whether the EM cluster correlation structure holds in the calm-bull regime and breaks specifically under stress, or whether it is already more correlated than the static weight suggests.

*Weight at entry: 10.3% → exit weight 9.3% | Contribution: +1.89% | Risk flag: EM cluster correlation collapse noted at rebalance*

MP Materials was exited during the month and contributed −0.80% before exit. The position was in a sector where momentum signals have been unstable — materials names have shown high sensitivity to macro narrative shifts that the cross-sectional signal does not anticipate cleanly. There is no clean lesson here. The position lost money, the signal deteriorated, and the exit was the process working as designed. Assigning a structural lesson to a single losing materials position at 39 sessions of live data would be an exercise in narrative construction, not analysis.

*Weight at entry → exited | Contribution: −0.80% | Signal: deteriorated to exit threshold*

SATS contributed −0.70% before exit, having entered May with a reconstructed momentum trajectory of +505% from signal inception. The May 27 rebalance flagged SATS specifically as showing distribution-top characteristics with what the risk note described as a concerning signal break. The loss is not a puzzle — a +505% name that rolls over will roll over fast, and a concentrated momentum portfolio will feel that. The honest question is whether the exit came one rebalance cycle too late, and whether a parabolic divergence flag should carry more position-sizing weight than it did when the original weight was set.

*Weight at entry → exited | Contribution: −0.70% | Risk flag: parabolic divergence, distribution-top characteristics flagged at May 27 rebalance*

---

III. ONE THING WE LEARNED

The month produced a return distribution with 8 positive alpha days and 12 negative ones out of 20 sessions, with the largest single positive alpha day at +4.30% and the largest single negative at −2.86%. The arithmetic is obvious: a portfolio that loses alpha on 60% of days but produces a positive gross return does so by concentrating its gains into a small number of large-magnitude sessions. This is not a coincidence of May — it is a structural property of concentrated momentum strategies that is often described but rarely examined precisely.

The relevant observation is this: cross-sectional momentum portfolios at high concentration generate return distributions that are positively skewed at the session level when the regime is cooperative, but the skew is fragile in a specific way. The large positive days — like the +4.30% session in May — tend to occur when a broad risk-on impulse aligns with the portfolio's highest-beta, highest-momentum names all moving in the same direction simultaneously. The large negative days tend to be smaller in magnitude but more frequent, because the same correlation structure that amplifies gains also means that when individual names sell off, they rarely all sell off on the same day at the same velocity. The result is a distribution with a fat right tail and a thicker-than-expected left body.

What makes this precise and not merely descriptive is the following: in May, the fund ran annualized volatility of 30.9% against the SPY's 9.8%, yet the beta to SPY on an OLS basis is 0.96. A beta-neutral portfolio running 3.2 times the index volatility is generating that excess volatility almost entirely from idiosyncratic sources — individual name dispersion, not market-level amplification. This means the kill switch and drawdown monitoring, which are calibrated against portfolio-level drawdown from peak, are doing most of the protective work that sector and factor limits might do in a more diversified book. There is no redundancy in the risk architecture at this concentration level. The 200-day SPY overlay and the regime classification are not supplementary tools — they are load-bearing.

The implication for simulation versus live operation is specific: in simulation, the session-level P&L is reconstructed from model weights and prices, and partial fills, market impact, and bid-ask friction are absent. In a live portfolio at these position sizes, the large positive alpha days — the ones the return distribution depends on — would be partially eroded by the cost of getting into the names that are already moving. The left side of the distribution would be roughly unchanged. The net effect is a compression of the positive skew that does not show up in any simulated Sharpe or walk-forward metric. This is not a theoretical concern. It is the specific mechanical reason why the live Sharpe, when it becomes calculable, should be expected to be lower than the walk-forward OOS Sharpe of 0.52.

This would be wrong if: in the next 30 days, the largest positive alpha sessions do not correlate with broad market up-moves but instead occur on flat or negative SPY days, suggesting that idiosyncratic name selection — rather than beta concentration — is driving the right tail.

---

IV. RISK STATE

Regime: calm_bull throughout May — no transition signals detected; implication is full gross exposure is justified by the model, but the EM cluster flag and two parabolic divergence flags at the May 27 rebalance indicate stress building within the regime classification rather than at its boundary.

Event Risk: June 10 rebalance is the next binary decision point; VICR and WDC ciarry active parabolic divergence flags and represent 14.0% combined weight; EWY at 9.3% remains the largest single position with a documented correlation collapse risk under stress.

Exposure: Gross equity ~100%; net equity ~100%; no short book; no defensive sleeve reallocation active at current regime reading.

Kill Switch: Current drawdown approximately 0.0% from peak at month-end; distance to soft warning at −8.0% is approximately 8.0 percentage points; distance to hard stop at −15.0% is approximately 15.0 percentage points; both thresholds are remote at present.

Monitoring note: The EM cluster at EWY/EWT/EEM combined represents a single correlated factor exposure that is not fully visible in individual position weights; any EM-specific stress event should be evaluated against combined cluster weight, not individual position size.

---

V. WHAT WOULD PROVE US WRONG

The strategy fails if momentum as a factor degrades. If the behavioral underpinnings of momentum — investor underreaction, anchoring, trend-following flows — are materially arbitraged away by systematic capital, the factor's expected return shrinks toward zero. This is a real and growing risk as quantitative AUM expands. Factor crowding is monitored through short interest as a percentage of float, momentum trajectory deceleration, and analyst consensus drift. There is no definitive answer to how much aggregate systematic capital the momentum factor can absorb before the return premium degrades.

The strategy fails if regime classification lags genuine transitions. The defensive infrastructure — gross exposure reductions in stressed and crisis regimes, the SPY 200-day overlay, defensive sleeve reallocation — is only valuable if the regime model identifies transitions before the damage is done. If it systematically lags genuine regime shifts by more than a few sessions, the protection is theoretical. The COVID crash was a regime transition of extraordinary speed. The walk-forward data suggests the model survived it. The live record has not been tested against anything similar.

The strategy fails if process is confused for edge. The risk I take most seriously is believing that because the analytical process is rigorous, the outcomes will be good. Process quality is necessary but not sufficient. I can make excellent decisions and lose money. I can make poor decisions and make money. The live record is too short to distinguish between them. Any allocator drawing strong conclusions from this period — in either direction — is drawing conclusions the data cannot support.

The strategy fails if concentration becomes a liability. Fifteen positions at full conviction is a risk-amplifying choice, not a risk-reducing one. In a market where specific names face simultaneous company-level stress — regulatory action, sector rotation, earnings disappointment — the portfolio has limited capacity to absorb the shocks. This trade-off is accepted because the information ratio of a concentrated portfolio is believed to exceed that of a diluted one at the signal quality available. That belief could be wrong.

---

VI. WHAT'S AHEAD

The next scheduled rebalance is June 10. What makes it structurally notable is not the calendar date but the flags it will inherit: VICR and WDC both carry active parabolic momentum divergence readings from the May 27 review, and SATS — which produced one of the two flagged cases before its exit — is now a data point in how quickly these situations resolve. The June 10 rebalance will either see those flags deepen or clear, and the position-sizing response to that outcome will be the first clean test of whether the divergence flag carries actionable weight in the process or operates as a post-hoc annotation.

The skill score warmup is the most concrete operational milestone approaching. As of month-end, the US equities agent has logged 18 trading days of live signal data, macro sits at 26, and international and alternatives each at 27. The earliest unlock requires approximately 36 more trading days, which places the first skill score threshold — wherever the warmup criterion is set — around mid-August 2026, assuming no trading interruptions. Until that unlock, signal confidence scores are operating on priors that are not calibrated to live price data, and the 0.62 confidence reading at the May 27 rebalance should be read in that context.

Two things at the position level are worth watching without drawing conclusions from them yet. First, EWY's trim to 9.3% reduced but did not eliminate the EM cluster concentration, and any significant move in dollar strength or a deterioration in South Korean export data would test whether the cluster correlation flag was a leading or lagging indicator. Second, the current portfolio carries no materials or energy exposure following the MP exit, which is a significant active bet relative to the index; if those sectors rotate into momentum leadership in June, the rebalance will have to decide whether to chase a signal the portfolio currently has no position in.

---

THANK YOU

You are following a fund in paper trading — which means you are extending trust before there is a track record to trust, and that distinction is not lost on me. I will write this letter the same way in a month where the fund is down 6% as I wrote it this month, and the same way in a month where it outperforms by 300 basis points. The one commitment I can make at this stage, with confidence, is that the numbers and the thinking here will be honest — including when the honest answer is that 39 sessions cannot tell us what we need to know.

---

*Ascent Capital is currently in paper trading mode. All performance figures are simulated against live market prices and do not represent audited live returns. Past simulated performance does not guarantee future results. This letter does not constitute an offer or solicitation. For accredited investors only.*

Ascent Capital Management | May 2026