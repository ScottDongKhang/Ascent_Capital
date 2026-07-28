# Why the AI Portfolio Manager looks like it is dragging the fund down

> **UPDATE 2026-07-28 — root cause of the measurement bug found and fixed.**
>
> The one-day date shift described in Finding 1 has a specific cause: two calls to
> `datetime.fromtimestamp(ts)` with **no timezone**, applied to Alpaca's UTC epochs.
> Alpaca stamps its 1D bars at the 16:00 ET close, which is 03:00 the *next* day in
> Vietnam (UTC+7), so every session shifted forward one calendar day. Friday became
> Saturday; Monday became unreachable. Shifting all 80 published labels back one day
> matches the NYSE calendar **80/80 exactly**, including the three absent Saturdays
> being precisely the weeks whose Friday was a market holiday.
>
> Fixed in `ascent/execution/alpaca_broker.py` and
> `scripts/generate_performance_page.py`, both now using the new
> `ascent/utils/market_time.py` helpers. The pipeline's root date
> (`run_all_agents.py`, `ascent/execution/eod_runner.py`) now derives from the US
> trading day instead of `date.today()`. 16 new tests; the guard tests were verified
> to fail against the pre-fix code.
>
> **Two further facts changed the picture:**
> 1. **The scheduler was never installed** — not a dead path as originally written
>    here. The `/Users/kdong/Downloads/...` string is a fossil in April's stderr log.
>    `logs/launchd_stderr.log` shows `Operation not permitted` for every attempt even
>    on the old machine, so **this system has never once run automatically.**
> 2. **No run ever hit the intended window.** Of 77 logged runs: 50% fired before the
>    US open, 45% during the session, and **0% in the post-close pre-settlement window
>    the "EOD" runner is named for.** So ~78% of rows were dated to a session that had
>    not yet closed. That is a second, independent contaminant on top of the date shift.
>
> Everything below is otherwise unchanged. Findings 3-6 (transmission, capacity,
> prompts, reasoning quality) are unaffected.

---

## REMEDIATION STATUS (2026-07-28)

Worked in the audit's own priority order: measurement, then transmission, then
re-measure. Capacity was deliberately **not** touched and the layer was **not**
disabled — the evidence supports neither.

| # | Item | Status |
|---|---|---|
| T0.1 | Scheduler running | **Done.** Heartbeat 6-hourly; EOD 09:00 local Tue–Sat = 18:00–19:00 PT Mon–Fri, post-close and post-settlement |
| T0.2 | Date basis / Alpaca shift | **Done.** `ascent/utils/market_time.py`; both epoch sites and both pipeline root dates |
| T0.3 | Retract the numbers | **Partly.** Recorded here and in commits; `CLAUDE.md` is being edited concurrently, so not touched |
| T1.4 | `protected_positions` contract | **Done.** Judge field + executor reads it; prose is no longer the only channel |
| T1.5 | `reduce_size` actually de-grosses | **Done.** Targets 0.90 gross; freed weight becomes cash, not a rotation |
| T1.6 | Threshold interaction | **Done.** Detection is pre-renormalization; the self-contradictory 3×1pp-vs-1pp-cap rule is gone |
| T1.7 | Judge change on discovery days | **Done.** Extracted `apply_judge_position_change`, called from both paths |
| T1.8 | Authority-cap the fallback | **Declined, with reasoning.** A per-name cap and an honoured protect-list are mutually exclusive; total gross is bounded instead |
| T2.9 | Derive overrides from weight deltas | **Done.** Replaying history: self-reported 0 → derived 14–19 per decision |
| T2.10 | Scorer field names + same-day short-circuit | **Done.** Both fixed; one `update_authority` call site owns the ladder |
| T2.11 | Date-gate stale priors | **Done.** 14-day expiry, fails closed. The real files (33 days stale) are now rejected |
| T2.12 | Deduplicate logs | **Partly.** New duplicates prevented by the same-session guard; existing rows not rewritten |
| T3.13–18 | Prompt fixes | **Done.** `wedge_21d` objective, authority disclosure, `get_alpha_wedge` in Phase 1, `xhigh`/12k tokens, force-seal keeps its instructions |
| — | Counterfactual backfill can insert dates | **Done.** Mechanism fixed |
| — | **Rebuild the 45 contaminated rows** | **DONE — see below** |

### The counterfactual log has been rebuilt (2026-07-28)

Rebuilt by `scripts/rebuild_counterfactual_log.py` (logic in
`ascent/monitoring/counterfactual_rebuild.py`, 17 tests) from sources independent
of the broken log: settled Alpaca 1D bars for Track B, the snapshot files priced
off `prices_live` for A★/A/D, SPY closes for Track C. Previous log backed up to
`logs/counterfactual_daily.pre_rebuild.20260728-114026.bak.jsonl`.

**45 rows → 86 rows. All 86 expected trading days present, none missing, no
duplicates, no weekends, no holidays.** The Juneteenth row that carried
`track_b +1.53%` on a closed market is gone.

| Track | Old log | Rebuilt |
|---|---|---|
| A★ pure quant | +23.59% | **+9.54%** |
| A quant + Phase-1 priors | +21.53% | **+0.73%** |
| B actual book | +16.03% | **+4.70%** |
| C SPY | +16.63% | **+13.13%** |
| D pure AI PM | +11.54% | **−2.12%** |
| **B − A★** | −7.82pp / 38d | **−5.92pp / 70 paired days, t = −1.24** |
| **D − A★** | −6.34pp / 29d | **−3.04pp / 47 paired days, t = −0.94** |

Window 2026-03-24 → 07-27.

**Two independent reconciliations, both exact.** Track C chained daily equals SPY
point-to-point to the basis point (+13.13% = +13.13%) — a chained series only
matches point-to-point when there are no gaps and no misalignment, which is
precisely the check the old +16.63% failed. Track B chained equals the raw equity
endpoints exactly (+4.70%, 100,000.00 → 104,695.25).

**The alignment defect is gone.** Same-day `corr(B, A★)` is now **+0.936**, and
lag-1 is **−0.118**. Before the rebuild those read −0.005 and +0.60 — two books
sharing 95% of their holdings appearing uncorrelated same-day and correlated at a
one-day lag, which is what generated the −7.82pp headline.

**What this changes.** The gap is real but smaller, and still **not statistically
significant** (t = −1.24 and −0.94). And the ranking changed in a way that matters:
**SPY (+13.13%) now beats pure quant (+9.54%)** over this window, where the old log
claimed quant beat SPY by 7pp. That is consistent with the documented structural
position — defensive sleeves plus the 200MA cut plus vol targeting cost beta in an
equity-only bull — but it is no longer hidden behind an inflated quant number.

A★/A/D still cannot be reconciled against an external source; they are
reconstructions from snapshot weights priced on `prices_live`, using the same
method `score_daily` uses live. One asymmetry is disclosed and deliberately not
"corrected": Alpaca equity is total-return, while `prices_live` closes on the
production cache are split-only, so B is not perfectly comparable to A★/D on
dividend-paying names.

`CURRENT_VERIFIED_NUMBERS.md` still needs its §3 updated with these figures — it
is being edited concurrently, so it was left alone.

**What this changes about the diagnosis.** Nothing in the conclusion. The AI PM's
judgment was never the problem, and the derived-override replay is the sharpest
evidence yet: it was making 14–19 position overrides per decision and every one
was recorded as zero, so nothing could be scored, so authority could never be
earned, so the proposals stayed diluted. That loop is now open at both ends.

**Do not re-litigate capacity yet.** The 5% cap is still 5%. Ask again after
8–10 cleanly scored rebalances, which is now possible for the first time.

**Date:** 2026-07-27
**Method:** 7 parallel audit agents (architecture, decision quality, performance, prompts, transmission) plus 3 adversarial verifiers whose job was to try to prove the findings wrong.
**Status of the numbers below:** every claim was checked against a real file. Where a verifier softened or corrected a claim, the corrected version is what is written here.

---

## The short answer

You asked whether the problem is capacity, bad prompting, or AI being unable to reason at a high level.

It is mostly **none of those three**. The dominant problem is that **the measurement is broken**, and the second problem is that **the plumbing throws away the decisions**. The AI's actual thinking is the healthiest part of the system.

In order of how much damage each one is doing:

1. **The scoreboard is wrong.** The "-7.82pp" number that made the AI PM look bad is not a measurement. It is a side effect of a date bug.
2. **The system was offline for a month.** Two scheduled rebalances never ran.
3. **Decisions get discarded downstream.** Roughly half of them never reach the portfolio at all.
4. **Capacity (the 5% cap) is genuinely tight** — about 86% of intended action is thrown away — but there is no evidence yet that lifting it would help.
5. **Prompting has real, fixable problems** — the biggest being that the AI is told to optimise one thing and graded on a different thing.
6. **"AI cannot reason at a high level" is not supported.** The reasoning was specific, falsifiable, and right more often than wrong.

---

## Finding 1: The number that condemned the AI PM is a date bug

`CLAUDE.md` says: `B-A* = -7.82pp/38d`, and `pure quant +23.59% beats actual (+16.03%) and SPY (+16.63%)`.

Those numbers are quoted correctly from `logs/counterfactual_daily.jsonl`. The log itself is broken in three ways:

**It says SPY made +16.63%. SPY actually made +5.31%.** Recomputed straight from `data_cache/prices_live.parquet`. The dashboard's own SPY series independently agrees with about +5.3%. Being 11 percentage points wrong about the *benchmark* invalidates every level in the table.

**Why:** the log has only **45 rows for an 81-business-day window**. 36 days are simply missing. The code chains daily returns together as if the missing days never existed. The skipped days happened to be net negative, so every track is inflated 2.5x to 4x. And the backfill functions in `ascent/monitoring/ai_pm_counterfactual.py` can only patch rows that already exist (`backfill_track_b` at line 179, `backfill_astar_d` at line 257) — neither can insert a missing date. So the holes are permanent by design.

**Track B is Track A-star shifted one day.** Correlation between the two at same-day is **-0.005**. At a one-day lag it is **+0.60**. Two portfolios that share 95% of their holdings cannot be uncorrelated — that is arithmetically impossible. It is a date-alignment defect. You can see it raw in the log: A-star is -2.54% on 05-07 and B is -2.55% on 05-08.

**The whole -7.82pp rests on 3 days out of 38.** Drop 2026-06-11, 2026-06-24 and 2026-05-08 and the number flips from **-7.82pp to +7.92pp**. All three are days where B posts a big loss against a big A-star gain — and in each case the matching A-star move shows up on the *neighbouring* day. The headline is the residual of the shift, not a performance fact.

Related: the public performance page (`docs/index.html`) has the same defect. Its date axis contains **14 Saturdays and only 1 Monday** across the window. Every stat computed from it — Sharpe "from 81 sessions", max drawdown -9.26%, best/worst day — is computed on misdated bars.

**Properly rebuilt over all 69 trading days from 2026-04-16 to 07-24:**

| Track | What the log says | Properly dated |
|---|---|---|
| Pure quant (A*) | +23.59% | **+10.35%** |
| Actual book (B) | +16.03% | **+4.11%** |
| SPY (C) | +16.63% | **+5.31%** |
| Pure AI PM (D) | +11.54% | **-2.12%** |
| **B - A*** | -7.82pp | **-6.24pp** |

So a gap does exist and pure quant really did beat the traded book. But it is smaller than advertised, it is **not statistically distinguishable from noise** (p = 0.61), and critically — see Finding 4 — the AI PM layer is mathematically incapable of causing most of it.

There is a bitter irony here. The docstring of `_common_window_diff` in that very file warns that comparing tracks over different windows "is meaningless - this is what produced the fictional -11.6pp 'AI PM cost'." The four-way comparison in `CLAUDE.md` compares numbers cumulated over 38, 45, 42 and 29 different day-sets. It is the exact mistake the code was written to prevent.

`CURRENT_VERIFIED_NUMBERS.md` already flags `B-A*` as UNSTABLE / low confidence. That caveat did not travel with the number when it was copied into `CLAUDE.md`.

---

## Finding 2: The system did not run for 19 trading days

`logs/eod_log.jsonl`, `logs/holdings_log.jsonl` and `logs/cost_log.jsonl` all jump from **2026-06-29 straight to 2026-07-27**.

This is not a logging failure. The evidence:

- The watchdog caught it: `logs/liveness.json` reads `status: CRITICAL`, `last_run: 2026-06-29`, `missed_days: 19`, `missed_rebalances: ["2026-07-08", "2026-07-22"]` — exactly the two dates in `rebalance_calendar.csv`.
- The 7/27 run diagnosed itself: `run_type: "catch_up"`, `"Outage recovery: 19 trading day(s) missed prior to this run; not replayed."`
- `git log` shows one commit on 6/29, then nothing until 17 commits on 7/27.
- No file anywhere in the repo carries a July date before the 27th.
- `logs/launchd_stderr.log` last wrote in **April** and points at a dead path (`/Users/kdong/Downloads/ascent capital v2 up to phase 5.1/scripts/run_eod.sh: Operation not permitted`). The scheduler was not pointing at this repo.

Consequences: the book's **-3.10% July drawdown is invisible to every track**, and the two rebalances the AI PM would have participated in never happened. This is a plain infrastructure failure and it costs more than any prompt tuning would gain.

---

## Finding 3: The plumbing throws the decisions away

This is the part that most directly answers "it thinks the right thing but cannot act."

### 3a. "reduce_size" has never once reduced size

The verdict channel named `reduce_size` cannot reduce exposure, by construction:

- `ascent/execution/eod_runner.py:569` instructs Haiku that **"weights must sum to 1.0."**
- Line 599-602 renormalizes back to 1.0 if it drifts.
- `_enforce_reduce_size` ends at line 539-542 by renormalizing to exactly 1.0.

Gross exposure on all three `reduce_size` dates, checked in the artifacts:

| Date | Gross before | Gross after |
|---|---|---|
| 2026-04-06 | 1.000003 | 1.000003 |
| 2026-04-15 | 1.000002 | 1.000002 |
| 2026-07-27 | 1.000000 | 1.000000 |

It is a rotation channel wearing a de-risking name.

### 3b. The 7/27 incident, reconstructed exactly

The judge wrote, in `verdict_2026-07-27.json`:

> "UUP and TLT rank higher on priority, but both are the hedge leg and **cutting insurance 48 hours before the catalyst it exists to hedge is poor timing** - the bull wins that specific exchange."

So it deliberately chose VNQ instead: 7.3% to 6.3%.

Here is what actually happened, reproduced in Python to a maximum error of 0.0084pp across all 23 positions:

1. Haiku **correctly obeyed the judge** and cut VNQ by 1.01pp. The judgment transmitted perfectly at this stage.
2. That left the book summing to 0.98990, so the renormalizer scaled everything back up — which shrank the VNQ cut to **0.947pp**.
3. The enforcement layer requires cuts of **at least 1pp** to count. 0.947 < 1.0, so it counted **zero reductions**.
4. It then force-trimmed the **top 5 positions by weight**: UUP, TLT, VNQ, IFRA, and **BIL (T-bills)**.
5. Gross stayed at exactly 1.0. The freed 10pp was pushed into the 18 smaller, mostly cyclical names.

So: the renormalizer erased the only reduction, then the detector punished its absence, then a size-sorted fallback sold the exact dollar and duration hedges the judge argued to protect — **plus the T-bill sleeve the same verdict called too small** — and rotated 10pp into equities 48 hours before FOMC. Under a "reduce risk" verdict.

The fallback cannot know better: `_enforce_reduce_size` (line 465) receives two dictionaries of numbers. It has no verdict, no reasoning, no protected list. The verdict schema in `debate/judge.py` has fields for `recommendation`, `reasoning`, `key_risks` and `position_changes` — but **no field for "do not touch this"**. The judge's protection existed only as English prose.

It also breached the authority cap: VNQ moved **-2.93pp** against a stated 1.0pp limit. The enforcement layer is not authority-aware.

That fallback shipped on 2026-04-17, after the first two `reduce_size` dates. **7/27 was its first opportunity and it fired: 1 for 1.**

### 3c. Off-calendar days run the debate and bin the answer

The code that applies the judge's position change lives only in the scheduled-rebalance branch (`run_all_agents.py:1858-1935`). The discovery path (`:2583-2637`) runs the full debate, writes a complete verdict file, and then honours only `halt_and_review` — it never applies `position_changes` and never calls `record_intervention()`.

Result: of 7 judge position changes ever produced, **4 (57%) were never applied and never scored** — 6/15, 6/22, 6/29 and 7/27, all off-calendar days. Confirmed by checking the executed weights: on 6/15 the judge said cut PK to 6.22%; PK executed at **0.072221**, untouched. Same for BAX on 6/22 and TLT on 6/29.

---

## Finding 4: Capacity is genuinely tight, but raising it is not the right first move

Your instinct about the 5% cap is mechanically correct. Measured on the real logged proposals:

| Rebalance | AI wanted to move | Actually landed | Fraction that survived |
|---|---|---|---|
| 2026-05-19 | 16.7% | 5.0% | 30% |
| 2026-05-27 | 26.3% | 5.0% | 19% |
| 2026-06-10 | 45.3% | 5.2% | 11% |
| 2026-06-24 | 36.5% | 5.0% | 14% |

**Median ~14% survives. About 86% of intended active risk is discarded.**

Worse, the cap changes the *meaning* of decisions. It converts exits into shaves:

| Date | AI PM wanted to exit | Still held after blending |
|---|---|---|
| 2026-06-10 | BWA, CNC, PK, VVV, WDC at 7.00% each | **6.26% each** |
| 2026-06-24 | TLT 10.0%, EWY 8.82%, UUP 6.28% | **8.63%, 7.61%, 5.42%** |

"Sell this position" becomes "trim it by 1.4pp." A conviction to be *out* of something cannot be expressed at all.

**But — and this matters — every measurement of "what if it had more room" points down, not up.** Excess return vs pure quant at 21 days, by authority level:

| Event | 5% | 15% | 30% | 100% |
|---|---|---|---|---|
| 2026-05-19 | -1.39 | -4.17 | -4.65 | -4.65 |
| 2026-05-27 | +0.16 | +0.49 | +0.85 | +0.85 |
| 2026-06-10 | -0.07 | -0.18 | -0.34 | -0.51 |
| 2026-06-24 | +0.17 | +0.51 | +1.02 | +1.24 |
| **Mean** | **-0.28** | **-0.84** | **-0.78** | **-0.77** |

Read that honestly, though, because it is weak evidence:

- **n = 4.** That is four decisions, not 38 days.
- **One event drives the entire negative result.** Remove 2026-05-19 and the mean flips positive. That event was the AI zeroing SNDK and cutting WDC three weeks before they ran +43% and +48% — one semiconductor-memory squeeze, not a pattern.
- **It depends on when you stop the clock.** That same event is -1.32pp at 5 days, -4.65pp at 21 days, and **-0.34pp at 47 days**.
- **The 29-day daily version of this test uses the broken log from Finding 1**, so it should carry little weight.
- A separate agent, measuring the same two scored rebalances from `logs/ai_pm_calibration.jsonl`, found the AI PM's **full** proposal *beat* the quant baseline both times (-1.04% vs -1.78% on 6/24; -0.82% vs -2.01% on 6/10). That contradicts the table above. The two used different baselines and horizons. **I cannot resolve which is right at n=4** — and that is the real finding.

**Conclusion on capacity:** the throttle is real and severe, but there is no credible evidence it is costing money, and a weak directional signal it is saving some. Do not raise it yet. Fix the measurement first, then the answer becomes knowable.

### The cap also cannot currently be earned

Promotion from Level 1 needs `n_decisions_evaluated >= 5` (`earned_authority.py:239`). That counter is fed by the `overrides_applied` field of `logs/ai_pm_decision_log.jsonl`, and **all 9 rows have it empty**. So it sits at 0 while demotion needs only a single bad day (`:196-219`). Currently `days_stuck: 19`.

The verifier found this is *empirically* rather than *structurally* stuck — and then found two more defects that would block promotion even if overrides did start firing:

1. **Field-name mismatch.** The tool emits overrides as `{symbol, ai_action, reason, override_type}` but the scorer reads `ov.get("ai_w")`, `ov.get("quant_w")`, `ov.get("type")` (`ai_pm_perf_feedback.py:131-134`). Real overrides would score 0.0, forcing `profit_factor = 1.0`, which fails the `>1.2` gate forever.
2. **Same-day short-circuit.** On rebalance days `update_authority` is called early at `run_all_agents.py:1760` with the counter defaulted to 0; the function then early-returns on `last_updated == today` (`:159-162`), so the *informed* check at `:2194` never runs.

So the ladder is a one-way ratchet downward, for three independent reasons.

---

## Finding 5: The prompting problems are real and cheap to fix

**The objective is wrong.** The prompt says `"OBJECTIVE: Sharpe ratio, not raw return"` (`ai_pm_agent.py:1073`) and asks for a 3-month view. It is graded on `wedge_21d` — a **21-day raw return difference** vs the quant. A model that correctly follows the instruction at line 1079 — *"When in doubt, choose the lower-volatility expression of the same thesis"* — will systematically lose the metric it is scored on, in a bull market. **It is being graded on a test it was told not to study for.**

**It is never told its own authority.** Grep finds no mention of Level 1, 5%, or the blend anywhere in its prompts. So it writes 9-10% conviction weights believing they will be implemented, when the most that can land is a ~1pp nudge. The debate judge, by contrast, *is* told its cap and visibly reasons well with it ("Capped at 1% by unproven adversarial_thesis authority (n=9)"). The AI PM is denied that same input.

**It is never told its track record.** `get_alpha_wedge` — the tool that would show it the counterfactual — is described in its own docstring as "Call in Phase 1", but it is **not in `PRE_THESIS_TOOLS`** (`:886-896`). In the live two-phase path it literally cannot be called.

**The prompt is a brake, not a licence.** There are **eleven** distinct stand-down instructions ("carry the quant weight and do not reduce", "Stand down", "prefer the quant baseline", "If you see no reason to deviate, submit it unchanged") against **one** pro-deviation channel that requires three simultaneous confirmations and is still capped near the normal weight band. The always-on calibration warning fires whenever IC < 0.05, which with a handful of samples is nearly always. Structurally, "do nothing" is the highest-reward action the prompt offers.

**The deepest-judgment call is under-resourced.** Phase 2 (Opus synthesis, the single most important call in the fund) runs at `effort="high"` with `max_tokens=4000` — which is *below* the wrapper's own 4096 thinking floor — and that budget must cover thinking plus ~20 weights plus a 12-key thesis memo. **No call site in the entire repo uses `xhigh` or `max`.** When that budget is exhausted, the force-seal fallback fires — and Phase 1's force-seal **drops the task instructions entirely** (`:2332` omits `_PRE_THESIS_PROMPT`), so the sealed thesis comes from a model never told what a thesis is.

**One live bug:** `eod_runner.py:578` is a non-f-string containing `{{"AAPL": 0.05}}`, so Haiku is shown doubled braces as its example of valid JSON.

**Claude 5 compliance** is good in `client.py` itself. Four `resp.content[0].text` sites remain (`falsifier_registry.py:218,416`, `run_all_agents.py:123`, `weekly_debrief.py:197`) — all on Haiku, so latent rather than broken, but they will silently return nothing the day anyone points them at Sonnet or Opus.

---

## Finding 6: Was the thinking actually good? Yes.

This is the answer to "can AI do high-level reasoning here."

Of 11 real dated decisions (one, 2026-04-12, is a test fixture with `reasoning: "ok"` and must not be counted):

- **6 = the reasoning was right but never reached the weights**
- **4 = it reached them but was clamped to irrelevance**
- **2 = unclear**
- **0 = the reasoning was actually wrong**

Position-level hit rate, deduplicated against replay duplicates in the log: **4 of 6 correct, averaging -3.49pp of 10-day relative underperformance** on the names it flagged. That is real short-side selection skill.

Quality examples worth reading:
- It caught the bull inflating the defensive allocation by miscounting PDBC/HYG/NEM — 19.4% claimed vs 7.5% actual. A checkable arithmetic catch.
- It flagged USO's contango decay at a 12.52% allocation: "even a correct oil call will underperform."
- It caught the model's own internal contradiction — quant VaR -0.96% vs regime-flip -7.1%.
- It refused to cut hedges 48 hours before FOMC, and named which exchange the bull had won.

That is not macro boilerplate.

**Two real self-contradictions, and both are caused by the plumbing:**

1. On 7/27 it set `recommendation: reduce_size` while arguing the two largest positions must not be cut — not knowing that `reduce_size`'s only mechanism is cutting the largest positions.
2. The 04-15 debrief scores itself **WRONG** because the portfolio rose 5.63%, then prescribes *more* aggression next time. But its premise is false: **nothing was reduced on 04-15** (gross stayed 1.0000). It was penalised for an action the architecture is incapable of taking.

That false penalty then became the dominant prior. The "April 15 +5.6%" self-criticism appears in **five consecutive verdicts** as the reason not to act. The system's own `blind_spots.json` logged this at severity `high`, `first_seen: 2026-05-27`, `last_seen: 2026-06-29` — and nothing changed.

**So the model spent five runs suppressing its own valid risk warnings to atone for a mistake it never made.** A plumbing failure manufactured the appearance of bad judgment.

---

## Two other things worth knowing

**Phase 1 has never seen the portfolio.** `ai_pm_agent.py:2255` reads `data_cache/merged_weights.json`. **That file does not exist** — every other reader and writer in the repo uses `execution/merged_weights.json`. So the holdings list is always empty and the entire "CAUSAL INTELLIGENCE" block is never injected. (Even if the file existed there, the code takes `.keys()` of a wrapper object, which would yield `date`/`weights`/`agents`, not tickers.)

**Track A-star, the "no-AI-PM baseline," is contaminated by the AI PM.** `ascent/main.py:641-670` reads `ai_prethesis_latest.json` and uses it to floor conviction names' alpha and **zero out the avoid-list** — with no authority cap and **no date check at all**. On the 7/27 run it read a pre-thesis dated **2026-06-24, 33 days stale**, and 6 of its 10 conviction names are in the executed book. `logs/regime_blend_log.jsonl` was written during that run from the month-old assessment. So the baseline used to judge the AI PM contains the AI PM's month-old opinions.

Also dead: `apply_guardrails()` in `ascent/strategy/ai_pm_guardrails.py` is documented as part of the path and is **never called in production** — only in tests.

---

## Next steps, in the order I would do them

### Tier 0 — do these before trusting any number again

1. ~~**Get the scheduler running.**~~ **DONE 2026-07-28.** Heartbeat watchdog installed
   on a 6-hourly `StartInterval` (survives sleep) and verified. EOD job installed at
   09:00 local Tue-Sat, which maps to 18:00-19:00 PT Mon-Fri — after the close *and*
   after Alpaca's ~17:00 PT settlement, with ≥1h margin in both DST regimes. The old
   plist asserted the machine was in PT and would have fired at 22:45 PT the previous
   day. A same-session guard (`already_ran_for_session`) now prevents the duplicate
   rows that put 8 copies of 2026-06-10 in the decision log.
2. **Rebuild the counterfactual log with insertable dates.** Make the backfills able to
   add missing rows, and date returns by the actual bar date rather than "last two
   non-NaN closes". The Alpaca key misalignment itself is now fixed at source, but
   **the 45 existing rows were written with the old shifted keys and are still wrong** —
   they need re-deriving from `get_portfolio_history()`. Also drop the ~10 closed-market
   rows (including a `track_b +1.53%` on Juneteenth) and the 7 stale duplicate Track B
   values. Until this is done, no counterfactual number is citable. (Finding 1)
3. **Retract the current numbers.** `CLAUDE.md`'s `-7.82pp`, `+23.59%/+16.03%/+16.63%/+11.54%`, and the public page's Sharpe and max-DD should be marked unsupported until the rebuild is done. The public page is the more urgent one — it is published.

### Tier 1 — make decisions survive the trip to the broker

4. **Add `protected_positions` to the verdict schema** (`debate/judge.py`) and pass it into `_enforce_reduce_size`, excluding those names from the top-5 sort. The 7/27 reasoning already contained this in prose; the model had nowhere to put it.
5. **Make `reduce_size` actually reduce gross.** Drop "weights must sum to 1.0" from the Haiku prompt and remove the two renormalize-to-1.0 steps (or renormalize to `1 - reduction_pct`). No LLM change needed. Highest value per line changed in the whole list.
6. **Fix the threshold interaction** — count reductions before renormalizing, and stop requiring three 1pp cuts while capping each intervention at 1.0pp. Those two rules contradict each other.
7. **Extract the judge-application block** (`run_all_agents.py:1858-1935`) into a function and call it from the discovery path too. This alone recovers 4 of 7 lost interventions and restores scoring.
8. **Authority-cap the fallback** so it cannot exceed what the judge itself was allowed.

### Tier 2 — make the learning loop able to close

9. **Derive overrides from the actual weight deltas** instead of trusting the model to self-report `quant_overrides`. Both vectors are already logged side by side. This unblocks the promotion counter.
10. **Fix the scorer's field names** (`ai_w`/`quant_w`/`type` vs `ai_action`/`override_type`) and the same-day short-circuit at `run_all_agents.py:1760`.
11. **Add a date gate** on `ai_prethesis_latest.json` and `ai_regime_assessment.json` reads, and snapshot A-star on discovery days.
12. **Deduplicate the decision log** (6/10 appears 8 times; the intervention log double-counts VNO 4x).

### Tier 3 — prompting

13. **Tell it what it is actually graded on**, and delete the Sharpe/Information-Ratio framing that contradicts the metric.
14. **Tell it its authority level**, the way the judge is already told. Then a 9% conviction becomes an honest statement rather than a wish.
15. **Add `get_alpha_wedge` to `PRE_THESIS_TOOLS`** and inject the counterfactual unconditionally.
16. **Balance the eleven stand-down instructions** with one explicit licence to disagree, and note that a submission identical to the baseline is itself a cost (turnover, spend, zero expected alpha).
17. **Raise Phase 2 to `effort="xhigh"` with `max_tokens` around 12000**, and make both force-seals inherit the real system prompt.
18. **Fix the doubled braces** at `eod_runner.py:578` and the four `content[0].text` sites.

### What not to do

**Do not disable the AI PM layer, and do not raise the 5% cap yet.** On the evidence, disabling it would remove the one component whose reasoning is demonstrably sound, and raising the cap would amplify decisions through plumbing that currently inverts them. The correct sequence is: run the system, fix the measurement, fix the transmission, then accumulate 8-10 cleanly scored rebalances. At that point the promotion gates answer the capacity question with a sample worth acting on. Right now, with n=4 and one event dominating the result, nobody can answer it — including this report.

---

## One-line summary

The AI PM is not failing at thinking. It is being graded by a broken scoreboard, on a system that was switched off for a month, through plumbing that sells the exact positions it argues to protect — and then it reads its own falsified failures back into the next decision and stands down.
