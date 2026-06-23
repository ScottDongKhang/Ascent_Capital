# Ascent Capital — Repo-Verified Technical Summary

> **⚠️ SUPERSEDED IN PART (2026-06-22).** Quote performance numbers only from
> [`CURRENT_VERIFIED_NUMBERS.md`](CURRENT_VERIFIED_NUMBERS.md). Since this file was first
> written, the walk-forward backtest was found to rest on a **corrupted `prices_live` cache**;
> the cache was then **re-fetched clean and the backtest re-run (2026-06-22)**. The verified
> WF figures are now **Sharpe 0.41 / CAGR +10.3% / +1pp excess vs SPY / max DD −32.9% / WFE
> −0.65** (the §3 backtest block in this file has been updated; the +12.61%/0.483 figures are
> dead). The AI-layer counterfactual is D−A★ −6.52pp/23d and B−A★ −6.57pp/32d (unstable).
> Treat the architecture/methodology sections here as current.

> Prepared for outreach drafting. Every number below is pulled from the actual repo
> (logs, backtest artifacts, config, git history) as of **2026-06-22**, not from prior
> descriptions. Where the repo's own docs and its recorded artifacts disagree, that is
> flagged explicitly. **Do not round numbers favorably.**

---

## 1. One-line description

A fully-automated, multi-agent stock-trading system that runs end-to-end on its own
once a day — it pulls market data, builds a portfolio with several competing models,
runs an LLM "risk committee" debate over the proposed trades, lets an LLM "portfolio
manager" make small adjustments, and submits the orders to a brokerage paper account —
while continuously scoring its own AI decisions against a no-AI baseline to decide how
much money the AI is allowed to touch.

---

## 2. Architecture — what's live vs. scaffolded

**Live and wired into the daily run** (`run_all_agents.py`, ~111K-line entrypoint, runs via launchd on a personal Mac):

- **4 quant specialist agents** — `us_equities` (901-symbol universe, 14 alpha sleeves),
  `macro` (12 ETFs), `international` (12 ETFs), `alternatives` (7 ETFs). Run in parallel,
  each emits an `AgentOutput`.
- **Orchestrator** (`orchestrator/central_intelligence.py`) — skill-weighted capital
  allocation across agents, correlation/crowding guards, sector and EM caps.
- **Regime engine** — HMM, K=3 states (calm_bull/stressed/crisis/euphoric/uncertain);
  adjusts sleeve weights and gross exposure. Currently reporting `calm_bull`.
- **Adversarial debate layer** — bull/bear/devil's-advocate/judge LLM agents with
  deliberately distinct epistemologies (Druckenmiller/Burry/Taleb framings). Produces a
  verdict JSON per rebalance and makes **exactly one falsifiable weight change + a 10-day
  prediction** that is scored later. Advisory only — cannot write to alpha/portfolio code.
- **AI PM** — two-phase: Sonnet "pre-thesis" (breadth) → Opus "synthesis" (judgment),
  blended in at a **5% active-weight budget**. Governed by an "earned authority" ladder.
- **Counterfactual measurement layer** — logs daily returns for 5 tracks: A★ (pure quant),
  A, B (actual book), C (SPY/benchmark), D (pure AI-PM). This is how the AI is graded.
- **Execution** — Alpaca paper trading, kill switches (8% warn / 15% halt), >2% NAV
  trades route through an approval gate.
- **Integrations** — OpenBB (prices/macro/options/COT), Exa (news), StockTwits (sentiment),
  MiroFish (Monte-Carlo / agent-based sentiment sim, runs against an external local server).
- **Public dashboard** — auto-published to GitHub Pages after every run.

**Scaffolded / present but unvalidated / disabled:**

- `event_agent.py`, plus `EVENT_TRADING`, `LONG_SHORT`, `TWAP`, `SELF_MODIFY` paths —
  exist in code, **kept off pending paper validation** (per CLAUDE.md, target ~July 2026).
  WF backtests of long-short and multi-asset variants scored *worse* than the live config.
- **Self-improve loop** (weekly variant search) — shadow-only, nothing auto-promoted.
- **Memory modules** (`memory/r2r_interface.py`, `reflection_agent.py`, `ticker_memory.py`,
  `regime_memory.py`) — present; depth of live use is unclear from logs (`stocktwits_ic.jsonl`
  is empty, some integration logs are thin).
- **MiroFish** — functional but fragile: needs an external server running and OpenRouter
  credits; has repeatedly failed/timeouted in logs.

---

## 3. Real performance metrics

### Live paper trading (Alpaca, since 2026-04-01 → 2026-06-22; ~3 months, ~60 trading days)

| Metric | Value | Source / caveat |
|---|---|---|
| Total return | **+8.82%** (from Apr-1 anchor) / +9.71% (from Mar-23 $100K anchor) | `holdings_log.jsonl`; current equity **$109,710**, 17 positions |
| SPY over same window | **+13.55%** | `holdings_log.jsonl` `spy_return`, n=68 days |
| Alpha vs SPY | **−5.08%** (i.e. it has *trailed* SPY) | README headline; consistent with the −4 to −5pp gap I computed |
| Sharpe (annualized) | 1.794 | **Not significant** — ~47–60 day sample; README itself flags the standard error as large |
| Max drawdown | −6.58% | README headline |
| Rebalances executed | **7 distinct sessions** (Apr 15, May 5, May 19, May 27, Jun 10, Jun 15, Jun 22) | `eod_log.jsonl`. Note: Jun 10 shows 8 duplicate entries from overnight reruns |

> ⚠️ The single most important honest framing: **over the only live period that exists,
> the system has returned ~+8.8% while SPY returned ~+13.5% — it has underperformed a
> plain index by ~5pp.** The team attributes this to deliberate defensive sleeves
> (~22% bonds/gold/cash/commodities) plus a 200-day-MA cut and a 15% vol-target overlay,
> which give up beta in an equity bull market. That's a plausible *structural* explanation,
> but it is an explanation for underperformance, not outperformance.

### AI PM counterfactual (the headline "AI-native" feature)

Computed directly from `counterfactual_daily.jsonl` and `earned_authority.json`:

- **Pure AI-PM (Track D) vs pure quant (Track A★): −6.52pp over 23 common days.**
- **Actual book (Track B) vs pure quant (A★): −5.27pp over 31 common days.**
- AI PM status: **Level 1 "Analyst," 5% authority, stuck 13 days, failing every promotion
  gate** (`ai_pm_perf_feedback.json`): sortino_edge −3.73 (needs +0.20), hit_rate 0.0,
  profit_factor 1.0, decisions_evaluated 0.

> So the AI layer is currently **value-neutral-to-negative** on a small sample, and the
> authority system is correctly **refusing to give it more capital.** This is honest and
> arguably the most interesting part of the system — but it is not yet a story of the AI
> adding alpha.

### Walk-forward out-of-sample backtest — VERIFIED CLEAN (2026-06-22)

> ⚠️ **The old 0.483 / +12.61% figures below the line are SUPERSEDED.** They came from a
> corrupted price cache. The cache was re-fetched clean (single source, 0 duplicate rows,
> 0 implausible jumps) and the backtest re-run on **2026-06-22**. Cite only the verified
> block. Single source of truth: `CURRENT_VERIFIED_NUMBERS.md`.

**Verified result** (`outputs/wf_results/wf_report_clean_2026-06-22.json`, OOS 2021-01-08 → 2026-01-14, 1,134 days, 21 folds; LLM sleeves off):

| Metric | Value |
|---|---|
| Sharpe | **0.41** (independently recomputed 0.417) |
| CAGR | **+10.3%** (recomputed +10.4%) |
| Excess CAGR vs SPY | **+1.0pp** (strategy 10.42% − SPY 9.41%, same window) |
| Regression alpha (annualized) | +2.24% |
| Beta | 0.73 |
| Max drawdown | −32.9% |
| Win rate | 50.2% |
| **Walk-forward efficiency** | **−0.65** (negative — IS optimizer adds no OOS value; disclose) |

> **Caveats:** modest edge (Sharpe ~0.4), thin +1pp/yr excess over SPY at defensive beta,
> single backtest (not a live track record), and the WFE is negative. The engine's Sortino
> field (0.04) is **miscomputed** — the real Sortino is ~0.68. This is a credible but
> unremarkable risk-adjusted profile, honestly reported.

<details><summary>Superseded numbers (corrupted cache — do not cite)</summary>

The old "current" artifact (`wf_report_2026-06-04.json`) showed Sharpe 0.483 / CAGR +12.61%
/ regression alpha +5.55% / Sortino 0.05 / max DD −36.8% / WFE −0.217. Those came from a
price cache with ~59% duplicate rows + 10×-type errors in 12 symbols, which inflated the
result. The README additionally showed a −23.4% drawdown and +2.54% alpha that matched **no**
saved artifact. All dead — see `AUDIT_DATA_INTEGRITY.md`.
</details>

---

## 4. Timeline

- **Git history begins 2026-04-12** (448 commits); earlier development predates this repo
  on a prior (school-managed) machine — exact original start not recoverable from git.
- **2026-04-01** — Alpaca paper trading goes live (29 orders, 9 initial positions).
- **2026-05-19** — AI PM shadow period begins.
- **2026-06-04** — AI PM promoted to Level 1 (Analyst), 5% authority budget.
- **2026-06-08 → 06-11** — OpenBB/CBOE/CFTC data, per-ticker memory, MiroFish, vol-target
  parity, falsifier enforcement added.
- **2026-06-18 → 06-22** — multiple sessions spent *fixing the measurement layer itself*
  (counterfactual self-heal, Track B repair, authority buffer reconciliation).
- **Current status:** paper only, ~3 months live, daily automated runs. No real capital.

---

## 5. Novel vs. standard

**Genuinely differentiated (mostly epistemic/validation, not alpha):**

- **Earned-authority model for the AI PM** — a Sortino-gated promotion/demotion ladder
  with automatic reversion, where the AI must beat a pure-quant counterfactual over 21
  rebalances before getting more capital. This is an unusual, disciplined way to deploy an
  LLM in a trading loop.
- **Daily counterfactual scoring (Tracks A★/A/B/C/D)** — the system explicitly measures
  "what would the AI alone have done" vs "pure quant" vs "the actual book" vs "SPY." Most
  systems don't instrument self-attribution this honestly.
- **Adversarial debate as a falsifiable risk committee** — distinct agent epistemologies,
  exactly one falsifiable change + a scored 10-day prediction per rebalance, and
  intervention types auto-suspended below 40% accuracy.
- **A visible culture of catching itself fooling itself** — multiple recent sessions are
  measurement-bug fixes, not return-chasing.

**Competent but standard quant infrastructure:**

- The alpha engine: cross-sectional momentum (58%), stat-arb, mean-reversion, XGBoost ML
  sleeve, HMM regime, sector-constrained optimization, vol targeting, 200MA overlay.
- Walk-forward OOS, CPCV with purge/embargo — best practice, not novel.

---

## 6. Known weaknesses / open questions a skeptic will flag

1. **The live track record is ~60 days** — far too short for any Sharpe to be meaningful
   (the repo admits this).
2. **In the only regime it has lived through, it underperformed SPY by ~5pp.** "Structural/
   defensive" is a fair explanation but it's still underperformance.
3. **The AI PM — the marquee feature — is currently subtracting ~6pp vs pure quant** and
   failing all promotion gates. Small sample, but the headline can't be "AI adds alpha."
4. **WF OOS is modest:** clean re-run (2026-06-22) gives Sharpe **0.41**, CAGR **+10.3%**,
   only **+1pp/yr** excess over SPY, max DD **−32.9%**, and **negative walk-forward
   efficiency (−0.65)** — the in-sample optimizer adds no out-of-sample value. A real,
   honestly-measured edge, but an unremarkable one. (The prior 0.483/+12.61% was inflated
   by a corrupted price cache, now fixed.)
5. **Heavy LLM-in-the-loop dependence** (Opus/Sonnet/Haiku + OpenRouter + external MiroFish
   server) — cost, latency, nondeterminism, hallucination risk. A large amount of code
   exists purely to prevent the LLMs from hallucinating, which itself signals the risk.
6. **Operational fragility:** single operator, runs on a personal Mac via launchd, fragile
   external dependencies, paper only.
7. **Measurement has been broken repeatedly** — many recent commits fix counterfactual/
   authority bugs. Reasonable to ask whether *any* current number is final.
8. **"Earn authority over 21 rebalances" but only ~7 real rebalances have happened** (with
   one date duplicated 8×). The governance story is mostly still theoretical.

---

## 7. Single most defensible claim

> **"It's a fully-automated multi-agent trading system that has run live on paper for ~3
> months and grades its own AI decision-layer against a pure-quant counterfactual every
> day — and because that honest measurement currently shows the AI subtracting ~6
> percentage points, the system has correctly capped the AI at 5% authority and refused to
> promote it."**

**Why this is the defensible one:** it leads with the *validation discipline*, which the
evidence actually supports (`counterfactual_daily.jsonl`, `earned_authority.json`, and
`ai_pm_perf_feedback.json` all show the promotion gates failing and the system *not*
promoting). It makes no claim about returns or alpha that the data doesn't back. To a
sophisticated investor, "I built a system that catches itself when the AI isn't working,
and acts on it" is more credible — and rarer — than any performance number this repo can
currently stand behind.

---

### Pointers to source-of-truth files (for anyone re-checking)
- Live equity / SPY: `logs/holdings_log.jsonl`
- AI PM tracks: `logs/counterfactual_daily.jsonl`, `data_cache/earned_authority.json`,
  `data_cache/ai_pm_perf_feedback.json`
- Rebalances: `logs/eod_log.jsonl`
- WF backtest: `outputs/wf_results/wf_report_2026-06-04.json` (+ `..._baseline_corrected_...`)
- Debate verdicts: `outputs/debate_log/verdict_YYYY-MM-DD.json`
