# Data-Integrity & Separation-of-Claims Audit
**Date:** 2026-06-22 · evidence pulled from repo files/git, not docs.
**System 1 = pure quant. System 2 = AI-native layer (weeks old).**

> **UPDATE 2026-06-22 (RESOLVED).** Q1's reconciliation below still holds (the README WF
> table was a stale splice), and the deeper cause — a **corrupted `prices_live` cache**
> (~59% duplicate rows + 10× errors in 12 symbols) — has now been fixed by a **clean
> single-source re-fetch + WF re-run**. The "Sharpe 0.483 → VERIFIED" finding below means
> only "matches the 2026-06-04 artifact," **not** "correct": that artifact was computed on the
> corrupted cache and is now **superseded**. The verified clean number is **Sharpe 0.41 /
> CAGR +10.3% / +1pp excess vs SPY / max DD −32.9% / WFE −0.65** (OOS 2021-01→2026-01, 21
> folds; artifact `outputs/wf_results/wf_report_clean_2026-06-22.json`). The "6yr OOS"
> framing was also wrong: the real OOS window is 2021→Jan 2026. **Single source of truth for
> all numbers: [`CURRENT_VERIFIED_NUMBERS.md`](CURRENT_VERIFIED_NUMBERS.md).**

---

## Q1 — WF backtest discrepancy: RECONCILED. The README table is a splice of two different runs.

**Mechanism (git blame on `README.md` lines 96–101):**
- L98–100 `CAGR +12.61% / Sharpe 0.483 / Alpha +2.54%` → commit `8e120143`, **2026-06-16**.
- L101 `Max Drawdown −23.4%` → commit `d42c02d0`, **2026-05-23** (value first entered `44801ca`, **2026-05-10**).

The drawdown line was **never updated** when the row above it was refreshed on June 16. They are from different runs ~3–4 weeks apart.

**What the artifacts actually contain** (`outputs/wf_results/`, all generated Jun 3–4):
| File | Sharpe | CAGR | regr. alpha | **max DD** | Sortino | WFE | beta |
|---|---|---|---|---|---|---|---|
| `wf_report_2026-06-04.json` (= `..._current_...`) | **0.483** | **12.61%** | +5.55% | **−36.8%** | 0.05 | **−0.217** | 0.733 |
| `wf_report_baseline_corrected_2026-06-04.json` | 0.159 | 8.73% | +0.44% | −46.9% | 0.256 | +0.015 | 1.03 |
| `wf_report_hs_2026-06-04.json` (closest DD) | 0.138 | 4.37% | −2.39% | −27.3% | 0.013 | −41.5 | 0.66 |

**Findings:**
- **Sharpe 0.483 / CAGR 12.61% → VERIFIED.** Exact match to `wf_report_2026-06-04.json`.
- **Max DD −23.4% → STALE / UNSUPPORTED.** No artifact in the repo shows −23.4%. The real drawdown for that exact run is **−36.8%**. Closest variant is −27.3% (a different config). The −23.4% is a leftover from a pre-May-23 run whose artifact no longer exists.
- **Alpha +2.54% → UNVERIFIABLE.** The artifact's `alpha` field is **+5.55%** (annualized regression alpha vs SPY). +2.54% is *plausibly* CAGR-minus-SPY-CAGR (12.61% − ~10.07%), a different definition — but **no SPY CAGR is recorded in any artifact**, so it cannot be confirmed from the repo.

**Single trustworthy set (the documented "current" config, `wf_report_2026-06-04.json`):**
> Sharpe **0.483** · CAGR **+12.61%** · Sortino **0.05** · regression alpha **+5.55%** (beta 0.733) · **max DD −36.8%** · win rate 49.6% · **WFE −0.217** · 21 folds / 1,134 OOS days.

**Bluntly:** the README table cannot be trusted as printed. The drawdown is definitively wrong (should be −36.8%), and the alpha number cannot be sourced. **You need a fresh WF run** to publish an honest alpha-vs-SPY and drawdown — and see Q2 before re-running.

---

## Q2 — System boundary on the backtest: CLEAN for the documented run, with one forward landmine.

**Verified pure-quant:**
- `grep` for `ai_pm|debate|counterfactual|adversarial|earned_author|judge|prethesis` across `ascent/research/wf_framework/` and `walk_forward_runner.py` → **zero hits.**
- `grep` for `anthropic|llm|generate_|openrouter|sonnet|opus|haiku` across `wf_framework/` → **zero hits.**
- `ascent_strategy.py` imports only `alpha.stack`, `features.build_features`, `portfolio.optimizer` (lines 47–49).

**The one nuance — the `llm_fundamental` sleeve:**
- It exists at 3% in `DEFAULT_ALPHA_WEIGHTS` (`ascent/alpha/stack.py:23`) and is loaded by `build_alpha_stack` (`stack.py:311`), which the WF strategy calls at `ascent_strategy.py:184`. `llm_fundamental_alpha` **does call Claude Haiku** (`ascent/alpha/llm_fundamental.py:27,102`).
- **But it was inert for the documented run:**
  1. `data_cache/llm_fundamental_cache.json` keys span **2025-04-01 → 2026-04-14 only** — no signals exist for the 2020–2025 backtest window.
  2. The sleeve emits a single cross-sectional snapshot for the latest date, not a per-fold historical series — structurally it can't feed 2020–2025 folds.
  3. Per `CLAUDE.md` (2026-06-19 session): it "had always returned empty before → silently skipped" until a raw→ratio fix on **June 19**. The WF artifact is dated **June 4 — before activation.**

**Conclusion:** The documented WF backtest had **zero AI-native involvement.** Verified.

**⚠️ FORWARD LANDMINE:** If you re-run WF now (post-June-19), `llm_fundamental` will **activate and call Haiku live**, putting an LLM into the backtest decision path. Worse, it would score 2020–2025 fundamentals with a **2026 model and no point-in-time data before 2025-04** — i.e. an anachronism/look-ahead. **Explicitly zero `llm_fundamental` (and `narrative_alpha`, `stack.py:392`) before re-baselining**, or your fresh number is no longer pure-quant or look-ahead-clean.

---

## Q3 — True AI-native activity calendar (from logs/git, not the narrative)

| Component | First real output | Live in production | Rebalances actually participated in |
|---|---|---|---|
| Debate (old format) | verdict `2026-04-04` | Apr 2026 | Apr 15 |
| Debate (adversarial redesign) | `adversarial_interventions.jsonl` first row **2026-05-27** | May 27 | May 27, Jun 10, Jun 15, Jun 22 |
| AI PM Phase 1 (shadow/pre-thesis) | `ai_pm_calibration.jsonl` **2026-05-17** | shadow only | none (advisory) |
| AI PM promoted L1 (5% authority) | `earned_authority.json` `level_start_date` **2026-06-04**; daily_views from Jun 4 | Jun 4 | — |
| AI PM **blended-portfolio decision** | `ai_pm_decision_log.jsonl` — **only 2026-06-10** | Jun 10 | **exactly 1 (Jun 10)** |
| Counterfactual Track D (pure AI PM) | first non-null **2026-05-19** | — | 23 valid days → Jun 22 |

**Hard facts:**
- **Debate did NOT run for 2 of 7 rebalances:** there is **no verdict file for May 5 or May 19** (rebalances confirmed in `eod_log.jsonl`). Verdicts exist only for Apr 15, May 27, Jun 10, Jun 15, Jun 22.
- **AI PM has exactly ONE logged blended decision (June 10).** `ai_pm_daily_views.jsonl` has rows through Jun 22, but `ai_pm_decision_log.jsonl` contains **only June 10 entries (8 rows = the overnight reruns).** There are **no decision-log entries for the Jun 15 or Jun 22 rebalances** despite a fix on June 10 that was supposed to "write on every rebalance including fallback." **Unresolved — either AI PM didn't blend, fell back silently, or logging regressed. Flag for investigation.**
- **Real age of System 2 with actual capital authority: ~2.5 weeks** (since Jun 4), one logged rebalance. Debate is older (~10 weeks) but ran in only ~5 rebalances. None of this resembles the 6-year quant record.

---

## Q4 — Places docs put quant and AI numbers where they can be conflated

1. **`README.md` — THE conflation risk.** The "Walk-Forward Out-of-Sample Results" table (L90–101, **pure quant, 6-year**) and the live-paper headline (`Total Return +8.82%`, `Sharpe 1.794`, the **actual blended book**) sit in the same README under an "AI-native fund" framing, **with no hard statement that the 6-year/Sharpe-0.483 record is 100% pre-AI.** A reader can easily believe the AI-native system has a six-year track record. **It does not — the AI layer is ~2.5 weeks old.** This is the single most dangerous instance.
2. **Live Sharpe 1.794 is itself ambiguous.** It's the *actual book* = System 1 (dominant) + 5% System 2. It is neither pure quant nor a measure of the AI PM. Nobody should attribute it to the AI layer — but its placement invites it.
3. **`CLAUDE.md` "Current state"** lists `WF OOS Sharpe 0.483` in the same block as AI PM authority/counterfactual discussion. Same conflation, internal doc.
4. **Top-of-README badges:** `AI PM Level 1`, `alpha sleeves 14`, `tests 977` adjacent — lower risk, but the AI-PM badge next to performance framing reinforces #1.

**Recommendation:** put a one-line firewall under the WF table: *"This 6-year OOS record is the pure quant engine. The AI-native layer (debate, AI PM) has been live since June 2026 and is NOT reflected in these numbers."*

---

## Q5 — Authority gates: HELD CONSTANT, not gamed.

Thresholds (`ascent/strategy/ai_pm_perf_feedback.py:240–243`):
| Transition | sortino_edge | hit_rate | profit_factor | min_decisions |
|---|---|---|---|---|
| L1→2 | 0.20 | 0.52 | 1.2 | 5 |
| L2→3 | 0.30 | 0.55 | 1.3 | 8 |
| L3→4 | 0.40 | 0.55 | 1.3 | 10 |
| L4→5 | 0.50 | 0.58 | 1.4 | 15 |

- `git log -S'"sortino_edge": 0.20'` → **one commit only** (`19ff6d7`, 2026-06-04 creation). **Never modified.**
- The only later edit to the file (`8defa51`, Jun 20) **did not touch the threshold dict** (diff is empty for those lines).
- **No loosening or special-casing since the AI PM began failing.** Gates are honest.
- Current state (`ai_pm_perf_feedback.json`): failing **4 of 7** — sortino_edge −3.73, hit_rate 0.0, profit_factor 1.0, **min_decisions 0** (it has evaluated zero decisions, so it is structurally unpromotable right now). Passing fade_rate, regime_gate (not yet evaluated), cooldown. **The system is correctly refusing promotion.**

**One adjacent caveat (not a gate, but a loosening):** `earned_authority.py` `ADVANCE_WINDOW` was reduced **10 → 5 rebalances** on 2026-06-01 (`9f3d9df`) — fewer rebalances required to advance. This predates promotion (Jun 4) and any failure, so it's **not** a reaction to failing, but it is a loosening of cadence you should be aware of.

---

## Q6 — External dependency failures (MiroFish) and log corruption risk

**MiroFish is essentially non-functional in the recorded pipeline, BUT it does not corrupt any performance number.**
- MiroFish data appears in **only 1 of 4 saved rebalance verdicts** (`verdict_2026-06-10.json`); 05-27, 06-15, 06-22 have **none**. No MiroFish/alignment entries in `multi_agent_run.jsonl` or the Jun-19 console log. `CLAUDE.md` documents repeated failures (OpenRouter 402s, prepare timeouts, Chinese-language report breaking the sentiment parser).
- **Critical:** MiroFish output feeds only the **debate / AI-PM layer (System 2)**. It does **not** write to `counterfactual_daily.jsonl`, `holdings_log.jsonl`, `*_pnl.jsonl`, or the WF artifacts. So its failures **degrade System 2 reasoning quality, but corrupt zero metrics.** No performance number is unreliable because of MiroFish.

**The real log-integrity risk in the last 30 days is self-inflicted, not MiroFish.** Per `CLAUDE.md` sessions Jun 18–20 (with `.bak` backups in `logs/`):
- `counterfactual_daily.jsonl` had **duplicate rows** (Jun-10 rerun wrote ~9 dupes), **Track A★/D frozen at 0.0/null** Jun 8–18, a **Track B disjoint-window artifact** (only 2 of ~38 days overlapped A★), and **fabricated 0.0 authority-buffer entries**.
- These were repaired, but **the repairs materially changed the headline at least 3 times in 5 days**: D−A★ went −2.33pp → −2.89pp → −6.06pp → **−6.52pp (current)** as bugs were healed.

**Dates/metrics to treat as unreliable:**
- **Any counterfactual or authority number dated before ~2026-06-20** (pre self-heal).
- The current AI-PM counterfactual (**D−A★ −6.52pp / n=23**, B−A★ −5.27pp / n=31) is only as of the Jun-20 heal, **small sample, and revised 3× in a week** — low confidence, directional at best.
- The WF numbers are **clean of this** (generated Jun 3–4, independent of the counterfactual layer) — their problem is staleness/splicing (Q1), not corruption.

---

## Bottom line
- **System boundary is architecturally clean** (Q2): the 6-year backtest had zero AI involvement — but re-running it now would break that, and the docs (Q4) blur the two systems for a reader.
- **The numbers you should not ship as-is:** README WF max DD (−23.4% is wrong; real −36.8%) and alpha (+2.54% unsourceable); any pre-Jun-20 AI-PM counterfactual.
- **The gates are honest** (Q5) and **MiroFish hasn't corrupted your metrics** (Q6) — the corruption risk was internal and is (claimed) repaired, but the AI-PM signal is n=23 and has moved a lot.
- **The one-sentence firewall** the docs are missing: *the 6-year record is pure quant; the AI layer is ~2.5 weeks and one logged rebalance old.*
