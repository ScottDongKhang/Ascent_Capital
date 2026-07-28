# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio → walk-forward → regime → 4 specialist agents → orchestration → AI PM (earned autonomy) → debate → Alpaca paper trading.

**Daily command**: `python run_all_agents.py` (branches on rebalance day)

---

## Environment

Python 3.12.13, venv at `.venv/`. Always use `.venv/bin/python`. API keys via `APIKeys.from_env()`. Config via `get_config()` — never `Config()` directly.

---

## LLM models (`ascent/llm/client.py`)

Import constants from here — never redefine locally:
```python
DEFAULT_MODEL = "claude-opus-5"             # AI PM synthesis (Phase 2)
SONNET_MODEL  = "claude-sonnet-5"           # debate agents, red team, pre-thesis (Phase 1)
HAIKU_MODEL   = "claude-haiku-4-5-20251001" # classifiers, weight adjustment
```

**Claude 5 rules (migrated 2026-07-27 from the 4.6 generation):**
- **Never index `resp.content[0].text`** — thinking is ON by default, so block 0 is
  usually a thinking block. Use `from ascent.llm.client import extract_text`.
- **Never pass `temperature` / `top_p` / `top_k` to the API** — 400 on Claude 5. The
  `temperature` kwarg still exists on the wrappers for call-site compatibility and is
  silently dropped for Claude 5 models; steer with prompts instead.
- **Never pass `thinking={"type": "enabled", "budget_tokens": N}`** — 400. Depth is set
  by `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`). Wrapper defaults:
  `chat_completion`/`generate_structured` = medium, `extended_thinking_completion` and
  `tool_completion` = high.
- **`max_tokens` caps thinking + visible text together.** The wrappers raise it to
  `_MIN_TOKENS_WITH_THINKING` (4096); direct `messages.create()` callers must leave
  their own headroom or thinking will consume the whole budget and return no text.
- **Keep thinking enabled in tool loops.** With thinking disabled, Claude 5 can emit a
  tool call as visible text instead of a `tool_use` block — the turn succeeds and the
  tool silently never runs.
- **Echo thinking blocks back unchanged** when continuing a turn (handled inside
  `tool_completion`); dropping or editing them is rejected.
- Haiku 4.5 is already current and is unchanged — it keeps the legacy parameter path.

---

## Key layout

```
ascent/           core engine (config, data, features, alpha, portfolio, backtest,
                  research, regime, risk, reporting, execution, monitoring, llm,
                  dashboard, strategy, causal)
agents/           us_equities_agent, macro_agent, international_agent,
                  alternatives_agent, ai_pm_agent, red_team_agent
orchestrator/     central_intelligence.py
debate/           debate_runner, agents, judge
run_all_agents.py daily entrypoint
ascent/main.py    pipeline entrypoint (called by each agent)
data_cache/       parquet caches, earned_authority.json, active_alpha_config.json,
                  ai_prethesis_latest.json, ai_pm_pattern_memory.json
logs/             eod_log.jsonl, multi_agent_run.jsonl, ai_pm_calibration.jsonl
outputs/debate_log/  verdict_YYYY-MM-DD.json
```

---

## Integrity constraints (never violate)

1. No look-ahead bias — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only.
2. No simulated data under live cache names (`prices_live` = Yahoo live only; fallback → `prices_live_fallback_simulated`).
3. Max-weight hard cap via `_water_fill_cap()` with post-condition check.
4. Sector constraint: < 80% coverage → skip caps + warn, never collapse to single name.
5. Debate is advisory only *at the module level* — nothing under `debate/` writes to alpha,
   portfolio, or execution modules. The one sanctioned exception: the judge's single
   `position_changes[0]` (max ONE position per rebalance) is applied and written to
   `execution/merged_weights.json` by `run_all_agents.py` itself (~line 1793-1868), not by
   `debate/`. The change is capped by earned authority (`adversarial_authority.py`,
   currently frozen at `low` = 1.0pp max per intervention since `n_scored: 0` for every
   type — see W4). Treat this as a bounded, authority-gated exception to advisory-only,
   not a violation of it — do not let debate code itself acquire write access elsewhere.
6. New alpha sleeves: update `DEFAULT_ALPHA_WEIGHTS` in BOTH `ascent/alpha/stack.py` AND `ascent/research/self_improve.py`.
7. Fundamental sleeve is disabled (IC-t = −4.75, anti-signal) — do not re-enable without positive IC-t.

---

## Non-obvious gotchas

- **Agent module names**: `agents.us_equities_agent` not `agents.us_equities` — `_agent` suffix required for all four.
- **`RegimeEngine` constructor**: takes `config=dict`, not a `Config` object.
- **`bdate_range(end="today")`**: returns empty on weekends — use explicit weekday rollback.
- **`apply_hedge_overlay`**: must accept both `RegimeSignal` and plain `str`.
- **ML sleeve cache**: must store `feature_names` — XGBoost crashes on shape mismatch if feature set changes between writes.
- **AI PM two-phase**: Phase 1 = `run_ai_pm_prethesis()` uses `SONNET_MODEL`. Phase 2 = `run_ai_pm(prethesis=...)` uses `DEFAULT_MODEL` (Opus). Never swap — Sonnet for breadth, Opus for judgment.
- **`propose_prethesis` vs `propose_portfolio`**: different tools, different result stores. Phase 1 ends with `propose_prethesis`, Phase 2 with `propose_portfolio`.
- **Pre-thesis runs before quant agents** in `run_all_agents.py`. Failure → `prethesis=None` → graceful single-phase fallback.
- **`run_ai_pm(quant_outputs=...)`**: pass `agent_outputs` list or AI PM re-runs all 4 agents (~160s wasted).
- **`ascent/main.py` returns 10-tuple** (adds `_alpha_breakdown`) — `eod_runner.py` and `us_equities_agent.py` must unpack correctly.
- **`_SPARSE_FILL_ZERO`** in alpha stack: must include ALL sparse panels or NaN-drop silently disables those sleeves.
- **PDF generation**: use `reportlab` — `weasyprint` requires system GObject/Pango unavailable here.
- **PDBC↔KMLM correlation (~0.81)**: frequently triggers orchestrator correlation guard → KMLM halved. Expected, not a bug.
- **Kill switches** (pending paper validation ~July 2026): `EVENT_TRADING_ENABLED=False`, `TWAP_ENABLED=False`, `SELF_MODIFY_ENABLED=False`, `LONG_SHORT_ENABLED=False`.
- **Dashboard subprocess**: spawn with `cwd=repo_root` + `PYTHONPATH=repo_root` — `ascent` isn't pip-installed, so running from `scripts/` breaks all imports.
- **Same-day Track B is unreliable**: Alpaca 1D bars settle ~17:00 PT; the 1:45 PM run sees `equity == last_equity` → fake 0.0. Use `alpaca_broker.get_portfolio_history()` for settled returns.
- **`loguru` not installed** — use `import logging` throughout; never `from loguru import logger`.
- **MiroFish on rebalance days**: LiteLLM proxy (port 4000 → Haiku) must be running (`/Users/scott/MiroFish/.env`). OpenRouter leaves `max_tokens` unset → 402 if credits low. Top up at `openrouter.ai/settings/credits`.
- **Discovery mini-rebalance is now add-only**: `_insert_candidate_weights` (not a full agent re-run). Suppressed within 3 trading days of next scheduled rebalance via `_is_near_scheduled_rebalance(window=3)`.
- **`run_eod_with_weights()` silently no-ops on non-rebalance days** — pass `force=True` for discovery/mini-rebalance paths.
- **AI PM decision log only appears on scheduled rebalances** (`is_rebalance=True`). Off-calendar discovery days run daily-view path, not Phase 2 — no decision log entry is expected or a bug.
- **`prices_live` was corrupted** (~59% dup rows from yahoo+yfinance_hub blending). `save_parquet` now normalizes the date dedup key. Existing cache still needs a clean re-fetch. See `CURRENT_VERIFIED_NUMBERS.md`.

---

## Data / caching

Cache name provenance — never hide: `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`, `macro_live`/`macro_simulated`, `profiles` (sector metadata).

Point-in-time joins: always use `as_of_join()` / `as_of_merge()` for cross-dataset alignment.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Rebalance recap (required after every rebalance)

After **any** run of `run_all_agents.py` that submits orders, write a four-part recap
unprompted. This is the primary interface to what the system decided — a status line is
not sufficient.

1. **Reasoning behind the decision.** Read `outputs/debate_log/verdict_YYYY-MM-DD.json`
   (`verdict.reasoning`, `verdict.key_risks`) and the adversarial intervention. Explain
   why *these* trades, which argument won which exchange, and what the judge explicitly
   declined to do. **Then verify execution matched the reasoning** — on 2026-07-27 the
   judge argued against cutting UUP/TLT 48h before FOMC and the `reduce_size` fallback
   (`[EodRunner] reduce_size: Haiku only reduced 0 positions — forcing trim on top
   positions`) sold them anyway. Quote the reasoning; don't paraphrase it away.
2. **Things to watch for.** Live catalysts (FOMC, earnings, ex-div), positions on thin
   ice, guards that nearly fired, data sources that were down, anything that changes the
   read next run.
3. **Performance since the last rebalance.** Portfolio vs SPY over the window, with
   per-position attribution when a few names dominate. Use
   `alpaca_broker.get_portfolio_history()` (settled bars) — never same-day
   `equity − last_equity`, which reads a fake 0.0 before ~17:00 PT.
4. **Why performance looks like that.** The causal story, separating sizing/structure
   from stock selection from things outside the model's control.

**Ground every number in a real artifact and flag anything reconstructed.** A confident
wrong number is worse than an acknowledged gap: on 2026-07-27 drawdown was reported as
−8.5% from a synthetic equity curve while the kill switch's actual NAV/peak read 4.7%.

---

## Current state (as of 2026-06-22)

- **AI PM**: Level 1 (Analyst, 5% authority). **As of 2026-07-27: B−A★ = −7.82pp/38d, D−A★ = −6.34pp/29d** (was −5.27pp/−6.52pp in June — the gap widened). Pure quant +23.59% beats both actual (+16.03%) and SPY (+16.63%); pure AI PM is worst at +11.54%.
  **Diagnose transmission before judging judgment.** On 2026-07-27 the judge's reasoning was sound (flagged a real correlation risk from the model's own contradiction: quant VaR −0.96% vs regime-flip −7.1%) but never reached the weights — `reduce_size` fired, the Haiku adjustment trimmed 0 positions, and a size-sorted fallback cut UUP/TLT, the exact positions the judge argued to protect. Disabling the layer is right only if the judgment is bad, not if a fallback is discarding it.
- **No alpha vs SPY is structural**: ~22% defensive non-equity sleeves + 200MA cut + 15% vol-target overlay cost beta in an equity-only bull. WF OOS confirms positive risk-adjusted alpha; raw-return lag is by design.
- **WF OOS**: ✅ VERIFIED (2026-06-22) on a clean re-fetched cache — **Sharpe 0.41, CAGR +10.3%, +1.0pp excess CAGR vs SPY, max DD −32.9%, beta 0.73**, OOS 2021-01→2026-01 (1134 days, 21 folds). WFE −0.65 (overfit — disclose); engine Sortino field buggy (real ≈0.68, don't cite). Supersedes the corrupted-cache 0.483/12.61%. Artifact `outputs/wf_results/wf_report_clean_2026-06-22.json`. Single source of truth: `CURRENT_VERIFIED_NUMBERS.md`.
- **Regime**: calm_bull.
- **Open**: clean cache is staging-only (`prices_live_clean_refetch.parquet`) — production `prices_live` NOT swapped (would change live close to total-return-adjusted before Jun 24). Verify the *ML-sleeve internal* "CPCV C(6,2)=15" alpha-table line (the WF-results-header CPCV line is corrected). Jun 24 = next scheduled rebalance.
- **GitHub Pages**: `https://scottdongkhang.github.io/Ascent_Capital` — auto-updated after every daily run.

---

## Session log

> Prior sessions archived to `docs/session_log_archive.md`.

### 2026-07-04 (ascent-agri — first real user feedback, shipped same day)
- **Chris Kornman (The Crown / Royal Coffee education) replied to Scott's outreach**: praise ("quite advanced for a high school student"), will refer people to the site, open invite to chat. His substance — the trade thinks in near-to-long-term trends (not intraday) and Royal is ~arabica-only — was built and deployed same day: "The long view" (full-history KC=F vs 200-day average) + "Growing conditions — Sul de Minas, Brazil" (rainfall anomaly + tmin with frost-risk line) live on the site. 156 tests.
- Threaded reply drafted in Gmail (accepts video-call offer; no URLs in body — Gmail API rewrites bare URLs into google.com/url tracking redirects, discovered when Scott flagged the sent emails looked scammy; fresh-compose checklist at outreach-drafts/00-SEND-CHECKLIST.md is the fix).
- Outreach round 1 fully sent 7/3 (Nguyen auto-ack promises human reply; DCN via form; Cafe Imports quiet). Log: outreach-drafts/log.csv.

### 2026-07-02 (ascent-agri build — separate repo, NO Ascent Capital code touched)
- Built the full backtest pipeline in the standalone `/Users/scott/Downloads/ascent-agri/` CALS artifact per the port audit: regime pkg (model/decision/posture/breaks near-verbatim; features/engine/integration rewritten for coffee + BRL/USD + Central Highlands weather), alpha pkg (trend/meanrev with rolling TS z-scores replacing cross-sectional, vol_sizing, meta_learner, minimal long-only stack), backtest engine/costs, research splits/cpcv/evaluation + single-series walk_forward_runner rewrite, demo.py + demo.ipynb, 64 new tests (120 total pass).
- **Two latent Ascent Capital bugs surfaced during the port (NOT fixed here — Ascent repo untouched):** (1) `ascent/regime/model.py` `_MarkovModel.filtered_probs()` ignores its input and returns TRAIN filtered probabilities — only matters if the Markov backend is ever asked to score new data (equities always uses the HMM backend for K=3); (2) `ascent/backtest/engine.py` weight drift renormalizes by `drifted.sum()` alone, which silently drops the cash bucket for portfolios at fractional gross exposure (fine when weights sum to ~1).
- Data: BRL/USD via Yahoo BRL=X (FRED unreachable from this network — fetcher tries FRED first), Open-Meteo weather for Buon Ma Thuot. Robusta bottleneck unchanged: 1 contract, no DATABENTO_API_KEY available.
- Files: all in ascent-agri repo + this CLAUDE.md entry. Open (ascent-agri): backfill robusta contracts via Databento when key exists; consider porting the two bug fixes back into Ascent Capital.
- **VIETNAMESE GROWER PAGE (2026-07-03, community impact):** `/vi/` on the live site — focused farmer-facing page in Vietnamese (crop stage + stage-weighted weather risk, rainfall anomaly, market state in plain words, farm-gate đồng/kg with VN number formatting via `transmission_line_vi`), generated by the same daily build; EN|VI toggle both pages; footer flags draft translation + invites corrections. Roadmap: native-speaker (family) review — quality control AND essay material. 154 tests.
- **AGRONOMY LAYER (2026-07-03, ag-science spike for CALS):** `ascentagri/agronomy/` — `phenology.py` (Dak Lak robusta crop calendar: flowering Jan–Mar drought-critical, early fruit Apr–May, filling Jun–Sep, harvest Oct–Dec wetness-sensitive; rainfall anomaly → stage-weighted crop stress index) + `economics.py` (USD/VND optional cache, futures→VND/kg farm-gate transmission, honestly labeled futures-equivalent). Site/brief/API now carry crop stage + stress band + farmer-margin line. `docs/APPLICATION-ANGLES.md` = CALS ag-science vs UC econ framings + claims inventory. Key line: "a rainfall z-score is statistics; knowing the same deficit ruins February flowering but speeds November drying is agronomy; measuring whether the market already knows is economics." 153 tests. Outreach emails drafted earlier same session (writing-as-scott skill + 4 researched targets in Gmail drafts, outreach-drafts/ gitignored).
- **DATA PRODUCT LAYER (same session, corporate-adoption kit):** site now emits `api/latest.json` (versioned schema: regime/posture/anomalies/brief) + `api/history.csv` (2,134 days of derived series — labels, risk multipliers, rain anomalies; NO raw market data redistributed). LICENSE added (MIT code / CC BY 4.0 derived data — unlicensed repo would legally block corporate use). `weekly-research.yml` also snapshots GitHub traffic → `data/ledger/traffic.jsonl` (timestamped usage evidence). `docs/OUTREACH.md` = tiered corporate outreach playbook (specialty importers/roasters first, no Reddit) + email template + the evidence checklist gating any "X users depend on my data" claim. 143 tests.
- **DAILY LEDGER LOOP (same session, Ascent daily-agent pattern applied to ascent-agri):** `ascentagri/ledger.py` — append-only public track record (`data/ledger/forecasts.jsonl`, committed to main by the daily workflow): regime call + target exposure recorded BEFORE outcomes, scored later from the ledger's own prices with 1-day execution delay, never edited. Site section "The ledger — the model in public" (chart activates at 10+ scored days). New `weekly-research.yml` (Mondays): re-runs weather study + WF on fresh data, commits artifacts — evaluation refresh, explicitly NOT parameter retuning. `CONTRIBUTING.md` opens the community on-ramp (new regions/crops/composites) with integrity ground rules. 141 tests.
- **RESEARCH LAYER (same session, Ivy-level upgrade):** built `ascentagri/research/weather_study.py` — a-priori event studies with permutation inference + placebo controls (VN dry→robusta, BR dry/cold→arabica, cross-placebos) — plus working paper `docs/research/weather-and-coffee-returns.{md,pdf}` with honest results: BR dry→arabica +5.1pp/5d (n=4, p=.04 uncorrected, disclosed as not surviving Bonferroni), placebo null, VN→robusta untestable (0 events in 17-month robusta span → quantifies the contract-backfill case), and the 2021 frost (+33.8%/5d) absorbed by the event cooldown rule (documented as event-definition risk, not retro-fixed). Site gained a Research section (PDF served) + RSS feed of daily briefs. `docs/ROADMAP-to-Nov-1.md` = Scott's dated critical path (July: contracts grind + GoatCounter + share URL; Aug: robusta re-run + paper v2 + optional JEI submission; Sep: essays + named users; Oct 15 feature freeze). 133 tests pass.
- **SHIPPED PUBLIC (same session, for real users):** created `github.com/ScottDongKhang/ascent-agri` (public) using the keychain credential and deployed the **Robusta Coffee Monitor** to `https://scottdongkhang.github.io/ascent-agri/` — dark-editorial daily page (regime posture + deterministic plain-English brief, price w/ regime shading, Buon Ma Thuot rainfall anomalies, BRL/USD driver, honest methods). Weekday cron (`update-site.yml`, 21:20 UTC) refetches + republishes; fail-safe = no successful fetch → no publish (stale-but-correct). GoatCounter snippet wired to `ascent-agri.goatcounter.com` — **Scott must claim that code (free signup) to see visitor counts.** 124 tests pass.

### 2026-06-24 (scheduled rebalance run + KLAC price-corruption root-cause → trend sleeve un-gated)
- **Ran** `run_all_agents.py` (scheduled rebalance). Verdict PROCEED (conf 0.58); 24 orders submitted, 0 skipped; kill switch OK (DD 0.8%); AI PM Level 1 / 5% / 0 overrides; day attribution +0.57% vs SPY −0.05%.
- **Investigated two anomalies in the run:**
  1. **IC gate zeroed the 70% trend sleeve** (`rolling mean_ic=-0.0641 < -0.005`). ROOT CAUSE: **KLAC** carried a recurring **×10 price error** in `prices_live` (351 up + 351 down implausible moves across 2020–2026; pipeline panel saw 702). Momentum ranks KLAC top after each fake +900% → the ~−90% revert follows → **trend IC driven strongly negative (−0.093, t −56)** while meanrev mirror-inflated (+0.072, t +55). Because logged `mean_ic` is a **cumulative all-history mean**, a bad source-blend ~6/13–15 stepped it negative on **6/15** — gate has zeroed trend every run since (6/15, 6/22 discovery, 6/24 rebalance).
  2. **AI PM force-seal** (both phases): designed fallback fired because the agentic loop exhausted `max_tool_calls` without `propose_portfolio`. Contributors: Falsifier Haiku JSON parse failure + neutral MiroFish (0.50/mixed). Net: force-sealed near quant baseline, 0 overrides — minimal original judgment today. Not corrupting; flagged.
- **FIX (data, surgical + reversible):** backed up `prices_live` → `data_cache/prices_live.pre_klac_fix.20260624-160508.bak.parquet`; fresh-fetched KLAC split-only (auto_adjust=False, 0 anomalies, $11–269 real range) and **replaced KLAC in place** (4,868 blended rows → 1,627 clean). Split-only basis preserved for the other 935 symbols (no TR swap, no discontinuity). **Verified:** trend `mean_ic` −0.093 → **−0.0025 (gate no longer fires)**, meanrev +0.072 → +0.012. The live source returns clean KLAC — corruption was purely old accumulated rows.
- **Deeper bug found, NOT yet fixed (durable class-fixes):** `save_parquet` append+dedup is **silently failing** — `prices_live` holds 3 blended source generations (`yfinance_hub` 2.98M + `yahoo` 1.43M + `yahoo_hub` 58K = 4.46M rows, ~3× bloat) because the tz-normalized dedup key doesn't match across fetches (mixed tz-aware/naive `date` → object dtype → normalization skipped). This is what let the 10× KLAC copy linger and get arbitrarily picked by the pivot. Also: **IC gate has no robustness to single-symbol outliers** — one corrupt symbol can disable a 70% sleeve (no winsorization / breadth requirement).
- Files: `data_cache/prices_live.parquet` (KLAC repaired), backup `.bak.parquet` (rollback). Open: (a) fix `save_parquet` dedup + collapse the 3-generation bloat; (b) harden IC gate vs outliers; (c) AI PM Falsifier JSON / tool-budget; (d) confirm next real run logs trend ungated & positive.
- **FOLLOW-UPS (a)+(b) DONE (TDD):** (a) `save_parquet` dedup now robust to **object/mixed tz-aware+naive `date`** via new `_calendar_day_key()` (was: `pd.concat` of aware+naive → object dtype → `is_datetime64_any_dtype` False → normalize branch skipped → dedup never fired = the 3-generation blend). One-time collapse of existing bloat: **prices_live 4.46M → 1.82M rows, 0 remaining dup (symbol,calday)**, KLAC stays clean; backup `prices_live.pre_dedup.*.bak.parquet`. (b) IC computation now winsorizes the fwd-return target cross-sectionally (1/99 pct, `_winsorize_rows()` in `main.py`) before `corrwith` — one data-error symbol can no longer drag a sleeve below the gate. **Validated:** clean-data gate decisions unchanged (no-op); on the corrupted backup, raw trend gates (−0.0148) but winsorized does NOT (+0.0010). Files: `ascent/data/store/parquet.py`, `ascent/main.py`, `tests/test_parquet_store_dedup.py` (+mixed-tz test), `tests/test_ic_outlier_robustness.py` (new). 6 new tests pass; full data/alpha/IC sweep 328 passed / 3 pre-existing fails (openbb-network, WF fixture IndexError, known-buggy Sortino — all fail on baseline too). Still open: (c) AI PM Falsifier JSON / tool-budget; (d) confirm next real run logs trend ungated.
- **FOLLOW-UP (c) DONE (TDD):** two independent issues from the run (Falsifier JSON + AI PM tool-budget — confirmed *separate*: `build_registry` runs in the rebalance flow, not the AI PM tool loop). (1) Both Haiku calls in `falsifier_registry.py` did `json.loads(text[start:end])` on the whole array → one missing comma OR truncation at `max_tokens` discarded EVERY falsifier and silently dropped the registry to all-news-watches (the run's `Expecting ',' delimiter: line 99 col 73`). New `_parse_json_objects()` scans balanced top-level `{...}` spans and parses each independently (skips malformed, drops truncated tail, survives the rest); `max_tokens` 800→2000 / 500→1500 to cut truncation. (2) `tool_completion` had no final-turn nudge — a model that over-grounds exhausts the budget into `[max iterations reached]` (empty result_store → caller force-seal), which is why AI PM Phase 1 & 2 both force-sealed. Added `_FINAL_TURN_NUDGE` injected into the tool-result turn on the penultimate iteration so the model gets one explicit chance to call its submission tool within budget (fires only when about to exhaust; happy path unchanged). Files: `ascent/strategy/falsifier_registry.py`, `ascent/llm/client.py`, `tests/strategy/test_falsifier_json_parsing.py` (new, 5), `tests/test_tool_completion_nudge.py` (new, 2). Sweep: 220 passed / 1 pre-existing openbb-network fail. Still open: (d) confirm next real run (Jul 8) logs trend ungated & that AI PM seals in-budget (nudge effect is live-only, not offline-testable).

### 2026-06-22 (clean price re-fetch + WF re-run — backtest number RESOLVED, no longer UNDER REVIEW)
- **What was wrong:** the WF OOS figure had been UNDER REVIEW because `prices_live` was corrupted (~59%/92% dup rows from 3 blended sources + 10×-type errors in 12 symbols). Could not cite a number.
- **What was done:** backed up the corrupt cache (`data_cache/_corrupt_backup_20260622-222216/`); re-fetched **all 938 symbols from a single source** (yfinance `auto_adjust=True`, total-return adjusted, 2020→Jun 2026) into staging `data_cache/prices_live_clean_refetch.parquet`. Verified clean: **0 dup (symbol,date) rows, 0 implausible jumps** (worst remaining are real events — GME/CAR/SHC/LUMN). The re-fetch repaired 11 of the 12 corrupt symbols; only **CHRD dropped** (irreparable source-side ticker-reuse history) → 936 symbols. Re-ran WF with `llm_fundamental`/`narrative` zeroed (logged `skipped`), same engine config as the canonical run.
- **New verified number:** **Sharpe 0.41 (independent recompute 0.417), CAGR +10.3% (recompute +10.4%), excess CAGR vs SPY +1.0pp (10.42% − 9.41%, shown), regression alpha +2.24%/yr, max DD −32.9%, beta 0.73, win rate 50.2%**, OOS 2021-01-08→2026-01-14 (1134 days, 21 folds). **WFE −0.65 (overfit — IS optimizer adds no OOS value; disclosed).** Artifact `outputs/wf_results/wf_report_clean_2026-06-22.json` (+ `wf_equity_clean_2026-06-22.csv`).
- **Sanity check (vs both priors):** lands far closer to the original 0.483 prior than the contaminated −0.14 — exactly as expected (identical 1134-day/21-fold window & beta 0.733; cleaning removes dup/spike-inflated trend signal → modest deflation). Even with dividends *added* (total-return close), CAGR *fell* 12.61→10.3%, confirming corruption was inflating returns. Engine **Sortino field is buggy** (0.042; real ≈0.68 — don't cite).
- **Judgment calls flagged:** (1) used total-return adjusted close (original used split-only) — lifts absolute CAGR ~equally for strat & SPY, leaves Sharpe/alpha ~flat; (2) did **not** swap the clean cache into production `prices_live` (would change live momentum signals to adjusted close before the Jun 24 rebalance) — staging only, corrupt cache backed up, live path dedupes on read.
- Docs updated: `CURRENT_VERIFIED_NUMBERS.md` §1 (UNDER REVIEW → VERIFIED), `README.md` WF section (+ corrected the WF-header "CPCV C(6,2)=15" → rolling 21 folds), this `CLAUDE.md`. Files: `data_cache/prices_live_clean_refetch.parquet` (new), `outputs/wf_results/wf_report_clean_2026-06-22.json` (new).
- Open: optionally swap staging cache → production (deliberate, post-Jun-24); verify the ML-sleeve internal CPCV line in the alpha-stack table (separate from the now-corrected WF header).

### 2026-06-22 (data-integrity audit — WF backtest UNDER REVIEW, prices_live corruption found + ingest fix)
- `prices_live` had ~59% dup rows (yahoo+yfinance_hub tz-keyed dedup never collapsed) + 10×-magnitude errors in 12 symbols. WF re-run blew to CAGR 10¹³%; repaired re-run (median-dedup, 27 folds) → Sharpe −0.14 but still source-blended. WF OOS now UNDER REVIEW.
- Live pipeline shielded: `ascent/main.py` dedupes on read (lines 393/745/789); WF framework does not.
- FIX: `ascent/data/store/parquet.py` `save_parquet` normalizes `date` dedup key to calendar day — stops future dup accumulation. Does NOT clean existing rows.
- Off-calendar Jun 15/22 AI PM blanks confirmed NOT a bug — discovery days use daily-view path, not Phase 2. True AI PM participation = 1/1 scheduled rebalances (Jun 10).
- Files: `ascent/data/store/parquet.py`, `tests/test_parquet_store_dedup.py` (new), `CURRENT_VERIFIED_NUMBERS.md` (new), `AUDIT_DATA_INTEGRITY.md`.
- Open: re-fetch `prices_live` clean; re-run WF; verify CPCV fold-count claim in README.

### 2026-06-22 (discovery churn fix — two guards on the off-calendar mini-rebalance)
- ROOT CAUSE: `_trigger_mini_rebalance` re-ran the FULL us_equities agent + orchestrator (32-order rotation for one new candidate) with no rebalance-calendar awareness — fired 2 days before the Jun 24 scheduled rebalance.
- FIX (TDD, 8 tests): (A) `_is_near_scheduled_rebalance(today, window=3, cal_path)` — suppresses discovery within 3 trading days of next scheduled rebalance (fail-open if calendar missing). (B) `_insert_candidate_weights(current, symbol, max_weight)` — ADD-ONLY: candidate gets 1/(n+1) slot, existing book trimmed pro-rata, max-weight capped. Full agent re-run eliminated from mini-rebalance path.
- Files: `run_all_agents.py`, `tests/strategy/test_discovery_guards.py` (new). 81 strategy+portfolio tests pass.
- NOT committed yet. Today's 32-order rotation already filled at Alpaca — left to ride into Jun 24 rebalance.
