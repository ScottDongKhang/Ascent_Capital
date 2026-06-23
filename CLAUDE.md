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
DEFAULT_MODEL = "claude-opus-4-6"          # AI PM synthesis (Phase 2)
SONNET_MODEL  = "claude-sonnet-4-6"        # debate agents, red team, pre-thesis (Phase 1)
HAIKU_MODEL   = "claude-haiku-4-5-20251001" # classifiers, weight adjustment
```

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
5. Debate is advisory only — never writes to alpha, portfolio, or execution modules.
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

---

## Data / caching

Cache name provenance — never hide: `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`, `macro_live`/`macro_simulated`, `profiles` (sector metadata).

Point-in-time joins: always use `as_of_join()` / `as_of_merge()` for cross-dataset alignment.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Current state (as of 2026-06-20)

- **AI PM**: Level 1 (Analyst, 5% authority budget). Day 1: June 4. Next rebalance June 24. Authority buffer rebuilt clean from the counterfactual log (22 honest paired obs; old shadow-log seed + duplicate `0.00124` removed). **Honest standalone AI PM signal: D−A★ = −6.06pp over 22 common days** (was a corrupted −2.89pp/13d — the truncated window hid the Jun 10–18 run-up where pure-AI-PM lagged most). Modestly value-subtracting in calm_bull; ~rounding-error at 5% authority (B−A★ defensive multi-agent drag, not AI-PM, is the SPY gap). n still smallish — worth watching at Jun 24, not yet a disable signal.
- **No alpha vs SPY is structural, not failure**: pure quant (A★ +12.55%) lags SPY (C +16.61%) by ~4pp; the actual book lags the same ~4pp. Cause = ~22% defensive non-equity sleeves + 200MA cut + 15% vol-target overlay costing beta in an equity-only bull. WF OOS confirms positive *risk-adjusted* alpha; raw-return lag vs a 100%-equity index is by design.
- **Counterfactual measurement now self-heals**: `backfill_astar_d()` (analog of `backfill_track_b`) recomputes null Track A★/A/D rows from as-of snapshots + 45d history; `rebuild_buffers_from_counterfactual()` reconciles the authority Sortino buffer to the log each run. Both wired into the daily path.
- **Regime**: calm_bull. **Tests**: 70 AI-PM-related passing (incl. 3 new); WF framework 6 pre-existing failures unchanged.
- **WF OOS baseline**: Sharpe 0.483, CAGR +12.61%, true alpha +2.54% vs SPY.
- **GitHub Pages**: `https://scottdongkhang.github.io/Ascent_Capital` — auto-updated after every daily run.

---

## Session log

> Full history archived to `docs/session_log_archive.md`.

### 2026-06-22 (daily run + discovery churn fix — two guards on the off-calendar mini-rebalance)
- Ran `run_all_agents.py` (non-rebalance). Clean exit; **but** the off-calendar **discovery** path fired: INDY (India ETF, conviction 0.82) tripped a mini-rebalance → **32-order full-book rotation 2 trading days before the Jun 24 scheduled rebalance**. INDY did not even survive into the final book — the re-optimization dropped it. Counterfactual now D−A★ = −6.52pp/23d, B−A★ = −5.27pp/31d.
- ROOT CAUSE (two design flaws): (1) `_trigger_mini_rebalance` re-ran the FULL us_equities agent + orchestrator → recomputed every weight (full rotation) for one candidate, not an "add one name"; could even discard the triggering ticker. (2) discovery had no rebalance-calendar awareness → churned the book right before a scheduled rebalance that recomputes it anyway (pure cost).
- FIX (TDD, 8 new tests, user chose "both guards"): (A) `_is_near_scheduled_rebalance(today, window=3, cal_path)` — reads `rebalance_calendar.csv`, suppresses discovery within 3 trading days of the next scheduled rebalance (fail-open if calendar missing). Wired at the discovery call site. (B) `_insert_candidate_weights(current, symbol, max_weight)` — ADD-ONLY: candidate gets an equal-weight slot (1/(n+1)), existing book trimmed pro-rata (relative order preserved), max-weight cap enforced via the optimizer's `_water_fill_cap`. `_trigger_mini_rebalance` rewritten to use it instead of the full re-run; regime label for debate read from the cycle's existing us_equities `AgentOutput` (no agent re-run).
- Verified against real data: Jun 22 → guard suppresses (2 days to Jun 24); inserting INDY into the live 19-name book → INDY @ 5.0%, sum 1.0, max 9.5%, only **1 of 19** existing names moves >0.5pp (vs the 32-order rotation that actually executed today).
- Files: `run_all_agents.py`, `tests/strategy/test_discovery_guards.py` (new). 81 strategy+portfolio tests pass.
- NOT done: today's 32-order rotation already filled at Alpaca — left to ride into the Jun 24 rebalance (reversing it = more churn). Guards take effect on the next run. Not committed (user hasn't asked).

### 2026-06-20 (AI PM / "no alpha" investigation — measurement repair + self-heal, honest signal now −6pp)
- Investigated "AI PM poor performance" + "Ascent makes no alpha." TWO separate causes, neither what the dashboard implied.
- **No-alpha-vs-SPY is structural, not failure**: recomputed tracks — pure quant (A★ +12.55%) lags SPY (C +16.61%) by ~4pp, and the actual book lags the SAME ~4pp. AI PM (5% authority) is not the driver. Cause = ~22% defensive non-equity sleeves (PDBC/EWJ/EFA/EWC/LQD/SGOV/KMLM) + 200MA cut + 15% vol-target overlay costing beta in an equity-only bull. WF OOS confirms positive risk-adjusted alpha; raw-return lag is by design.
- **AI PM could not be honestly evaluated — measurement was broken two ways, both masking a STRONGER signal**: (1) Track A★/A/D recorded `null` Jun 8–18 (un-healed pre-Jun-19 freeze/NaN damage; Track B self-heals via `backfill_track_b` but A★/A/D had NO analog → starved eval window). (2) `earned_authority.json` buffer was a Frankenstein — 9 obs seeded Jun 11 from the *shadow log* (different A★ series + duplicated `0.00124`) + 1 from the counterfactual log; `update_authority` only appends on non-null days so the bad seed never reconciled. The Sortino promotion/demotion engine was scoring on corrupted data.
- FIX (TDD, 3 new tests): `ai_pm_counterfactual.backfill_astar_d(closes)` — recomputes null A★/A/D from as-of snapshot weights (`_load_snapshot_asof`) + historical closes (`_return_from_closes`), idempotent, fills None only. `earned_authority.rebuild_buffers_from_counterfactual()` — reconciles both rolling buffers to the log's common-window pairs (single source of truth). Both wired into the daily run: backfill next to `backfill_track_b`; buffer reconcile AFTER `update_authority` so today isn't double-counted.
- One-time repair (backups `logs/counterfactual_daily.jsonl.bak-astar-*`, `data_cache/earned_authority.json.bak-rebuild-*`): healed 25 null cells (Jun 8–18 fully real; residual nulls are April/early-May pre-AI-PM where D legitimately never existed). Buffer rebuilt to 22 clean paired obs, duplicate gone, A★ now matches log exactly.
- **Healing CHANGED the verdict**: honest standalone AI PM signal D−A★ = **−6.06pp over 22 days** (was a corrupted −2.89pp/13d). The truncated window hid the Jun 10–18 run-up where pure-AI-PM gave up upside AND took more downside. Real, modestly value-subtracting in calm_bull; n still smallish — watch at Jun 24, not yet a disable signal.
- Files: `ascent/monitoring/ai_pm_counterfactual.py`, `ascent/strategy/earned_authority.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`, `tests/test_ai_pm_authority.py`. 70 AI-PM-related tests pass.
- Verified: dashboard headline (`scripts/generate_performance_page.py`) reads `get_cumulative_returns()` directly — surfaces the corrected −6.06pp/22d D−A★ and −3.99pp/30d B−A★ (common-window, not the stale cumulative); chart A★/D lines no longer flat-line post-Jun-4 after the heal. No stale hardcoded figure anywhere in the generator. Going forward A★/A/D + buffer self-heal each run.

### 2026-06-19 (Track B trace + fix — the "actual gave up 11pp of quant" gap was a measurement artifact)
- Traced why Track B (actual book) showed +0.58% vs Track A★ (pure quant) +11.35% over the live window. ROOT CAUSE: a disjoint-window artifact (same disease class as Jun 18, only half-fixed). Track B had real data on only 12 of 38 days, A★ on 21 — **only 2 days overlapped**, so the per-track cumulative headline compared different calendars. The honest common-window diff was −4.02pp over n=2 (noise), contaminated further by today's Track B = 0.0.
- Underlying cause of Track B sparsity: `_log_holdings` derives `day_ret = (equity − last_equity)/last_equity`, but Alpaca's 1D portfolio-history bar settles only after ~17:00 PT — after the 1:45 PM run — so intraday `equity == last_equity` → fake 0.0 → fed to `track_b_return` (and the Jun-18 repair nulled those exact-0.0s to None). The same-day Track B was never trustworthy.
- FIX (TDD, 3 new tests): `alpaca_broker.get_portfolio_history()` returns `{date: settled_day_return}` from Alpaca's published 1D bars (the authoritative source, reuses the dashboard's exact computation). `ai_pm_counterfactual.backfill_track_b(history)` replays settled bars over `counterfactual_daily.jsonl`, overwriting every settled day's track_b_return (idempotent; today's unsettled row left to self-heal next run). Wired into the daily scoring path between `score_daily` and `print_cumulative_report`.
- One-time heal against live Alpaca (61 settled bars Mar 24→Jun 18): **35 Track B rows corrected.** Track B: 12→**38 valid days**; B-vs-A★ common window: 2→**21 days**. CORRECTED honest number: **AI value add B−A★ = −0.42pp over 21 days** (was −4.02pp/n=2) — the live book essentially MATCHES pure-quant; it did NOT give up ~11pp. Track D (pure-AI-PM) unchanged at −2.89pp/13d (priced from yfinance, not Alpaca) — still the one real, small-sample caution.
- Live-vs-SPY lag (+8.71% Apr 1→Jun 19 vs SPY +16.56%) is real but structural: ~25% of the book is defensive (gold IAU+GLD, bonds TLT+HYG, cash SGOV+BIL, long-vol VIXY) + the 200MA/vol-target overlay, in an equity-only bull. Diversification costing beta, not alpha destruction.
- Files: `ascent/execution/alpaca_broker.py`, `ascent/monitoring/ai_pm_counterfactual.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`. Backup: `logs/counterfactual_daily.jsonl.bak-trackb-*`. 40 tests pass.
- Open: dashboard headline (`scripts/generate_performance_page.py`) should be verified to read the corrected `n_common`-annotated B−A★ (−0.42pp), not a stale cumulative. Going forward Track B self-heals each run; B vs A★ is now trustworthy for the Jun 24 rebalance authority decision.

### 2026-06-19 (daily run — 4 bugs diagnosed/fixed, today's logs reset, clean rerun)
- Daily `run_all_agents.py` (non-rebalance) had completed exit-0 but degraded: us_equities produced 0 positions, counterfactual + dashboard + authority all errored. Diagnosed all four via real-pipeline repro (`/tmp/repro_useq.py`), not guesswork.
- BUG 1 (critical): `[USEquities] Pipeline failed: No axis named 1 for object type Series` → `ascent/alpha/stack.py:_cs_normalize` got the `llm_fundamental` sleeve as a cross-sectional Series (per-symbol), not a date×symbol DataFrame. Triggered by the uncommitted `llm_fundamental.py` raw→ratio auto-conversion that finally "activated" the 3%-weight sleeve (it had always returned empty before → silently skipped). Fix: broadcast the Series across `features["close"]` dates before adding to `alphas` (mirrors `fundamental_alpha`). us_equities now returns 10 positions.
- BUG 2: counterfactual `name 'pd' is not defined` in `_log_holdings` daily price-fetch block (used `pd.MultiIndex`; `pd` only imported in a different function). Fix: local `import pandas as pd`. Track A★/D now price real returns again (was recording None → frozen).
- BUG 3: dashboard `ModuleNotFoundError: No module named 'ascent'` — generator subprocess ran `python scripts/...py` so `scripts/` (not repo root) was on `sys.path`; `ascent` is not pip-installed. Worked under launchd (inherited PYTHONPATH) but not a manual run. Fix: pass `cwd=repo_root` + `PYTHONPATH=repo_root` to the subprocess.
- BUG 4 (masked by #2): `Feedback/authority update skipped: float + NoneType` → `ai_pm_perf_feedback._sortino` summed a buffer containing the June-18 missing-data `None` sentinel; `compute_feedback` built `d_rets`/`as_rets` with `.get(key, 0.0)` which doesn't replace present-but-None. Fix: filter None when building the buffers + defensive None-filter in `_sortino`.
- Reset today's logs before rerun: backed up `logs/` → `logs.bak-2026-06-19-rerun`, stripped all `2026-06-19` rows from 17 append-only JSONLs. Final clean rerun: us_equities 10 / merged 19 positions (wt 0.9946), regime calm_bull, counterfactual real (Track D +4.59%, 13 common days), dashboard pushed, no errors, single non-duplicated today-row per log.
- Tests: +3 regression (`tests/alpha/test_stack_weights.py` Series-sleeve broadcast; `tests/test_ai_pm_perf_feedback.py` _sortino None-skip + compute_feedback None-rows). 104 passed across alpha/strategy/feedback/authority/counterfactual.
- Files: `ascent/alpha/stack.py`, `run_all_agents.py`, `ascent/strategy/ai_pm_perf_feedback.py`, `tests/alpha/test_stack_weights.py`, `tests/test_ai_pm_perf_feedback.py`.
- Open: the 3 pre-existing uncommitted edits (`llm_fundamental.py` raw→ratio, `main.py` `_fund_raw` wiring, `factor_model.py` tz-strip) remain uncommitted — they activated the llm_fundamental sleeve and are now functional with the stack fix; commit when ready. `logs.bak-2026-06-19-rerun` left in place as a safety net — delete when satisfied.

### 2026-06-18 (AI PM "−11.63pp alpha destruction" investigation — measurement artifact, not real)
- ROOT CAUSE: the "AI PM cost −11.63pp vs quant" headline is a **measurement artifact in the counterfactual layer**, not real alpha destruction. The only honest signal (Track D vs A★ over the 12 days both have genuine data, May 19–Jun 4) is **−2.33pp** — pure-AI-PM is modestly *defensive* in calm_bull, giving up upside, n=12 (too few to disable on).
- Five bugs found & fixed in `ascent/monitoring/ai_pm_counterfactual.py` + the daily scoring block in `run_all_agents.py`:
  1. **`score_daily` appended unconditionally** → the Jun-10 overnight rerun wrote ~9 duplicate same-date rows; `get_cumulative_returns` multiplied every one into the product. Fixed: `_upsert_daily()` (last-wins per date).
  2. **Track A★/D froze at 0.0 after Jun 4** while Track B kept accruing → cross-track diff compared disjoint windows. Mechanism: `_portfolio_return` returned 0.0 when no symbol could be priced. Fixed: returns `None` when 0 symbols priced (None = skipped, not frozen).
  3. **NaN poisoning** — yfinance `period="5d"` returns a trailing all-NaN row (today's unpublished bar); one NaN `curr` NaN'd the whole track. Fixed: `_portfolio_return` skips NaN; daily fetch uses `period="7d"` + `dropna()` last-2 closes. Verified A★/D now produce real returns going forward.
  4. **`get_cumulative_returns` compared tracks cumulated over disjoint windows.** Fixed: each track cumulates over its own non-None days; B−A★ and D−A★ use the common window only, and the report now prints the common-day count (B−A★ = 1 common day → flagged noise; D−A★ = 12 days → real).
  5. **Test leak** — `tests/test_plan_a.py::test_holdings_log_has_benchmark` called `_log_holdings` whose counterfactual `DAILY_LOG` is an absolute `_REPO` path that `monkeypatch.chdir` does NOT sandbox → every test run wrote a placeholder `2026-04-16` row (b=0.01/c=0.005) to the **real** log (and pre-fix, 0.0/0.0 pairs to the real `earned_authority.json` via `update_authority`). Fixed: patch `DAILY_LOG` to tmp in the test.
- `blend()` math confirmed NOT a drag source: 5% active-weight budget → max single-position delta ~±1.5pp, total added turnover capped at 5pp (~sub-bp cost). Judge verdicts (May 27/Jun 10/Jun 15) all `proceed` + one ~1pp trim — not over-conservative, not the drag.
- Data repairs (backups `.bak-2026-06-18`): `logs/counterfactual_daily.jsonl` deduped 59→37 rows, exact-0.0 across all tracks → None (missing-data sentinel). `data_cache/earned_authority.json` track buffers stripped of 6 fabricated trailing 0.0/0.0 pairs (15→9 honest obs, matches shadow log).
- Tests: +5 in `tests/test_ai_pm_counterfactual.py` (idempotent upsert, None-on-no-prices, NaN-skip, common-window diff), test_plan_a isolation fixed. 16/16 local pass. Full suite 977 passed / 13 failed — all 13 pre-existing (wf_framework + test_new_ingest/openbb drift), none touch changed modules.
- Files: `ascent/monitoring/ai_pm_counterfactual.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`, `tests/test_plan_a.py`, `logs/counterfactual_daily.jsonl`, `data_cache/earned_authority.json`.
- Open: B vs A★ stays uncomparable until both record real values over a common window (≈Jun 19 onward, now that the freeze is fixed). The dashboard headline (`scripts/generate_performance_page.py`) likely still reads the old uncorrected cumulative — verify it picks up the new `n_common_*`-annotated numbers. AI PM calm_bull defensiveness (−2.33pp/12d) is worth watching at the Jun 24 rebalance but n is too small to disable the layer.

### 2026-06-11 (next-phase improvements implemented — all 3 workstreams + repairs)
- Implemented `docs/superpowers/specs/2026-06-10-alpha-sharpe-ainative-spec.md` in full.
- **C3 profiles**: `backfill_missing_profiles()` + `check_book_sector_coverage()` in `ascent/data/ingest/supplementary.py`; coverage guard wired into daily runner. Live book now 100% sector-labeled (was 78.8%); 59 more universe symbols backfilled. Also fixed `_get_portfolio_symbols()` — it always returned [] (read payload keys instead of nested `weights`), so alt-data collection had been getting an empty portfolio list.
- **Authority repair (closes prior session's open item)**: `earned_authority.json` track buffers rebuilt from shadow log — 9 honest paired observations, all fabricated 0.0 entries removed (backup: `.bak-2026-06-11`).
- **A exposure parity**: new `ascent/portfolio/exposure.py` = single source of truth for VIX-confirmed 200MA cut + vol targeting (15% target, floor 0.25, cap 1.0, frozen at rebalance rows). `ascent/main.py` and `wf_framework/ascent_strategy.py` both delegate to it — production now vol-targets, research now VIX-gates. Config knobs in `BacktestConfig`. 15 tests incl. parity + no-lookahead.
- **C1+C2 risk construction**: `_apply_inverse_vol_tilt()` (half-strength, clip [0.5,2]) via `sector_constrained_weighted(vol_panel=...)`; `enforce_cluster_cap()` (corr>0.70 connected components capped at 20%, pro-rata redistribution, max_weight re-enforced). Wired in main.py (live row) + WF (per rebalance row), flags in BacktestConfig. 11 tests. NOTE: pre-existing `_water_fill_cap` precision ~cap+1e-4 (clamp-renorm loop), not introduced here.
- **B falsifier enforcement**: new `ascent/strategy/falsifier_registry.py` — registry built each rebalance from prethesis `what_would_change_my_mind` (one Haiku structuring call) + AI PM pre_mortem; judge predictions registered as relative-vs-SPY conditions at intervention time; causal early exits folded in. Daily Gate 4 now calls `check_all()` (price/macro evaluated in code, news via one Haiku call) and `_apply_falsifier_trim()` executes ONE 25% trim (floor 4%, proceeds to cash, single sell) gated by shared mini-rebalance cooldown + new `falsifier_trim` authority type (scored at 10d like judge interventions). Old shadow-log-only Gate 4 removed. 14 tests.
- **Latent bug fixed**: `run_eod_with_weights()` silently no-opped on non-rebalance days — discovery mini-rebalances never actually executed orders. Added `force=True` calendar bypass (kill switch + approval still apply); discovery path now passes it.
- WF framework's 6 failures confirmed pre-existing via stash test (same failures without these changes).
- Files: ascent/portfolio/exposure.py (new), ascent/strategy/falsifier_registry.py (new), ascent/data/ingest/supplementary.py, ascent/main.py, ascent/config/settings.py, ascent/portfolio/optimizer.py, ascent/research/wf_framework/ascent_strategy.py, ascent/execution/eod_runner.py, debate/adversarial_authority.py, run_all_agents.py, tests/portfolio/test_exposure.py (new), tests/portfolio/test_risk_construction.py (new), tests/strategy/test_falsifier_registry.py (new), data_cache/earned_authority.json.
- Open: re-run WF baseline to re-baseline Sharpe/CAGR with the parity overlays — "Current state" numbers are stale until then. First live falsifier registry builds at the June 24 rebalance.

### 2026-06-11 (MiroFish 10-round fix + AI PM Phase 2 force-seal + clean rebalance rerun)
- MiroFish "preparation timeout" root-caused as three stacked bugs, none round-count-related: (1) client polled `/prepare/status` with simulation_id only, which can never report failure — a server-side prepare failure looked like a timeout and burned the whole deadline; (2) Zep entity classification is stochastic on thin event text (22:30 run got 0 classified entities of 5 nodes → prepare failed instantly by design); (3) the reddit runner idles in wait-for-commands mode after its rounds, so `run-status` stays "running" forever — even successful sims looked like timeouts (today's earlier 3-round runs included).
- `mirofish_client.py` fixes: `_prepare_simulation` returns ready/no_entities/failed/timeout (fast-fail via `expected_entities_count` and sim-state polling); `run_sync` rebuilds the graph once on 0 entities; `_create_project` event text enriched with named companies + participant archetypes (9 classified entities vs 0-3 before); `_wait_for_simulation` detects rounds-done via `/env-status` `env_alive` then POSTs `/stop`; all requests send `Accept-Language: en` (report language follows it — Chinese reports broke the English keyword sentiment parser, "mixed 0.50" vs real "bullish 0.82/alignment 0.89").
- `get_mirofish_sentiment.py`: `_N_ROUNDS` 3→10, `_TIMEOUT_SECS` 900→1500 (measured full 10-round run: ~403s), prints `[MiroFish] alignment_score=...` on success.
- `agents/ai_pm_agent.py`: Phase 2 force-seal pass mirroring Phase 1 — if the main tool loop exhausts without `propose_portfolio`, direct Anthropic call with `tool_choice={"type":"tool","name":"propose_portfolio"}` on `_phase2_model`, validation gates kept intact with one retry feeding the rejection back. New `_PROPOSE_PORTFOLIO_TOOL` constant. Worked live: first rejection (missing prethesis_disposition) → retry → sealed → blend applied.
- Clean rebalance rerun (`--date=2026-06-10`, run overnight): Phase 1 + Phase 2 both force-sealed, MiroFish alignment_score=0.82 bullish, PROCEED 0.62, AI PM blend applied (ai_weight=5%), 31 orders submitted to Alpaca (queue for June 11 open). June 11 daily non-rebalance run executed after.
- Tests: +3 MiroFish client tests, +1 wait-for-commands test, +4 Phase 2 force-seal tests (tests/agents/test_ai_pm_phase2_force_seal.py new). Stale test updated for session-scoped disposition gate. 885 passed; 2 pre-existing failures from earlier uncommitted drift (`test_new_ingest` patches removed `_get_obb` in famafrench_factors; `test_openbb_client` COT snapshot) — not from this session.
- MiroFish env note: `/Users/scott/MiroFish/.env` now points LLM to LiteLLM proxy (port 4000 → Haiku). Proxy must be running alongside the MiroFish server on rebalance days. Earlier OpenRouter 402s were because profile-gen leaves max_tokens unset → provider defaults to model max (64000) → OpenRouter afford-check rejected.
- Dashboard fixes (`scripts/generate_performance_page.py`): (1) NAV chart always ended at yesterday — Alpaca publishes the 1D portfolio-history bar only after ~00:00 UTC (17:00 PT), later than any daily run; now appends today's live `/v2/account` equity when the series hasn't rolled. (2) Execution days derived from `logs/eod_log.jsonl` (`rebalanced=true`) instead of a hardcoded date set — June 10 shows "Rebalance #5 — Executed · Proceed" and future rebalances self-label. (3) Stale "Phase 0 — Shadow Mode" chip / "0% authority" note replaced with live earned-authority state (Level 1 — Analyst · 5%).
- AI PM first real contribution measured (from `logs/ai_pm_decision_log.jsonl`): exactly 5.0pp active weight moved — adds EWT +1.3pp/VNQ +0.9/KMLM +0.7/IFRA +0.6/HYG +0.5/DBB +0.4 vs cuts STRL −1.0/VNO −1.0/intl sleeve −0.4 each. AI PM and judge independently faded VNO.
- Dashboard redesigned as a product page (`bc1c278`): hero with system pitch + UTC freshness chip + dominant NAV, 5-node architecture strip, narrative section order (performance → debate → AI PM authority → holdings), equity curve with gradient/glow/progressive draw + icon-only verdict markers + regime boundary labels, counterfactual chart date-dedupe + shadow-period annotation. Same generator injection slots — daily regeneration unaffected. Fixed along the way: authority-ladder CSS selectors never matched (`ph-active` vs `.active`), hero negative margin = 48px mobile overflow, `generated_at` was local time labeled UTC. Verified with headless Chrome (zero JS errors, desktop + mobile).
- Files: `ascent/integrations/mirofish_client.py`, `ascent/integrations/get_mirofish_sentiment.py`, `agents/ai_pm_agent.py`, `scripts/generate_performance_page.py`, `tests/test_mirofish_integration.py`, `tests/agents/test_ai_pm_phase2_force_seal.py` (new), `tests/agents/test_ai_pm_prethesis_fixes.py`, CLAUDE.md.
- Open: fix the 2 stale tests for famafrench/openbb drift; earned_authority.json buffer repair still pending before June 24.

### 2026-06-10 (next-phase improvement spec — alpha/Sharpe/AI-native/drawdown)
- New spec: `docs/superpowers/specs/2026-06-10-alpha-sharpe-ainative-spec.md`. Three workstreams, diagnosis verified in code, no code changed this session.
- KEY NEW FINDING — research/production divergence: WF baseline (`wf_framework/ascent_strategy.py`) applies `_apply_vol_target` (15% target, floor 0.25) which production `ascent/main.py` lacks; production's VIX-gated 200MA cut is absent from research (research cuts on MA alone). Live book and validated strategy are different strategies in both directions.
- Workstream A: port vol targeting to production, backport VIX gate to research, parity test, re-run WF baseline.
- Workstream B: falsifier enforcement — `check_early_exits` (Gate 4, run_all_agents.py:1086) flags causal breaks but only writes a shadow-log line; prethesis `what_would_change_my_mind` / judge predictions / pre-mortems never checked. Build unified falsifier registry + daily code/Haiku evaluation + bounded 25% trims via existing mini-rebalance path, scored like judge interventions.
- Workstream C: inverse-vol half-tilt in `sector_constrained_weighted()`, correlation-cluster cap (20%, corr>0.70), profiles backfill — verified VVV/WDC/YETI = 21.2% of live book missing sector labels.
- Implementation order: C3 → A → C1+C2 → B (B last; shares run_all_agents.py with audit-fix PR).
- Files: docs/superpowers/specs/2026-06-10-alpha-sharpe-ainative-spec.md (new), CLAUDE.md.

### 2026-06-10 (AI PM alpha audit fixes — all 11 findings implemented)
- Implemented all fixes from Fable's alpha audit spec (`docs/superpowers/specs/2026-06-10-ai-pm-alpha-audit.md`).
- Finding 1+10: removed fallback-append from `_tool_propose_portfolio` rejection path; `run_ai_pm` revision now falls back to initial result if revision is empty/fallback; decision log writes on every rebalance including fallback.
- Finding 3: `score_daily()` returns `None` (not 0.0) for missing Track D/A★; `update_authority()` skips buffer append on None — authority ladder no longer fed fabricated zeros.
- Finding 2: `blend()` rewritten as active-weight budget — `ai_weight=0.05` means 5pp one-way tracking-error cap, not 5% mixing coefficient. `DUST_THRESHOLD=0.005` replaces 2% floor inside blend.
- Finding 4+9: judge can now `conviction_press` (increase) as well as reduce; parse failure defaults to `proceed+degraded` not `reduce_size`.
- Finding 5+11: post-mortem fires on any decision ≥21d old (override filter dropped); pattern memory reads from `ai_pm_pattern_memory.json` via `get_pattern_summary()` (path mismatch fixed); `AIPMResult.tool_failures` records unavailable tools per rebalance.
- Finding 6+7+8: `_prethesis_universe` defined from holdings + top alpha scores; `_build_data_grounding` returns news even with empty symbols; `AIPreThesis` gains `conviction_reasons`, `sector_thesis`, `directional_stance` fields; `directional_stance` required in schema with falsifier; `_tool_propose_portfolio` enforces `prethesis_disposition: FOLLOWED|OVERRIDDEN`.
- 52 new tests across 6 new test files. 928 passing, 6 pre-existing failures unchanged.
- Files: `agents/ai_pm_agent.py`, `ascent/monitoring/ai_pm_counterfactual.py`, `ascent/strategy/ai_pm_learning.py`, `ascent/strategy/earned_authority.py`, `debate/adversarial_authority.py`, `debate/judge.py`, `run_all_agents.py`, + 6 new test files.
- Open: `data_cache/earned_authority.json` buffer repair (strip leading 0.0/0.0 pairs from Jun 4-9) should be done manually before June 24 rebalance. First real AI PM blended portfolio expected June 24.

### 2026-06-10 (AI PM alpha audit — diagnosis only, no code changed)
- Full audit of why AI PM contributes zero alpha. Spec: `docs/superpowers/specs/2026-06-10-ai-pm-alpha-audit.md`.
- CRITICAL confirmed: (1) feedback gate × red-team revision swallows valid Phase 2 portfolios into fallback — Jun 10 rebalance applied nothing, decision log never written once; (2) `earned_authority.blend()` is a 5% mixing coefficient + 2% floor → ±2pp overrides become ±0.1pp, AI cannot add names below Level 3; (3) Track D fed literal 0.0 since Jun 4 promotion (shadow log proof) — authority ladder scoring fabricated data; (4) judge is reduction-only + reduce_size parse default → net judge delta strictly negative (−1.96pp + Apr 15 book trim); (5) learning loop triple-gated — pattern memory empty forever without code change.
- HIGH: `_prethesis_universe` undefined → Phase 1 gets no grounding/news; `_strip_prethesis_for_phase2` reads nonexistent attrs → sourced claims never reach Phase 2; no directional/falsifiable macro stance required → hedges pass as theses.
- Files touched: docs/superpowers/specs/2026-06-10-ai-pm-alpha-audit.md (new), CLAUDE.md.
- Open: all findings now fixed (see entry above).

### 2026-06-08 (OpenBB integration — hub reliability + CBOE/CFTC/FF alpha data + AI PM live tools)
- `ascent/integrations/openbb_client.py` (new): central adapter — `fetch_symbol` (tiingo→yfinance fallback), `fetch_return`, `get_live_macro`, `get_options_snapshot` (IV skew/PCR/ATM IV), `get_cot_snapshot`. All OpenBB calls go here.
- `ascent/data/hub.py`: `_fetch_symbol` now tries openbb_client first; yfinance fallback preserved.
- `memory/ticker_memory.py`: `_fetch_return` now tries openbb_client first; yfinance fallback preserved.
- `ascent/data/ingest/cboe_options.py` (new): historical CBOE options rows — IV skew, PCR, ATM IV, iv_rank_52w. Writes to `options_flow.parquet`.
- `ascent/data/ingest/cftc_positioning.py` (new): CFTC COT S&P 500 e-mini speculator positioning. Writes to `cftc_positioning.parquet`.
- `ascent/data/ingest/famafrench_factors.py` (new): Fama-French 5-factor + momentum daily returns. Writes to `famafrench_factors.parquet`.
- `ascent/features/feature_defs.py`: `factor_loadings(returns, factor_df, window)` appended — rolling beta per factor for ML sleeve.
- `agents/ai_pm_agent.py`: `get_live_options_flow` + `get_cot_positioning` tools added to Phase 2 (`AI_PM_TOOLS`). Both Phase 2 only — not in `PRE_THESIS_TOOLS`.
- `run_all_agents.py`: CBOE/CFTC/FF ingest wired after hub run using `us_symbols`, each in independent try/except.
- Tests: 11 openbb_client + 13 new_ingest = 24 new tests pass. 777 existing tests unchanged. No regressions.
- New env vars (optional): `TIINGO_TOKEN` (price reliability), `CFTC_APP_TOKEN` (COT rate limiting).
- Nothing left open.

### 2026-06-08 (TradingAgents integration — per-ticker memory + instrument identity + StockTwits)
- `memory/ticker_memory.py` (new): per-ticker AI PM decision log, outcome scoring (incremental alpha at 10d/21d), win/miss/fade/early classification. Storage: `memory/ticker_memory.jsonl`.
- `ascent/integrations/stocktwits.py` (new): StockTwits public API crowd sentiment, band classification, stale guard. Logs IC to `logs/stocktwits_ic.jsonl`.
- `agents/ai_pm_agent.py`: `_build_data_grounding()` now includes sector/industry identity from profiles.parquet. Phase 2 prompt injects per-ticker track record + cross-ticker lessons. Both phases receive StockTwits sentiment block.
- `run_all_agents.py`: `_write_decision_log()` records per-ticker AI PM decisions; daily path scores outcomes; StockTwits fetch gated inside `if is_rebalance:`.
- `tests/memory/test_ticker_memory.py` (new): 10 tests. `tests/integrations/test_stocktwits.py` (new): 8 tests. All 18 passing.
- Fixes: `scored=True` on None fetch (now `continue`), atomic write via `.tmp`+rename, zero-alpha → `None` verdict, walrus operator for ticker context, weak test assertion strengthened.
- Nothing left open.

### 2026-06-07 (MiroFish sentiment validation layer + judge → Opus)
- `ascent/integrations/mirofish_client.py` (new): MiroFish REST client — 8-step API flow, 8-min deadline timer, graceful None on any failure.
- `ascent/integrations/analogue_matcher.py` (new): TF-IDF cosine similarity against 25-event `data_cache/mirofish_analogues.json` library; keyword fallback if sklearn unavailable.
- `ascent/integrations/mirofish_calibration.py` (new): `bootstrap_calibration()` + `get_base_rate()` + `record_entry()`. Idempotent bootstrap from analogues library.
- `ascent/integrations/get_mirofish_sentiment.py` (new): full tool executor — alignment score formula, decision rules (amplify >0.70, trim 25% <0.40 + negative base rate), timeout fallback.
- `agents/ai_pm_agent.py`: `get_mirofish_sentiment` wired into `AI_PM_TOOLS` (Phase 2 only, not PRE_THESIS_TOOLS).
- `debate/agents.py`: devil's advocate attacks AI PM thesis when mirofish alignment < 0.50.
- `debate/judge.py`: judge upgraded from `SONNET_MODEL` → `DEFAULT_MODEL` (Opus) per user direction.
- `run_all_agents.py`: forwards `mirofish_sentiment` from AI PM result into `portfolio_state` for debate.
- `data_cache/mirofish_analogues.json` (new): 25 landmark market events (force-added, gitignored dir).
- `data_cache/mirofish_calibration.json` (new): empty initial state, bootstrapped at runtime.
- 19/19 MiroFish tests + 77/77 debate+AI PM tests passing.
- Files: above new + `ascent/integrations/__init__.py`, `agents/ai_pm_agent.py`, `debate/agents.py`, `debate/judge.py`, `run_all_agents.py`.
- Nothing left open.

### 2026-06-08 (AutoHedge integration — Exa news, yfinance fundamentals, ticker discovery)
- `ascent/integrations/exa_news.py` (new): `fetch_news()` calls Exa API (free tier, 1k/month), returns {sym: [headline...]} for portfolio universe daily. Delays 0.2s between requests.
- `agents/ai_pm_agent.py`: `_fetch_financials()` (new) — yfinance quarterly ratios (current_ratio, D/E, rev_growth_yoy, gross_margin), 24h cache at `data_cache/financials_cache.json`, atomic write. `_build_data_grounding()` extended: appends FUNDAMENTALS + LIVE NEWS blocks; accepts `news_context` param.
- `run_ai_pm_prethesis()` and `run_ai_pm()` both accept `news_context_arg` — wired from `run_all_agents.py`.
- `ascent/main.py` + `agents/us_equities_agent.py`: `extra_symbols: list[str] | None` passthrough — lets mini-rebalance inject a discovered ticker in-memory without mutating config.
- `ascent/strategy/ticker_discovery.py` (new): `run_discovery()` uses HAIKU_MODEL to identify ONE new ticker from Exa news grounded in actual headlines. Returns `DiscoveryResult` or None if conviction < 0.75.
- `run_all_agents.py`: `_check_mini_rebalance_cooldown()` (5 trading day gate), `_write_mini_rebalance_log()`, `_trigger_mini_rebalance()` (runs full agent + debate gate before submitting). Exa fetch runs daily before non-rebalance path. Discovery wired into non-rebalance block.
- Tests: 6 (exa_news) + 7 (financials) + 7 (ticker_discovery) = 20 new tests, all passing.
- Files: `ascent/integrations/exa_news.py`, `ascent/strategy/ticker_discovery.py`, `agents/ai_pm_agent.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `run_all_agents.py`, `tests/integrations/test_exa_news.py`, `tests/agents/test_ai_pm_financials.py`, `tests/strategy/test_ticker_discovery.py`.
- Open: set `EXA_API_KEY` env var before first production run. Key gotcha: `loguru` not installed — use `import logging` pattern, not loguru.

### 2026-06-07 (debate persona upgrade — Druckenmiller / Burry / Taleb)

The debate layer had a structural flaw: all three LLM agents (bull, bear, devil's advocate) shared the same underlying epistemology — risk/return framing, Gaussian assumptions, generic "concentration risk" arguments. They disagreed on conclusions but asked the same questions. That's not a real risk committee; it's one brain arguing with itself.

This session rewired each agent with a distinct analytical lens borrowed from the `ai-hedge-fund` repo (but adapted to Ascent's data — no `financial_datasets` API, no fundamentals):

**Bull → Druckenmiller**: The question changes from "is this a good portfolio?" to "where is the asymmetry?" Druckenmiller's framework demands the bull quantify the upside/downside ratio explicitly — if the Monte Carlo p95 is 3x the p5 loss, say so. Forces momentum-first arguments and penalizes defensive hedging in strong-regime setups. Identifies positions the adversarial engine rated hard-to-short as structural longs.

**Bear → Burry**: The question changes from "what could go wrong?" to "what is the weakest number?" Burry's framework demands the bear lead with a specific adversarial score, VaR figure, or event-risk flag — not a generic drawdown warning. The Burry persona also introduces an anti-manufacturing constraint: if the adversarial engine found no compelling bear case, say so. Generic bear arguments without hard numbers are dismissed.

**Devil's Advocate → Taleb**: The question changes from "what's the biggest risk?" to "is this portfolio convex or concave?" This is a fundamentally different question. A portfolio can look fine on every traditional metric (positive expected return, no individual position > 10%) and still be structurally concave — more downside than upside in expectation. That's the Taleb insight no generic devil's advocate would ever surface.

New `_section_tail_asymmetry()` function computes this from existing Monte Carlo data:
- Tail ratio = (p95 − p50) / (p50 − p5). Below 1.0 = provably concave.
- Regime entropy check: low entropy (<0.5) triggers "turkey problem" warning — the model is confident but that confidence is itself a fragility signal.
- Injected into devil's advocate context only (bull and bear don't need it; it would dilute their respective lenses).

What the debate layer can now catch that it couldn't before: a portfolio that passes the quant sanity check, has no adversarial flags, and looks clean to the regime specialist — but is structurally concave because the strategy's upside is capped (by max_weight, sector limits) while the downside is fat-tailed. That's a construction-level fragility, not a position-level one. Only a Taleb-framed agent looks for it.

- Files: `debate/agents.py`
- Nothing left open from this session.

### 2026-06-05 (AI PM learning system + hallucination prevention)
- `ascent/strategy/ai_pm_learning.py` (new): daily Sonnet brief, post-mortem (~21d lag), pattern memory (Haiku → `data_cache/ai_pm_pattern_memory.json`). Injected into every Phase 1+2 prompt.
- Hallucination prevention (all 4 in code, not prompts): `_build_data_grounding()`, `_apply_recency_gate_python()`, feedback citation gate in `_tool_propose_portfolio()`, conviction inflation cap.
- AI PM bootstrapped to Level 1 (5% authority). Day 1: June 4.
- Files: `ascent/strategy/ai_pm_learning.py` (new), `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-04 (AI PM Progressive Authority System + quant two-way integration)
- `earned_authority.py` rewrite: 5-level ladder (Shadow→Analyst→Associate→Manager→Director), Sortino-based promotion/demotion, 5-day cooldown, 63-day stuck alert.
- `ai_pm_guardrails.py` (new), `ai_pm_counterfactual.py` (new), `ai_pm_perf_feedback.py` (new).
- `build_alpha_stack(return_breakdown=True)` → signal quality in `AgentOutput.metadata`. Pre-thesis alpha floor in `ascent/main.py`. Phase 1 writes `data_cache/ai_prethesis_latest.json`.
- Honest WF OOS baseline: Sharpe 0.483. Multi-asset/long-short all scored worse — current config is the ceiling.
- Files: `earned_authority.py`, `ai_pm_guardrails.py`, `ai_pm_counterfactual.py`, `ai_pm_perf_feedback.py`, `ascent/alpha/stack.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-17 (AI PM calibration learning loop fix)
- `_compute_calibration_returns()` added to `run_all_agents.py`: reads `prices_live.parquet`, computes cumulative returns since entry date for symbols in calibration log entries ≥21 days old with null `realized_21d`. Replaces the empty-dict call that silently no-opped every daily run since May 18.
- May 18 + May 19 entries (17 positions each) now fully filled. `get_calibration_report()` produces IC = +0.25 [Calibrated] for first time.
- Files: `run_all_agents.py`, `tests/test_calibration_tracker.py`, `tests/agents/test_ai_pm_fallback_fix.py`.
- Open: nothing.

### 2026-06-03 (WF OOS framework + signal fixes)
- WF framework at `ascent/research/wf_framework/`. Fundamental sleeve disabled. KMLM overweight fixed. IC gate tightened (-0.010 → -0.005). `_log_holdings` day_return now from Alpaca `last_equity`.
- Files: `orchestrator/central_intelligence.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py`, `run_all_agents.py`, `ascent/monitoring/attribution.py`.

### 2026-06-02 (regime hardening)
- Hard crisis override (VIX>30 + SPY 5d<−7%), asymmetric hysteresis (down 0.40 / up 0.70), entropy penalty (entropy<1e-6 → ×0.90).
- Files: `ascent/regime/engine.py`, `ascent/regime/decision.py`, `ascent/regime/types.py`.

### 2026-06-01 (causal intelligence + investor letter)
- `ascent/causal/` module: PC algorithm DAG (`causal_discovery.py`), per-symbol graph builder (`dag_builder.py`), compatibility gates, early-exit tracker (`tracker.py`). Devil's advocate receives causal mechanisms.
- `ascent/reporting/investor_letter.py` (new): Sonnet monthly letter on first trading day of month.
- Files: `ascent/causal/` (new module), `agents/ai_pm_agent.py`, `run_all_agents.py`, `debate/agents.py`, `ascent/reporting/investor_letter.py`.

### 2026-05-31 (anti-hallucination hardening)
- `generate_structured` gains `json_schema` param (wire-level Anthropic enforcement). Amnesia/grounding instructions + evidence schema in `llm_fundamental` and `narrative_alpha`. `_EVIDENCE_RULE` in `debate/agents.py`.

### 2026-06-10 (rebalance run — Phase 1 force-seal + MiroFish diagnosis)
- AI PM Phase 1 never sealing fixed: added force-seal pass that bypasses `tool_completion` and calls the Anthropic API directly with `tool_choice={"type":"tool","name":"propose_prethesis"}` — hard-forces the model to seal. Phase 1 now seals: 12 conviction names confirmed.
- MiroFish 500 on `ontology/generate` root-caused: OpenRouter account drained (402, "can only afford 3569 tokens, requested 64000"). Killed 3 zombie `run_reddit_simulation.py` processes. Fix: top up OpenRouter credits at `openrouter.ai/settings/credits`.
- MiroFish timeout increased: 480s → 900s in both `mirofish_client.py` and `get_mirofish_sentiment.py`.
- `clean_index` dedup fix (options/short panels, `feature_defs.py`) confirmed working — US equities no longer zero-positions.
- Rebalance completed: PROCEED verdict (0.6 confidence), 27 orders submitted to Alpaca.
- Files: `agents/ai_pm_agent.py`, `ascent/integrations/mirofish_client.py`, `ascent/integrations/get_mirofish_sentiment.py`.
- Open: Add OpenRouter credits to restore MiroFish functionality on next run.

### 2026-05-25 — 2026-05-28 (adversarial intelligence + two-phase AI PM + GitHub Pages)
- Debate redesigned as genuine risk committee: `adversarial_engine.py`, `adversarial_authority.py`, `adversarial_monitor.py`. ONE falsifiable change per rebalance.
- AI PM two-phase: Sonnet pre-thesis (before quant) + Opus synthesis. Crowding gate (`get_crowding_signal`). `momentum_exhaustion` override type replaces valuation abuse.
- GitHub Pages dashboard (`scripts/generate_performance_page.py` + `docs/index.html`). Auto-pushed after every daily run.
