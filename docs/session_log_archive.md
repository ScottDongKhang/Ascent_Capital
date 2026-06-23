# Session Log Archive — Ascent Capital

Sessions prior to 2026-06-22. Maintained as reference; active sessions live in `CLAUDE.md`.

---

### 2026-06-20 (AI PM / "no alpha" investigation — measurement repair + self-heal, honest signal now −6pp)
- Investigated "AI PM poor performance" + "Ascent makes no alpha." TWO separate causes, neither what the dashboard implied.
- **No-alpha-vs-SPY is structural**: pure quant (A★ +12.55%) lags SPY (C +16.61%) by ~4pp, actual book lags the SAME ~4pp. Cause = ~22% defensive non-equity sleeves + 200MA cut + 15% vol-target overlay costing beta in an equity-only bull.
- **AI PM measurement broken two ways**: (1) Track A★/A/D recorded null Jun 8–18 (no self-heal analog to `backfill_track_b`). (2) `earned_authority.json` buffer seeded from shadow log (different A★ series + duplicate `0.00124`) → Sortino engine scoring corrupted data.
- FIX (TDD, 3 new tests): `backfill_astar_d()` recomputes null A★/A/D from as-of snapshots + history (idempotent). `rebuild_buffers_from_counterfactual()` reconciles Sortino buffers to log. Both wired into daily run.
- **Honest result**: D−A★ = −6.06pp/22d (was a corrupted −2.89pp/13d). Modestly value-subtracting in calm_bull; n small.
- Files: `ascent/monitoring/ai_pm_counterfactual.py`, `ascent/strategy/earned_authority.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`, `tests/test_ai_pm_authority.py`.

### 2026-06-19 (Track B trace + fix — the "actual gave up 11pp of quant" gap was a measurement artifact)
- Track B had real data on only 12 of 38 days vs A★ on 21 — only 2 days overlapped. Honest common-window diff was −4.02pp/n=2 (noise). The ~11pp gap was a disjoint-window artifact.
- Root cause: `_log_holdings` computes `day_ret = (equity − last_equity)/last_equity` but Alpaca's 1D bar settles ~17:00 PT — after the 1:45 PM run → fake 0.0 → Track B.
- FIX: `alpaca_broker.get_portfolio_history()` returns settled 1D bars. `backfill_track_b()` replays them over the log (idempotent). 35 rows corrected; B−A★ now −0.42pp/21d (was −4.02pp/n=2).
- Files: `ascent/execution/alpaca_broker.py`, `ascent/monitoring/ai_pm_counterfactual.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`.

### 2026-06-19 (daily run — 4 bugs diagnosed/fixed, today's logs reset, clean rerun)
- BUG 1: `_cs_normalize` got `llm_fundamental` sleeve as cross-sectional Series, not date×symbol DataFrame. Fix: broadcast Series across `features["close"]` dates.
- BUG 2: `name 'pd' is not defined` in `_log_holdings` price-fetch block. Fix: local `import pandas as pd`.
- BUG 3: Dashboard subprocess ModuleNotFoundError — `scripts/` as cwd breaks `ascent` import. Fix: `cwd=repo_root` + `PYTHONPATH=repo_root`.
- BUG 4: `_sortino` summed buffer containing None sentinel → TypeError. Fix: filter None in buffers + `_sortino`.
- Files: `ascent/alpha/stack.py`, `run_all_agents.py`, `ascent/strategy/ai_pm_perf_feedback.py`, tests.

### 2026-06-18 (AI PM "−11.63pp alpha destruction" investigation — measurement artifact, not real)
- The "−11.63pp" headline was a measurement artifact: `score_daily` appended unconditionally (duplicates), Track A★/D froze at 0.0 (returned 0.0 when no prices → should be None), NaN poisoning from yfinance trailing all-NaN row, `get_cumulative_returns` compared disjoint windows.
- Honest signal (Track D vs A★, 12 common days): −2.33pp — pure-AI-PM is modestly defensive in calm_bull. n=12, too few to disable on.
- Also fixed: test leak where `test_plan_a.py` wrote to the real counterfactual log via `monkeypatch.chdir` that didn't sandbox the absolute `_REPO` path.
- Files: `ascent/monitoring/ai_pm_counterfactual.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`, `tests/test_plan_a.py`.

### 2026-06-17 (AI PM calibration learning loop fix)
- `_compute_calibration_returns()` added: reads `prices_live.parquet`, fills `realized_21d` for log entries ≥21d old. Was silently no-opping since May 18.
- Files: `run_all_agents.py`, `tests/test_calibration_tracker.py`, `tests/agents/test_ai_pm_fallback_fix.py`.

### 2026-06-11 (MiroFish 10-round fix + AI PM Phase 2 force-seal + clean rebalance rerun)
- MiroFish timeout root-caused: status poll can't detect server-side prepare failure; Zep classification stochastic on thin event text; reddit runner idles in wait-for-commands → looks like timeout.
- Fixes: fast-fail via sim-state polling, graph rebuild on 0 entities, `Accept-Language: en` (prevents Chinese reports breaking sentiment parser), rounds-done via `/env-status env_alive` + POST `/stop`.
- AI PM Phase 2 force-seal: direct Anthropic call with `tool_choice={"type":"tool","name":"propose_portfolio"}` if tool loop exhausts. Worked live: first rejection → retry → sealed.
- Clean rebalance rerun (Jun 10): MiroFish alignment_score=0.82, PROCEED 0.62, ai_weight=5%, 31 orders submitted.
- MiroFish env: LiteLLM proxy at port 4000 (→ Haiku) must be running. OpenRouter 402 = `max_tokens` unset → provider defaults to model max (64k) → afford-check fails.
- Files: `ascent/integrations/mirofish_client.py`, `ascent/integrations/get_mirofish_sentiment.py`, `agents/ai_pm_agent.py`, `scripts/generate_performance_page.py`.

### 2026-06-11 (next-phase improvements — all 3 workstreams + repairs)
- **C3 profiles**: `backfill_missing_profiles()` + `check_book_sector_coverage()` in `ascent/data/ingest/supplementary.py`. Live book 100% sector-labeled. Fixed `_get_portfolio_symbols()` — always returned [] (read payload keys, not nested `weights`).
- **A exposure parity**: `ascent/portfolio/exposure.py` — single source of truth for VIX-confirmed 200MA cut + vol targeting (15% target, floor 0.25, cap 1.0). Production + WF both delegate to it.
- **C1+C2 risk construction**: `_apply_inverse_vol_tilt()` (half-strength, clip [0.5,2]); `enforce_cluster_cap()` (corr>0.70, 20% cap, pro-rata redistribution).
- **B falsifier enforcement**: `ascent/strategy/falsifier_registry.py` — registry from prethesis `what_would_change_my_mind` + judge predictions + pre-mortems. Daily Gate 4 runs `check_all()` + `_apply_falsifier_trim()` (25% ONE trim, floor 4%).
- Fixed: `run_eod_with_weights()` silently no-opped on non-rebalance days — discovery paths now pass `force=True`.
- Files: `ascent/portfolio/exposure.py` (new), `ascent/strategy/falsifier_registry.py` (new), `ascent/data/ingest/supplementary.py`, `ascent/main.py`, `ascent/portfolio/optimizer.py`, `ascent/research/wf_framework/ascent_strategy.py`, `ascent/execution/eod_runner.py`, `run_all_agents.py`.

### 2026-06-10 (AI PM alpha audit fixes — all 11 findings implemented)
- `blend()` rewritten as active-weight budget (5pp one-way TE cap, not 5% mixing). `DUST_THRESHOLD=0.005`.
- `score_daily()` returns None (not 0.0) for missing Track D/A★; `update_authority()` skips None.
- Judge can now `conviction_press` (increase); parse failure defaults to `proceed+degraded` not `reduce_size`.
- `_prethesis_universe` defined from holdings + top alpha scores; `directional_stance` required with falsifier.
- Files: `agents/ai_pm_agent.py`, `ascent/monitoring/ai_pm_counterfactual.py`, `ascent/strategy/ai_pm_learning.py`, `ascent/strategy/earned_authority.py`, `debate/adversarial_authority.py`, `debate/judge.py`, `run_all_agents.py`.

### 2026-06-10 (next-phase improvement spec — alpha/Sharpe/AI-native/drawdown)
- KEY FINDING: `wf_framework/ascent_strategy.py` applies vol-target (15%) which production `ascent/main.py` lacked; production's VIX gate absent from research. Live book and validated strategy were different.
- Spec: `docs/superpowers/specs/2026-06-10-alpha-sharpe-ainative-spec.md`.
- Implementation order: C3 → A → C1+C2 → B (all done Jun 11).

### 2026-06-10 (rebalance run — Phase 1 force-seal + MiroFish diagnosis)
- Phase 1 force-seal: direct Anthropic API call with `tool_choice={"type":"tool","name":"propose_prethesis"}`. Phase 1 now seals: 12 conviction names confirmed.
- MiroFish 402: OpenRouter drained (requested 64000 tokens, `max_tokens` unset). Fix: top up credits.
- Rebalance: PROCEED (0.6 confidence), 27 orders submitted.
- Files: `agents/ai_pm_agent.py`, `ascent/integrations/mirofish_client.py`, `ascent/integrations/get_mirofish_sentiment.py`.

### 2026-06-08 (OpenBB integration — hub reliability + CBOE/CFTC/FF alpha data + AI PM live tools)
- `ascent/integrations/openbb_client.py` (new): central adapter (tiingo→yfinance fallback). All OpenBB calls go here.
- `ascent/data/ingest/cboe_options.py`, `cftc_positioning.py`, `famafrench_factors.py` (all new).
- AI PM Phase 2 gains `get_live_options_flow` + `get_cot_positioning` tools.
- Optional env vars: `TIINGO_TOKEN`, `CFTC_APP_TOKEN`.
- Files: `ascent/integrations/openbb_client.py`, `ascent/data/ingest/cboe_options.py`, `cftc_positioning.py`, `famafrench_factors.py`, `ascent/features/feature_defs.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-08 (AutoHedge integration — Exa news, yfinance fundamentals, ticker discovery)
- `ascent/integrations/exa_news.py` (new): Exa API headlines per symbol daily.
- `ascent/strategy/ticker_discovery.py` (new): `run_discovery()` uses HAIKU_MODEL, conviction threshold 0.75.
- `run_all_agents.py`: 5-day mini-rebalance cooldown, `_trigger_mini_rebalance()`, daily Exa fetch.
- `ascent/main.py` + `us_equities_agent.py`: `extra_symbols: list[str] | None` passthrough for in-memory ticker injection.
- Required: `EXA_API_KEY` env var. Gotcha: `loguru` not installed — use `import logging`.
- Files: `ascent/integrations/exa_news.py`, `ascent/strategy/ticker_discovery.py`, `agents/ai_pm_agent.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `run_all_agents.py`.

### 2026-06-08 (TradingAgents integration — per-ticker memory + StockTwits)
- `memory/ticker_memory.py` (new): per-ticker AI PM decision log, outcome scoring at 10d/21d.
- `ascent/integrations/stocktwits.py` (new): crowd sentiment, band classification, IC logging.
- Files: `memory/ticker_memory.py`, `ascent/integrations/stocktwits.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-07 (MiroFish sentiment validation layer + judge → Opus)
- `ascent/integrations/mirofish_client.py` (new): MiroFish REST client, 8-min deadline, graceful None on failure.
- `ascent/integrations/get_mirofish_sentiment.py` (new): alignment score, decision rules (amplify >0.70, trim <0.40).
- Judge upgraded to `DEFAULT_MODEL` (Opus).
- `data_cache/mirofish_analogues.json` (new): 25 landmark market events.
- Files: above + `ascent/integrations/analogue_matcher.py`, `ascent/integrations/mirofish_calibration.py`, `debate/agents.py`, `debate/judge.py`, `run_all_agents.py`.

### 2026-06-07 (debate persona upgrade — Druckenmiller / Burry / Taleb)
- Bull → Druckenmiller: quantify upside/downside asymmetry explicitly (Monte Carlo p95 vs p5).
- Bear → Burry: lead with specific adversarial score or VaR, not generic warnings.
- Devil's Advocate → Taleb: "is this portfolio convex or concave?" `_section_tail_asymmetry()` computes tail ratio (p95−p50)/(p50−p5); entropy check for turkey-problem warning. Injected into devil's advocate context only.
- Files: `debate/agents.py`.

### 2026-06-05 (AI PM learning system + hallucination prevention)
- `ascent/strategy/ai_pm_learning.py` (new): daily Sonnet brief, post-mortem (~21d lag), pattern memory → `data_cache/ai_pm_pattern_memory.json`.
- Hallucination prevention in code: `_build_data_grounding()`, `_apply_recency_gate_python()`, feedback citation gate, conviction inflation cap.
- Files: `ascent/strategy/ai_pm_learning.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-04 (AI PM Progressive Authority System + quant two-way integration)
- `earned_authority.py` rewrite: 5-level ladder (Shadow→Analyst→Associate→Manager→Director), Sortino-based promotion/demotion, 5-day cooldown, 63-day stuck alert.
- `ai_pm_guardrails.py`, `ai_pm_counterfactual.py`, `ai_pm_perf_feedback.py` (all new).
- Phase 1 writes `data_cache/ai_prethesis_latest.json`. AI PM bootstrapped at Level 1 (5% authority) from this date.
- Files: `earned_authority.py`, `ai_pm_guardrails.py`, `ai_pm_counterfactual.py`, `ai_pm_perf_feedback.py`, `ascent/alpha/stack.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-03 (WF OOS framework + signal fixes)
- WF framework at `ascent/research/wf_framework/`. Fundamental sleeve disabled. KMLM overweight fixed. IC gate −0.010 → −0.005.
- Files: `orchestrator/central_intelligence.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py`, `run_all_agents.py`, `ascent/monitoring/attribution.py`.

### 2026-06-02 (regime hardening)
- Hard crisis override (VIX>30 + SPY 5d<−7%), asymmetric hysteresis (down 0.40 / up 0.70), entropy penalty (entropy<1e-6 → ×0.90).
- Files: `ascent/regime/engine.py`, `ascent/regime/decision.py`, `ascent/regime/types.py`.

### 2026-06-01 (causal intelligence + investor letter)
- `ascent/causal/` module: PC algorithm DAG, per-symbol graph builder, early-exit tracker. Devil's advocate receives causal mechanisms.
- `ascent/reporting/investor_letter.py` (new): Sonnet monthly letter on first trading day of month.
- Files: `ascent/causal/` (new module), `agents/ai_pm_agent.py`, `run_all_agents.py`, `debate/agents.py`, `ascent/reporting/investor_letter.py`.

### 2026-05-31 (anti-hallucination hardening)
- `generate_structured` gains `json_schema` param (wire-level Anthropic enforcement). `_EVIDENCE_RULE` in `debate/agents.py`.

### 2026-05-25 — 2026-05-28 (adversarial intelligence + two-phase AI PM + GitHub Pages)
- Debate redesigned as genuine risk committee: `adversarial_engine.py`, `adversarial_authority.py`, `adversarial_monitor.py`. ONE falsifiable change per rebalance.
- AI PM two-phase: Sonnet pre-thesis (before quant) + Opus synthesis. `momentum_exhaustion` override type.
- GitHub Pages dashboard (`scripts/generate_performance_page.py` + `docs/index.html`). Auto-pushed after every daily run.
