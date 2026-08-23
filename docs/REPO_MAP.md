# REPO_MAP.md

Navigation index for an AI coding agent working in this repo: where things live and what to grep for.
It is a pointer index, not a spec. Behaviour, constraints, and current state live in `CLAUDE.md`; performance figures live in `CURRENT_VERIFIED_NUMBERS.md`.
`scripts/verify_docs.py` is the drift guard that makes the claims in `CLAUDE.md` machine-checkable. Run it after structural changes.

Ignore `.venv/` and `.claude/worktrees/` when searching. The worktrees are stale copies and will double every grep hit.

---

## 1. Entrypoints

| Command | File | Does | Safe? |
|---|---|---|---|
| `.venv/bin/python run_all_agents.py` | `run_all_agents.py` (`main`) | Daily driver. Weekend branch, startup validation, data hub, `us_equities_agent`, orchestrator, EOD execution. Branches on rebalance day. Rewritten 2026-08-23 to the core skeleton after the AI PM / debate / falsifier noise-layer removal — no line-number map below for it yet. | **Submits real Alpaca paper orders** on rebalance days via `run_eod_with_weights` |
| `.venv/bin/python ascent/main.py` | `ascent/main.py` (`run_pipeline` line 305, `main` line 918) | Single-universe research pipeline: data → features → alpha stack → portfolio → backtest → regime. Returns a 10-tuple. Called by each quant agent. | Read-only (writes caches + `dashboard/` json) |
| `.venv/bin/python scripts/run_ascent_wf.py [--smoke]` | `scripts/run_ascent_wf.py` (`main` line 193, `_run_one` line 136) | Walk-forward OOS harness over `ascent/research/wf_framework/`. Flags: `--smoke --simplified --compare --high-sharpe --multi-asset --long-short --compare-all --live-system`. Writes `outputs/wf_results/`. | Read-only, slow |
| `.venv/bin/python ascent/research/walk_forward_runner.py` | `walk_forward_pipeline` (line 44) | Lighter walk-forward used inside the pipeline (point-in-time slicing via `_pit_slice` / `_pit_macro`). | Read-only |
| `.venv/bin/python scripts/generate_performance_page.py [--push]` | `scripts/generate_performance_page.py` (`main` line 1833) | Builds `docs/index.html` (GitHub Pages dashboard). `--push` commits and pushes. Spawned by `run_all_agents.py` line ~2212. | Read-only unless `--push` |
| `.venv/bin/python ascent/dashboard/live_dashboard.py` | `ascent/dashboard/live_dashboard.py` | Streamlit-style live view reading `logs/` + `execution/merged_weights.json`. | Read-only |
| `.venv/bin/python scripts/verify_docs.py [--quiet\|--json]` | `scripts/verify_docs.py` | 24 assertions defending `CLAUDE.md` / `CURRENT_VERIFIED_NUMBERS.md` claims (model constants, alpha weight agreement, kill switches off, 10-tuple, no loguru, noise-layer-removed, ...). Exit 1 on any FAIL. | Read-only. Run this first when unsure if a doc claim is stale |
| `bash scripts/run_eod.sh` | wrapper | launchd EOD wrapper around the daily run (`com.ascentcapital.eod.plist`). | **Submits orders** |
| `.venv/bin/python scripts/heartbeat_check.py` | `scripts/heartbeat_check.py` | Liveness check against `logs/eod_log.jsonl` + `rebalance_calendar.csv` (`com.ascentcapital.heartbeat.plist`). | Read-only |

Order submission funnels through `ascent/execution/eod_runner.py` (`run_eod`, `run_eod_with_weights`) into `ascent/execution/alpaca_broker.py` (`submit_order`). Grep `submit_order` to find every live-money path.

---

## 2. Package responsibilities

| Package | Owns | Key symbols to grep |
|---|---|---|
| `ascent/config` | Typed config + API keys. Never `Config()` directly. | `get_config`, `Config`, `APIKeys`, `BacktestConfig`, `RegimeConfig`, `AgentOutput` |
| `ascent/data/ingest` | One module per vendor/source (yahoo, tiingo, polygon, fred, sec_filings, earnings_transcripts, cboe_options, cftc_positioning, google_trends, insider, short_interest, analyst, famafrench_factors, reddit_sentiment, capitol_trades, edgar_listener, simulated). | `fetch_daily_bars`, `fetch_universe_daily`, `fetch_all_macro`, `update_sec_signals`, `update_transcript_signals`, `update_cot_cache`, `run_supplementary_ingest` |
| `ascent/data/store` | Parquet cache + point-in-time joins + optional TimescaleDB. | `save_parquet`, `load_parquet`, `validate_cache`, `_calendar_day_key`, `as_of_join`, `as_of_merge` |
| `ascent/data` (top) | Parallel fetch hub, universe membership. | `run_hub`, `hub_is_fresh`, `get_universe_on_date`, `build_historical_universe`, `get_removed_symbols` |
| `ascent/data/normalize`, `/validate`, `/streaming` | Schema normalization, alt-data IC validation, Alpaca websocket. | `normalize_prices`, `pivot_prices`, `validate_altdata_source`, `start_stream` |
| `ascent/features` | Feature panel construction and forward-return targets. | `FeatureBuilder`, `momentum_rank`, `rolling_volatility`, `zscore`, `forward_return`, `build_targets` |
| `ascent/alpha` | Active sleeves (meanrev, statarb) + the blending stack and IC gate. | `build_alpha_stack`, `alpha_to_ranks`, `DEFAULT_ALPHA_WEIGHTS`, `IC_GATE_THRESHOLD`, `meanrev_alpha`, `statarb`, `build_ml_alpha`, `_SPARSE_FILL_ZERO`, `SleeveMetaLearner` |
| `ascent/portfolio` | Weight construction, caps, overlays, optimizers. | `enforce_constraints`, `_water_fill_cap`, `enforce_cluster_cap`, `enforce_risk_budget_cap`, `sector_constrained_weighted_mvo`, `SectorDataError`, `apply_exposure_overlays`, `ma_filter_scale`, `vol_target_scale`, `optimize_mvo`, `black_litterman_views`, `tc_aware_weights`, `build_long_short_weights` |
| `ascent/risk` | Factor model, covariance, exposure bounds, correlation guard. | `compute_factor_loadings`, `compute_portfolio_exposures`, `check_factor_bounds`, `check_cross_agent_correlation`, `apply_correlation_adjustments`, `classify_regime` |
| `ascent/regime` | HMM/Markov regime engine, features, posture, structural breaks. | `RegimeEngine` (**takes `config=dict`**), `RegimeModel`, `RegimeSignal`, `RegimeLabel`, `RegimeFeatureBuilder`, `RegimeDecisionEngine`, `BreakDetector`, `compute_posture_from_regime`, `regime_max_weight` (defined but unreachable — see `ascent/regime/integration.py` module docstring), `check_emergency_refit_triggers` |
| `ascent/backtest` | Vector backtest engine + cost model. | `BacktestEngine`, `BacktestResult`, `estimate_trade_cost`, `estimate_rebalance_costs` |
| `ascent/research` | Walk-forward frameworks, CPCV, self-improvement, factor discovery. | `WalkForwardEngine`, `AscentPortfolioStrategy`, `FullOrchestrationStrategy`, `WindowGenerator`, `PerformanceAnalyzer`, `FoldResult`, `ParameterOptimizer`, `CPCVSplitter`, `run_self_improve`, `run_shadow_promotion`, `run_factor_discovery`, `scan_for_leakage`, `SELF_MODIFY_ENABLED` |
| `ascent/execution` | Broker, order sizing, EOD orchestration, kill switch, TWAP, slippage. | `run_eod`, `run_eod_with_weights`, `submit_order`, `get_portfolio_history`, `compute_orders`, `Order`, `KillSwitchTriggered`, `check` (kill_switch), `should_use_twap`, `TWAP_ENABLED`, `EVENT_TRADING_ENABLED`, `compute_slippage`, `compute_is` |
| `ascent/monitoring` | Post-trade tracking, alerts, weekend/weekly cycles, pre-rebalance checklist. | `run_forward_pnl_cycle`, `export_skill_scores`, `check_alerts`, `run_checklist`, `compute_position_health`, `run_attribution`, `compute_signal_health`, `run_exit_alerts`, `run_weekend`, `run_weekly_debrief`, `run_scenario_planning` |
| `ascent/llm` | Single Anthropic wrapper. Import model constants from here. | `DEFAULT_MODEL`, `SONNET_MODEL`, `HAIKU_MODEL`, `extract_text`, `chat_completion`, `generate_structured`, `extended_thinking_completion`, `tool_completion`, `_MIN_TOKENS_WITH_THINKING`, `_FINAL_TURN_NUDGE`, `log_costs` |
| `ascent/reporting` | Debriefs, blind spots, verified-number loader. | `write_debrief`, `detect_blind_spots`, `scan_catalysts`, `load_wf_report`, `canonical_wf`, `MissingArtifact` |
| `ascent/integrations` | External non-market services. | `MiroFishClient`, `get_mirofish_sentiment`, `fetch_symbol` / `get_live_macro` / `get_cot_snapshot` (openbb), `find_analogues` |
| `ascent/dashboard` | Live dashboard loaders. | `_load_jsonl`, `_merged_weights`, `_nav_series_df` |
| `ascent/utils` | Market-clock helpers (use these instead of naive `datetime`). | `market_now`, `market_today`, `market_date_from_epoch`, `MARKET_TZ` |
| `agents/` | `us_equities_agent` (the only agent `run_all_agents.py` calls), plus `red_team_agent`, `event_agent`. Module names require the `_agent` suffix. `ai_pm_agent`, `macro_agent`, `international_agent`, `alternatives_agent` removed 2026-08-23. | `run_us_equities_agent`, `run_event_agent`, `run_red_team` |
| `orchestrator/` | Cross-agent capital allocation, skill weighting, correlation guard wiring. | `merge_agent_outputs`, `run_orchestrator`, `SKILL_SCORES_PATH`, `DEFENSIVE_AGENTS`, `REGIME_ALLOCS` |
| `compliance/` | Hash-chained audit trail, GIPS report, risk disclosures. | `AuditTrail`, `record_event`, `compute_gips_performance`, `generate_risk_disclosures` |
| `memory/` (top level) | R2R retrieval, regime memory. `ticker_memory.py` and `reflection_agent.py` removed 2026-08-23 (AI-PM-only). | `query_memory`, `ingest_verdict`, `log_episode`, `query_episodes` |
| `simulation/` | Scenario simulator. | `simulation/scenario_simulator.py` |

---

## 3. The four large files

**Line numbers drift with every edit and are a starting offset for a targeted `Read`, not a citation.** These four files moved by 20-50 lines while this map was being written. Always grep the symbol name to confirm the position before relying on it. If a number is off, the symbol name is still correct.

Re-extract fresh at any time:
`.venv/bin/python -c "import ast,sys;t=ast.parse(open(sys.argv[1]).read());[print(n.lineno,n.name) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]" <file>`

The AI PM agent module and the debate package's agents module (their line maps used to live here) were removed
2026-08-23. `run_all_agents.py` was rewritten to the core skeleton the same day (weekend branch,
startup validation, data hub, `run_us_equities_agent`, `run_orchestrator`, vol/exposure overlay,
`run_eod_with_weights`, holdings/run logging, discovery mini-rebalance) — re-extract its symbol
list with the one-liner above rather than trusting a stale line map here.

### `scripts/generate_performance_page.py` (~1915 lines)

| Line | Symbol |
|---|---|
| 34-38 | `PAPER_BASE`, `LIVE_START`, `DOCS_DIR`, `OUTPUT_PATH`, `PRICES_LIVE_PATH` |
| 51 / 104 / 126 | `fetch_portfolio_history` / `fetch_current_positions` / `fetch_spy` |
| 145-186 | display constants: `VERDICT_COLORS`, `REGIME_COLORS`, `REGIME_ACCENT`, `ACTUAL_REBALANCE_DATES`, `EXCLUDE_VERDICT_DATES`, `HARDCODED_EVENTS` |
| 232-427 | loaders: `_execution_dates_from_eod_log` 232, `load_regime_bands` 327, `load_latest_allocation` 404 |
| 464 / 470 / 474 | `_REDACTION_LABELS` / `_redaction_label` / `_position_reasoning` |
| 495 / 541 | `_CONSTRUCTION_STAGES` (redaction-audited copy) / `_construction_section_html` |
| 604 / 662 | `compute_stats` / `build_chart_data` |
| 693 | `_wf_honesty_line` |
| 717-1204 | HTML section builders: `_timeline_html` 1047, `_positions_html` 1091, `_book_section_html` 1204 (the AI-PM/debate-specific builders `_earned_authority_html`, `_promotion_gates_html`, `_override_scorecard_html`, `_thesis_html`, `_debate_html`, `_verdict_section_html` still exist as of 2026-08-23 but now source only frozen, non-updating logs — see CLAUDE.md constraint 5) |
| 1255 / 1477 | `_EDITORIAL_CSS` / `_PAGE_JS` |
| 1531 | `build_html` |
| 1788 / 1833 | `_update_readme_stats` / `main` |

`debate/agents.py`'s line map (bull/bear/devil's-advocate/regime-specialist agents, sanity
thresholds, round-2 rebuttals) was removed 2026-08-23 along with the file.

---

## 4. Data and artifact map

All paths verified present. Writers/readers determined by grepping the path string (worktrees excluded).

| Artifact | Holds | Written by | Read by |
|---|---|---|---|
| `data_cache/prices_live.parquet` | Yahoo live daily bars. Provenance is load-bearing: `prices_simulated`, `prices_live_fallback_simulated`, `prices_live_clean_refetch` (staging), `prices_macro`, `macro_live`/`macro_simulated`, `profiles` are distinct names, never aliased. | `ascent/data/hub.py`, `ascent/data/store/parquet.py` (`save_parquet`) | `ascent/main.py`, agents, `ascent/research/*`, `ascent/risk/*`, `ascent/monitoring/*`, `scripts/run_ascent_wf.py` |
| data_cache/active_alpha_config.json | Live sleeve weights after self-improve / shadow promotion / slippage feedback. **Not a permanent file** — deliberately deleted 2026-08-14 (stale-sleeve-name corruption, see `docs/superpowers/plans/2026-08-14-alpha-weight-override-fix.md`); regenerates on the next write, absent until then is expected, not broken. | `ascent/research/shadow_promoter.py`, `ascent/research/self_improve.py`, `ascent/monitoring/slippage_ic_feedback.py` | `ascent/alpha/stack.py`, `ascent/alpha/altdata_alpha.py`, `ascent/data/validate/altdata_validator.py` |
| `data_cache/ml_model_us_equities.pkl`, `sleeve_posteriors.json` | ML sleeve model (must carry `feature_names`), meta-learner posteriors. | `ascent/alpha/ml_sleeve.py`, `ascent/alpha/meta_learner.py` | `ascent/alpha/stack.py` |
| `logs/eod_log.jsonl` | One record per EOD execution (orders, NAV). Canonical execution history. | `run_all_agents.py`, `ascent/execution/eod_runner.py` | `ascent/reporting/debrief.py`, `ascent/monitoring/live_vs_backtest.py`, `ascent/monitoring/pre_rebalance_checklist.py`, `ascent/execution/kill_switch.py`, `scripts/heartbeat_check.py`, `scripts/generate_performance_page.py` |
| `logs/multi_agent_run.jsonl` | Per-run agent outputs and merged weights. | `run_all_agents.py` | `scripts/generate_performance_page.py` |
| `logs/holdings_log.jsonl` | Daily position snapshots. | `run_all_agents.py` (`_log_holdings`) | `ascent/dashboard/live_dashboard.py`, `ascent/monitoring/weekly_debrief.py`, `ascent/monitoring/scenario_planner.py` |
| logs/sleeve_ic_log.jsonl | Rolling per-sleeve IC. Drives the IC gate. **Not a permanent file** — deliberately deleted 2026-08-14 alongside `active_alpha_config.json` (same stale-data cleanup); `_get_gated_weights()` degrades safely to a no-op when it's absent, and it regenerates automatically on the next live pipeline run once trading resumes. Absent today because live trading has been paused since 2026-07-27, not because anything is broken. | `ascent/main.py` (`_log_sleeve_ic`) | `ascent/alpha/stack.py`, `ascent/alpha/meta_learner.py`, `ascent/monitoring/signal_health.py` |
| `logs/audit_trail.jsonl` | Hash-chained event log. | `compliance/audit_trail.py` | `scripts/verify_audit_trail.py` |
| Other `logs/*.jsonl` | `alerts`, `attribution_log`, `hedge_log`, `slippage`, `event_trades`, `regime_episodes`, `skill_scores_log`, per-agent `*_pnl.jsonl`. Each has one owning module named after it. `ai_pm_*`, `counterfactual_daily.jsonl`, `alpha_wedge.jsonl`, `stocktwits_ic.jsonl` are frozen, non-updating history from the removed AI-PM layer — read-only, do not resurrect the writer. | see owning module `_LOG` / `LOG_PATH` constant | various |
| `outputs/wf_results/wf_report_*.json`, `wf_equity_*.csv` | Walk-forward fold results and equity curves. | `scripts/run_ascent_wf.py`, `ascent/research/wf_framework/metrics.py` | `ascent/reporting/verified_numbers.py` (`load_wf_report`, `canonical_wf`), `scripts/verify_docs.py` |
| `outputs/ai_pm_theses/`, `scenarios/` | Formatted thesis docs, scenario plans (`investor_reports/` output and its generator, and the thesis formatter, removed 2026-08-23). | `ascent/monitoring/scenario_planner.py` | dashboard generator |
| `execution/merged_weights.json` | The live target book (orchestrator-merged allocation). | `run_all_agents.py`, `ascent/execution/eod_runner.py` | `ascent/portfolio/hedge_overlay.py`, `ascent/monitoring/{position_health,scenario_planner,weekend_runner,pre_rebalance_checklist}.py`, `ascent/dashboard/live_dashboard.py`, `compliance/risk_disclosure.py` |
| `execution/pending_approvals.json`, `approval_override.json` | Human-approval queue. | `ascent/execution/approval_server.py` | same |
| `dashboard/regime_signal.json`, `regime_labels.csv` | Current regime signal + historical labels. | `ascent/regime/engine.py`, `run_all_agents.py` | `scripts/generate_performance_page.py`, `scripts/evaluate_hedge.py`, `ascent/research/factor_discovery/discovery_runner.py`, `ascent/monitoring/regime_trajectory.py` |
| `dashboard/factor_exposures.json` | Portfolio factor exposures. | `ascent/risk/factor_exposure.py` (`export_factor_exposures`) | dashboard |
| `dashboard/agent_skill_scores.json` | 63-day rolling Sharpe per agent. Drives orchestrator capital allocation. | `ascent/monitoring/skill_tracker.py` (`export_skill_scores`) | `orchestrator/central_intelligence.py` (`SKILL_SCORES_PATH`) |
| `dashboard/live_vs_backtest.json` | Live vs backtest comparison. | `ascent/monitoring/live_vs_backtest.py` | dashboard |
| `rebalance_calendar.csv` | Single column `rebalance_date`. Defines scheduled rebalance days. | manual | `run_all_agents.py`, `ascent/utils/market_time.py`, `scripts/heartbeat_check.py`, `scripts/generate_performance_page.py` |
| `docs/index.html` | Public GitHub Pages dashboard. | `scripts/generate_performance_page.py` | published |

Backups follow a `*.bak*` / `*.pre_*.bak.parquet` convention and `data_cache/_corrupt_backup_*/`. Do not read them as current state.

---

## 5. Test map

124 `test_*.py` files. Config in `pyproject.toml` (`testpaths = ["tests"]`, `addopts = "-v --tb=short"`).

Flat files in `tests/` are historical: named per feature (`test_hedge_overlay.py`, `test_mvo_optimizer.py`), per phase (`test_phase1_hardening.py` ... `test_phase6_signals.py`), and per plan (`test_plan_a.py` ... `test_plan_d.py`). Subdirectories are the newer convention.

| Subdir | Covers |
|---|---|
| `tests/agents/` | Emptied by the 2026-08-23 AI PM removal (tools, calibration gate, fallback/force-seal, financials, news, pre-thesis fixes were all AI-PM-only) — check whether anything still lives here before adding new tests. |
| `tests/alpha/` | Meta-learner, stack weight resolution |
| `tests/data/` | New ingest sources, CBOE options time bound |
| `tests/features/` | Individual feature defs (HY spread direction, sector relative momentum) |
| `tests/integrations/` | OpenBB client |
| `tests/monitoring/` | Alerts/liveness, position health, rebalance trigger, regime trajectory, signal health, weekend mode |
| `tests/portfolio/` | Exposure overlays, long/short, risk budget cap, risk construction |
| `tests/regime/` | AI-vs-quant regime blend |
| `tests/scripts/` | Heartbeat check, LLM cache seeding |
| `tests/strategy/` | Discovery guards |
| `tests/test_wf_framework/` | Walk-forward engine, windows, strategies, execution, optimizer, metrics |

Useful subsets:

```bash
.venv/bin/python -m pytest -q                          # everything (slow)
.venv/bin/python -m pytest tests/portfolio tests/alpha -q
.venv/bin/python -m pytest tests/test_wf_framework -q   # slow
.venv/bin/python -m pytest tests/agents tests/strategy -q
.venv/bin/python -m pytest tests -q -k "regime and not wf"
.venv/bin/python -m pytest tests/test_parquet_store_dedup.py tests/test_ic_outlier_robustness.py -q
.venv/bin/python scripts/verify_docs.py                 # doc drift, not pytest
```

Known failures (verified by running the specific suspects):

| Test | Status |
|---|---|
| `tests/integrations/test_openbb_client.py::test_get_cot_snapshot_returns_dict` | FAILS. Needs network (`get_cot_snapshot` returns `None` offline). Not a regression. |
| `tests/test_wf_framework/test_metrics.py` | PASSES (10/10). The buggy-Sortino issue was fixed; `verify_docs.py` has `check_sortino_annualized_once` guarding it. Old notes calling this a known failure are stale. |
| `tests/test_wf_framework/test_ascent_engine.py` | FAILS. Observed `FF` on the first two of six tests in two separate runs. Also very slow and timing-unstable (each test runs a real multi-fold walk-forward; individual tests have exceeded 9 minutes). The rest of `tests/test_wf_framework/` passes 54/54 in under 10s, so use `--ignore=tests/test_wf_framework/test_ascent_engine.py` for fast iteration. |

`tests/test_walkforward_institutional.py` passes (5/5).

---

## 6. Common tasks to where to look

| Task | Files | Symbols |
|---|---|---|
| Add an alpha sleeve | `ascent/alpha/<new>.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py` | Add to `DEFAULT_ALPHA_WEIGHTS` in **both** stack.py and self_improve.py; add to `_SPARSE_FILL_ZERO` if the panel is sparse; `build_alpha_stack` |
| Change position sizing / caps | `ascent/portfolio/optimizer.py` | `enforce_constraints`, `_water_fill_cap`, `enforce_cluster_cap`, `enforce_risk_budget_cap`, `top_n_equal_weight`, `rank_weighted` |
| Change gross exposure / hedging | `ascent/portfolio/exposure.py` | `apply_exposure_overlays`, `MA_WINDOW`, `VOL_TARGET`, `ma_filter_scale`, `vol_target_scale` |
| Find why a weight got capped | `ascent/portfolio/optimizer.py`, `orchestrator/central_intelligence.py`, `ascent/risk/correlation_guard.py` | `_water_fill_cap` post-condition, `SectorDataError`, `enforce_cluster_cap`, `merge_agent_outputs`, `check_cross_agent_correlation` |
| Trace why a sleeve was zeroed | `ascent/alpha/stack.py`, `ascent/main.py`, logs/sleeve_ic_log.jsonl (see data-file table above — may not exist until the next live run) | `IC_GATE_THRESHOLD`, the gate block near stack.py line ~119-170, `_log_sleeve_ic`, `_winsorize_rows` |
| Debug a bad rebalance | `outputs/debate_log/verdict_<date>.json`, `logs/eod_log.jsonl`, `run_all_agents.py` (`main`, execution from ~1653), `ascent/execution/eod_runner.py` | `verdict.reasoning`, `position_changes`, `run_eod_with_weights`, `_enforce_reduce_size`, `_apply_falsifier_trim`, `_log_run` |
| Trace an order that should not have fired | `ascent/execution/eod_runner.py`, `ascent/execution/order_engine.py`, `ascent/execution/alpaca_broker.py` | `run_eod_with_weights` (`force=` no-ops off-rebalance), `compute_orders`, `MIN_TRADE_THRESHOLD`, `submit_order` |
| Add an LLM call | `ascent/llm/client.py` (import only, do not redefine) | `chat_completion`, `generate_structured`, `tool_completion`, `extract_text`, `DEFAULT_MODEL` / `SONNET_MODEL` / `HAIKU_MODEL`. Never index `content[0].text`; never pass `temperature` / `thinking=` |
| (AI PM tools, debate behaviour, AI PM authority/guardrails) | — | Removed 2026-08-23; see CLAUDE.md constraint 5 and the 2026-08-23 session log entry rather than looking for these files. |
| Add a data source | `ascent/data/ingest/<new>.py`, `ascent/data/hub.py`, `ascent/data/validate/altdata_validator.py` | `save_parquet` with a provenance-honest cache name, `run_hub`, `register_altdata_source`, `validate_altdata_source` |
| Change the regime model | `ascent/regime/model.py`, `engine.py`, `integration.py` | `RegimeModel`, `RegimeEngine(config=dict)`, `walk_forward_model_select`, `check_emergency_refit_triggers` |
| Change the public dashboard | `scripts/generate_performance_page.py` | `build_html` (~1486), the `_*_html` builders, `_EDITORIAL_CSS`, `_CONSTRUCTION_STAGES` (confidentiality-audited copy) |
| Add a doc-drift guard | `scripts/verify_docs.py` | write `check_*` returning `(ok, detail)`, register in `CHECKS` |
| Reproduce the walk-forward number | `scripts/run_ascent_wf.py`, `outputs/wf_results/`, `ascent/reporting/verified_numbers.py` | `--live-system`, `load_wf_report`, `canonical_wf`, `CANONICAL_WF_ARTIFACT` |

