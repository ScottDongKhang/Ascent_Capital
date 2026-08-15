# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha (2 sleeves) →
portfolio → walk-forward → regime → **one** capital-allocating agent (`us_equities`) →
orchestration → Alpaca paper trading. The AI PM, the debate layer, and the falsifier registry
all still run, but **advisory only**: they produce theses, verdicts, and proposals that are
logged and scored, and none of them writes to live weights (see integrity constraint #5). The
other three specialist agents (`macro`, `international`, `alternatives`) are code-intact but
not invoked in the daily run.

**Daily command**: `.venv/bin/python run_all_agents.py` (branches on rebalance day)

---

## Read this first

This file holds **durable rules only** — things that are true across sessions and cannot be
derived by reading the code. It deliberately contains **no performance numbers, no dates, and
no line numbers**, because those go stale within days and a stale figure in an always-loaded
file becomes a confidently-wrong belief rather than a harmless typo.

| You need | Read |
|---|---|
| Any performance / risk figure | `CURRENT_VERIFIED_NUMBERS.md` — the only citable source |
| Where code lives, what to grep | `docs/REPO_MAP.md` |
| What happened in past sessions | `docs/session_log_archive.md` |
| Whether this file is still true | `.venv/bin/python scripts/verify_docs.py` |

**`scripts/verify_docs.py` is the guard on this file.** Every mechanically checkable claim
below has a corresponding check. Run it after touching code that this file describes; a red
check means either the code changed or this file lied. Fix one of those two — never the guard.

**Grounding rule.** Every number you report must come from a named artifact you actually read
in this session. If you reconstructed it, say so. A confident wrong number is worse than an
acknowledged gap — this project has published a synthetic drawdown as fact before.

---

## Environment

Python 3.12.13, venv at `.venv/`. Always use `.venv/bin/python`. API keys via
`APIKeys.from_env()`. Config via `get_config()` — never `Config()` directly.

---

## LLM models (`ascent/llm/client.py`)

Import constants from here — never redefine locally:
```python
DEFAULT_MODEL = "claude-opus-5"             # AI PM synthesis (Phase 2)
SONNET_MODEL  = "claude-sonnet-5"           # debate agents, red team, pre-thesis (Phase 1)
HAIKU_MODEL   = "claude-haiku-4-5-20251001" # classifiers, weight adjustment
```

**Claude 5 rules** (migrated from the 4.6 generation):
- **Never index `resp.content[0].text` for a Claude 5 model** — thinking is ON by default, so
  block 0 is usually a thinking block. Use `extract_text` from `ascent/llm/client.py`.
  *Carve-out*: existing `content[0].text` sites are Haiku-only `messages.create` calls where
  block 0 really is the text. They are safe, and the guard keeps them Haiku-only — repointing
  one at Opus or Sonnet trips `content0_haiku_only`.
- **Never pass `temperature` / `top_p` / `top_k`** — 400 on Claude 5. The `temperature` kwarg
  still exists on the wrappers for call-site compatibility and is silently dropped for
  Claude 5 models; steer with prompts instead.
- **Never pass `thinking={"type": "enabled", "budget_tokens": N}`** — 400. Depth is set by
  `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`). Wrapper defaults:
  `chat_completion`/`generate_structured` = medium, `extended_thinking_completion` and
  `tool_completion` = high. The one surviving `budget_tokens` call is the legacy branch inside
  `ascent/llm/client.py`, gated by `_CLAUDE_5_MODELS`; do not add another anywhere else.
- **`max_tokens` caps thinking + visible text together.** The wrappers raise it to
  `_MIN_TOKENS_WITH_THINKING`; direct `messages.create()` callers must leave their own
  headroom or thinking consumes the whole budget and returns no text.
- **Keep thinking enabled in tool loops.** With thinking disabled, Claude 5 can emit a tool
  call as visible text instead of a `tool_use` block — the turn succeeds and the tool
  silently never runs.
- **Echo thinking blocks back unchanged** when continuing a turn (handled inside
  `tool_completion`); dropping or editing them is rejected.
- Haiku 4.5 is current and unchanged — it keeps the legacy parameter path.

---

## Key layout

```
ascent/           core engine: config, data, features, alpha, portfolio, backtest,
                  research, regime, risk, reporting, execution, monitoring, llm,
                  dashboard, strategy, causal, integrations, memory, utils
agents/           us_equities_agent, macro_agent, international_agent,
                  alternatives_agent, ai_pm_agent, red_team_agent, event_agent
orchestrator/     central_intelligence.py
debate/           debate_runner, agents, judge, adversarial_authority,
                  adversarial_engine, adversarial_monitor, agent_tools,
                  disagreement_scorer, outcome_tracker
run_all_agents.py daily entrypoint
ascent/main.py    pipeline entrypoint (called by each agent)
data_cache/       parquet caches, earned_authority.json, active_alpha_config.json,
                  ai_prethesis_latest.json, ai_regime_assessment.json
logs/             eod_log.jsonl, multi_agent_run.jsonl, ai_pm_calibration.jsonl
outputs/debate_log/  verdict_<date>.json, debrief_<date>.json
```

See `docs/REPO_MAP.md` for what to grep for and which files are worth reading whole.

---

## Integrity constraints (never violate)

1. **No look-ahead bias** — walk-forward uses `get_universe_on_date()` per fold; regime fitted
   on the training slice only.
2. **No simulated data under live cache names** — `prices_live` = Yahoo live only; on
   fetch failure the name becomes `prices_live_fallback_simulated`.
3. **Max-weight hard cap** via `_water_fill_cap()` in `ascent/portfolio/optimizer.py`, with a
   post-condition check.
4. **Sector constraint**: under 80% coverage → skip caps and warn. Never collapse to a single
   name.
5. **The three judgment layers are advisory only — none of them writes to live weights.**
   Named precisely, because "no exceptions" hides what is being excepted:
   - **The debate judge's position change** (`apply_judge_position_change`). Both call sites
     removed 2026-08-14; `debate_judge_intervention` scored CUT (p=0.75) in the proof audit.
   - **The AI PM's earned-authority blend** (`ascent/strategy/earned_authority.py`
     `authority_blend`/`blend`). Blend call site removed 2026-08-14; `earned_authority` scored
     CUT (p=0.35, track_d vs track_astar).
   - **The falsifier trim** (`_apply_falsifier_trim`). Its `run_eod_with_weights(...,
     dry_run=False, force=True)` — a real order — was removed 2026-08-14. Not because it was
     measured negative but because it was **never measured at all**: it is not one of the 23
     components in `ascent/analyst/proof_audit/components.py`, and it was built entirely on
     unmeasured AI PM output.

   The rule behind all three: **an unproven or unmeasured live-write mechanism goes
   advisory-only until it is actually proven.** All three keep running, keep producing
   verdicts / theses / fired falsifiers, and keep recording their proposals
   (`record_intervention(..., applied=False)`) so the counterfactual evidence for ever
   reinstating them keeps accumulating. Nothing under `debate/` has ever written to alpha,
   portfolio, or execution, and now nothing in `run_all_agents.py` does so on their behalf
   either. Do not reinstate any of the three without an artifact-backed positive result.
6. **Alpha sleeve set**: the active sleeves are `meanrev` and `statarb` (2 sleeves). Update
   `DEFAULT_ALPHA_WEIGHTS` in BOTH `ascent/alpha/stack.py` AND `ascent/research/self_improve.py`
   if changing the set. The guard enforces that their key sets match.
7. **Fundamental sleeve is disabled** (measured anti-signal). Do not re-enable without a
   positive, artifact-backed IC-t.

---

## Non-obvious gotchas

- **Agent module names**: `agents.us_equities_agent`, not `agents.us_equities` — the `_agent`
  suffix is required for all of them.
- **`RegimeEngine` constructor** takes `config=dict`, not a `Config` object.
- **`bdate_range(end="today")`** returns empty on weekends — use an explicit weekday rollback.
- **Market dates**: the host runs at UTC+7, so `date.today()` is a day ahead of the US market
  for much of the day. Use `ascent/utils/market_time.py`.
- **ML sleeve cache** must store `feature_names` — XGBoost crashes on shape mismatch if the
  feature set changes between writes.
- **AI PM is two-phase**: Phase 1 `run_ai_pm_prethesis()` uses `SONNET_MODEL` (breadth),
  Phase 2 `run_ai_pm(prethesis=...)` uses `DEFAULT_MODEL` (judgment). Never swap them.
- **Pre-thesis runs AFTER the quant agents and the orchestrator merge**, despite some stale
  in-code comments saying otherwise. It writes `data_cache/ai_regime_assessment.json`, which
  is therefore consumed by the *next* run, not the current one. Failure → `prethesis=None` →
  graceful single-phase fallback.
- **`propose_prethesis` vs `propose_portfolio`**: different tools, different result stores.
  Phase 1 ends with the former, Phase 2 with the latter.
- **`run_ai_pm(quant_outputs=...)`**: pass the `agent_outputs` list or AI PM re-runs all four
  agents and wastes minutes.
- **`ascent/main.py` `run_pipeline` returns a 10-tuple** (last element `_alpha_breakdown`) —
  `eod_runner.py` and `us_equities_agent.py` unpack it positionally.
- **`_SPARSE_FILL_ZERO`** in the ML sleeve must include ALL sparse panels, or a NaN-drop
  silently disables those sleeves.
- **PDF generation**: use `reportlab`. `weasyprint` is installed but fails to import here
  (missing system Pango/GObject).
- **The cross-agent correlation guard can no longer fire.** `check_cross_agent_correlation`
  in `orchestrator/central_intelligence.py` is gated on `len(agent_weights) >= 2`, and only
  `us_equities` runs, so the guard is unreachable in the daily pipeline (the module and its
  tests are intact). The historical PDBC↔KMLM example — the guard halving KMLM because the
  macro and alternatives agents held correlated instruments — described exactly that
  cross-agent case and can no longer happen. Concentration *within* one agent's book is
  handled elsewhere (`_water_fill_cap`, the final position cap), not here.
- **Kill switches** stay `False` pending paper validation: `EVENT_TRADING_ENABLED`,
  `TWAP_ENABLED`, `SELF_MODIFY_ENABLED`, `LONG_SHORT_ENABLED`. The code comment on
  `LONG_SHORT_ENABLED` carries the authoritative precondition (a minimum number of paper
  rebalances) — trust it over any date written in prose.
- **Dashboard subprocess**: spawn with `cwd=repo_root` and `PYTHONPATH=repo_root`. `ascent`
  is not pip-installed, so running from `scripts/` breaks every import.
- **Same-day Track B is unreliable**: Alpaca 1D bars settle late afternoon PT, so an early
  run sees `equity == last_equity` and records a fake 0.0. Use
  `alpaca_broker.get_portfolio_history()` for settled returns.
- **`loguru` is not installed** — use `import logging`; never `from loguru import logger`.
- **MiroFish on rebalance days**: the LiteLLM proxy (port 4000 → Haiku) must be running.
  OpenRouter leaves `max_tokens` unset → 402 if credits are low.
- **Discovery mini-rebalance is add-only**: `_insert_candidate_weights`, not a full agent
  re-run. Suppressed near a scheduled rebalance via `_is_near_scheduled_rebalance`.
- **`run_eod_with_weights()` silently no-ops on non-rebalance days** — pass `force=True` on
  discovery / mini-rebalance paths.
- **The AI PM decision log only gets entries on scheduled rebalances.** Off-calendar discovery
  days run the daily-view path, not Phase 2 — a missing entry is expected, not a bug.
- **`reduce_size` cannot reliably reduce size.** When the Haiku adjustment trims too few
  positions, the fallback force-trims the largest positions and renormalizes to 1.0. Be aware
  this may trim positions that align with the judge's advisory suggestions. Diagnose
  transmission before concluding the AI PM's judgment was bad.
- **Wide-format caches carry their date in a `date` COLUMN, and every consumer restores the
  index itself.** `prices_macro` / `prices_international` / `prices_alternatives` are wide
  (one column per symbol, no id column). `save_parquet` now converts a `DatetimeIndex` input
  into a `date` column up front — reusing the existing `id_cols` / calendar-day-dedup
  machinery — because its `pd.concat(..., ignore_index=True)` and `to_parquet(index=False)`
  otherwise dropped all date information on every save. `load_parquet` is deliberately
  unchanged (it is generic, used by every cache) and returns that `date` column as-is on a
  `RangeIndex`, so **each caller must do `df.set_index("date")` itself** — and `.sort_index()`
  too if it slices positionally. Known call sites, all fixed:
  `agents/macro_agent.py`, `agents/international_agent.py`, `agents/alternatives_agent.py`
  (two per file: the fresh-cache read and the stale-cache fallback),
  `ascent/analyst/proof_audit/run.py::_load_agent_price_matrix`, and
  `ascent/risk/correlation_guard.py::_load_combined_prices` (sorts — its window is
  `returns.iloc[-63:]`). Add the restore to any new consumer.
  **Still true:** the three on-disk caches written before this fix are corrupt (dateless
  `RangeIndex`, 176k/151k/150k rows for 9-13 symbols) and cannot be repaired in place —
  deleting and re-fetching them is a separate planned data operation, not yet done.

---

## Data / caching

Cache-name provenance — never obscure it: `prices_live` (Yahoo live), `prices_simulated`
(GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`,
`macro_live` / `macro_simulated`, `profiles` (sector metadata).

Point-in-time joins: always use `as_of_join()` / `as_of_merge()` from
`ascent/data/store/point_in_time.py` for cross-dataset alignment.

**`prices_live` duplicate rows have recurred three times.** `save_parquet` dedups on a
`_calendar_day_key`, the live read path additionally defends itself with
`~index.duplicated(keep="last")` in `ascent/main.py`, and first-write dedup plus an
evening-stamp rollover were added after the second recurrence. **Do not treat the cache as
clean on trust** — a previous cleanup had already been reported as holding when it was not.
The third occurrence (2026-08-15 audit) was structurally different from the first two: a
same-day duplicate stamped at a non-midnight intraday timestamp (a late hub fetch, typically
19:00/20:00) with disjoint symbol coverage from the midnight row for that day — not caught by
the existing dedup mechanisms because `pivot_prices()`'s plain `.dt.normalize()` disagreed with
`_calendar_day_key`'s evening-rollover rule on which trading day the phantom row belonged to.
Fixed by the Task 2 repair (commit `9fd74ea`, which merged the two row populations per
(symbol, trading day) and rewrote phantom-only cells to midnight) plus the `pivot_prices`/
`validate_cache` hardening (commit `9f145fc`, both now grouping/checking against
`_calendar_day_key` consistently). Measure it: `scripts/reconcile_numbers.py` reports the live
duplicate count AND the non-midnight (phantom-row) row count in its data-integrity section.
This matters most for backtests, because the walk-forward framework does **not** dedupe on
read; only the live pipeline does.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing a fix. Never propose without
tracing first. `ast.parse` after each patch. Planning: Opus for specs → Sonnet for
implementation.

Prefer a test that reproduces the bug before the fix. Note that a long-standing "known-buggy,
don't cite" number is often a *failing test that was right all along* — check whether one
already exists before writing a new one.

---

## Rebalance recap (required after every rebalance)

After **any** run of `run_all_agents.py` that submits orders, write a four-part recap
unprompted. This is the primary interface to what the system decided; a status line is not
sufficient.

1. **Reasoning behind the decision.** Read the day's `outputs/debate_log/verdict_<date>.json`
   (`verdict.reasoning`, `verdict.key_risks`) and the adversarial intervention. Explain why
   *these* trades, which argument won which exchange. Note that the judge's verdict is advisory
   only — it does not execute or veto positions. Quote the reasoning; don't paraphrase it away.
2. **Things to watch for.** Live catalysts (FOMC, earnings, ex-div), positions on thin ice,
   guards that nearly fired, data sources that were down, anything that changes the next read.
3. **Performance since the last rebalance.** Portfolio vs SPY over the window, with
   per-position attribution when a few names dominate. Use
   `alpaca_broker.get_portfolio_history()` (settled bars) — never same-day
   `equity − last_equity`, which reads a fake 0.0 before late afternoon PT.
4. **Why performance looks like that.** The causal story, separating sizing and structure from
   stock selection from things outside the model's control.

Ground every number in a real artifact and flag anything reconstructed.

---

## Current state

Live state is **not recorded here** — it goes stale faster than this file is edited. Read:

- `CURRENT_VERIFIED_NUMBERS.md` — performance, risk, AI PM authority level, and what is
  explicitly *not* verifiable. This file wins any disagreement.
- `dashboard/regime_labels.csv` — the current regime label. Note that
  `dashboard/regime_signal.json` and `data_cache/ai_regime_assessment.json` can lag it, and
  `run_all_agents.py` reads its smart-model trigger from the JSON, so a stale label there
  changes behaviour.
- `rebalance_calendar.csv` — the next scheduled rebalance.
- `logs/eod_log.jsonl` — what actually ran, including `catch_up` runs after an outage.

Structural facts that do not go stale: the raw-return lag versus SPY is by design (defensive
non-equity sleeves, a 200MA cut, and a vol-target overlay all cost beta in an equity-only
bull); walk-forward OOS confirms positive risk-adjusted alpha with a negative
walk-forward-efficiency figure that must be disclosed as overfit.

**GitHub Pages**: `https://scottdongkhang.github.io/Ascent_Capital` — auto-updated after every
daily run, which means an unretracted wrong number there is published, not just recorded.

---

## Session log

Session history lives in `docs/session_log_archive.md`, newest first. Append there, not here.
