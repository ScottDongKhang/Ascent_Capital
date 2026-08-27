# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha (2 sleeves) →
portfolio → walk-forward → regime → **one** capital-allocating agent (`us_equities`) →
orchestration → Alpaca paper trading.

**2026-08-23: the AI PM, debate layer, falsifier registry, earned-authority ladder, causal
module, and the three dormant specialist agents (`macro`, `international`, `alternatives`)
were removed outright**, not merely gated advisory-only. They had run advisory-only since
2026-08-14 and the measured counterfactual was negative-or-insignificant on every axis (see
`docs/session_log_archive.md` and the archived `CURRENT_VERIFIED_NUMBERS.md` section 3 for the
numbers this decision was based on) — carrying the complexity forward bought nothing. The goal
changed at the same time: this system no longer targets beating SPY. It targets steady,
low-volatility monthly returns with capital preservation as the first priority. See integrity
constraint #5 for exactly what was removed and why "removed" replaces "advisory."

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
DEFAULT_MODEL = "claude-opus-5"             # reserved for future synthesis use
SONNET_MODEL  = "claude-sonnet-5"           # reserved for future breadth use
HAIKU_MODEL   = "claude-haiku-4-5-20251001" # classifiers, weight adjustment
```
DEFAULT_MODEL/SONNET_MODEL have no live caller since the 2026-08-23 noise-layer removal (their
callers were the AI PM and debate layer). Left defined for whatever uses this system takes on
next; do not resurrect the old callers without a fresh design.

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
                  dashboard, strategy, integrations, utils
agents/           us_equities_agent, red_team_agent, event_agent
orchestrator/     central_intelligence.py
run_all_agents.py daily entrypoint
ascent/main.py    pipeline entrypoint (called by each agent)
data_cache/       parquet caches, active_alpha_config.json
logs/             eod_log.jsonl, multi_agent_run.jsonl
```

The debate package, the causal DAG module, the AI PM agent, the three dormant specialist
agents, `ascent/memory/`'s decision-memory tracker, the earned-authority ladder, the falsifier
registry, and their satellite modules no longer exist — removed 2026-08-23. See integrity
constraint #5 for the full list.

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
5. **The AI PM / debate / falsifier / earned-authority / dormant-agent layer has been removed
   outright — not gated, not advisory, gone.** History, for judgment calls later:
   - Gated advisory-only since 2026-08-14 (debate judge's position change, the earned-authority
     blend, and the falsifier trim's `run_eod_with_weights(..., dry_run=False, force=True)`
     call all had their live-write call sites removed that day).
   - Measured through 2026-08-23: `debate_judge_intervention` and `earned_authority` both
     scored CUT in the proof audit; actual-book-vs-pure-quant and pure-AI-PM-vs-pure-quant were
     both negative and not statistically significant (see `CURRENT_VERIFIED_NUMBERS.md`
     section 3, retained as history); the falsifier trim was never measured at all (not one of
     the 23 proof-audit components, built entirely on unmeasured AI PM output). No axis showed
     positive, proven value.
   - **Removed 2026-08-23**, along with everything that only existed to feed, score, or log
     it: the AI PM agent module, the entire debate package (runner, judge, adversarial
     engine/authority/monitor, agent tools, disagreement scorer, outcome tracker), the
     earned-authority ladder, the falsifier registry, the AI PM counterfactual/learning/
     guardrails/perf-feedback modules, the conviction gate, the investor-letter and
     investor-report generators, the three dormant specialist agents (macro/international/
     alternatives — never invoked by `run_all_agents.py` even before this), the causal DAG
     module, and about a dozen smaller satellite modules (ticker discovery, calibration
     tracking, conviction/alpha-wedge tracking, decision/ticker memory, the stocktwits/exa
     news integrations). See `docs/session_log_archive.md`'s 2026-08-23 entry for the full
     file list. `run_all_agents.py` and `ascent/execution/eod_runner.py` were rewritten to
     drop every call site into this layer.
   - `scripts/verify_docs.py::check_noise_layer_removed` asserts these paths stay gone. If you
     are tempted to bring any of them back, that check is the thing to update, and it should
     only change alongside an artifact-backed positive result — the standard this layer never
     met is exactly why it is gone rather than merely re-gated.
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
- **Discovery mini-rebalance is add-only**: `_insert_candidate_weights`, not a full agent
  re-run. Suppressed near a scheduled rebalance via `_is_near_scheduled_rebalance`.
- **`run_eod_with_weights()` silently no-ops on non-rebalance days** — pass `force=True` on
  discovery / mini-rebalance paths.
- **Live trading and the walk-forward backtest must draw from the same universe.**
  `eod_runner.py`'s tradeable-symbol filter calls `build_historical_universe(strict=True,
  sp500_only=True)`, matching `walk_forward_runner.py` exactly — both restricted to the
  survivorship-correct S&P 500 + tracked removals, all with real addition dates. Do not
  loosen the live call back to `strict=False` / `sp500_only=False`: that pulls in
  non-S&P500 symbols with no recorded addition date, which silently default their
  `start_date` to `UNIVERSE_START`, so live trading would select from and backdate symbols
  the backtest never validated on.
- **`prices_macro` / `prices_international` / `prices_alternatives` are stale, historical-only
  caches.** They were the three dormant specialist agents' own price data; nothing writes or
  reads them since those agent modules were removed 2026-08-23. Leave them on disk (harmless)
  but do not treat them as live.

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

1. **Reasoning behind the decision.** Explain why *these* trades — which alpha sleeve(s) drove
   which positions, what the regime engine's posture was, what the optimizer's vol-target
   overlay did to gross exposure. There is no debate verdict to quote anymore (removed
   2026-08-23); the reasoning is now the quant pipeline's own signal chain, so trace it there.
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

- `CURRENT_VERIFIED_NUMBERS.md` — performance, risk, and what is explicitly *not* verifiable.
  This file wins any disagreement. Its SYSTEM 2 (AI-native layer) section describes a system
  that no longer exists as of 2026-08-23 — read it as history, not current state.
- `dashboard/regime_labels.csv` — the current regime label. Note that
  `dashboard/regime_signal.json` can lag it.
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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- graphify-out/obsidian/ is a standalone code-knowledge vault (one note per node, plus graph.canvas), separate from the user's main Obsidian second-brain vault. It answers "how does the code connect" questions; the main vault answers "what did we decide and why." Consult both when a question touches project history as well as code structure — the second-brain instructions in the user's global CLAUDE.md still apply on top of this.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
