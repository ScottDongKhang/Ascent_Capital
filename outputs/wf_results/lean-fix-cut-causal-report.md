# Cut `ascent/causal/` — BLOCKED at verification step

Date: 2026-08-16
Branch: main (no commits made — no code was changed)

## Step 1: Read `ascent/causal/` in full (859 LOC across 6 files)

All files read in full:

- `ascent/causal/__init__.py` (1 LOC) — module docstring only, says "causal intelligence for the AI PM"
- `ascent/causal/causal_discovery.py` (219 LOC) — runs the PC constraint-based causal discovery
  algorithm (`causallearn`) on FRED macro series + sector ETF weekly returns, writes
  `data_cache/macro_causal_dag.json`. Entry point `run_discovery()`.
- `ascent/causal/dag_builder.py` (295 LOC) — per-symbol causal graph builder. Calls Haiku
  (`generate_structured`) to produce 1-3 falsifiable causal mechanisms per holding from
  fundamentals/transcript/SEC summaries. Caches to `data_cache/causal_graphs/{symbol}_{q}.json`.
  Also exposes `build_portfolio_graphs()` (batch entry point) and `load_or_build()` (cache-only read).
- `ascent/causal/tracker.py` (253 LOC) — writes one prediction record per mechanism
  (`write_predictions`), checks outcomes weekly (`check_outcomes`), flags early exits daily
  (`check_early_exits`), and reports accuracy stats (`get_track_record`). Logs to
  `logs/causal_predictions.jsonl`.
- `ascent/causal/compatibility.py` (66 LOC) — pure static dict lookup, no I/O/LLM. Filters
  mechanisms by regime compatibility (`regime_compatible`, `filter_mechanisms`).
- `ascent/causal/velocity.py` (25 LOC) — pure function, no imports. `mechanism_velocity_score()`.

## Step 2: Grep verification — NOT self-contained to `agents/ai_pm_agent.py`

Ran a repo-wide grep (excluding `.venv`, `.git`, `.claude/worktrees`) for
`ascent\.causal|ascent/causal|causal_discovery|dag_builder|causal_graphs|macro_causal_dag|CausalMechanism|causal_predictions`.

Real (non-test) callers found, beyond `agents/ai_pm_agent.py`:

1. **`run_all_agents.py:1460`** —
   ```python
   from ascent.causal.tracker import get_track_record as _get_track_record
   _causal_track_record = _get_track_record()
   ...
   ai_pm_result = run_ai_pm(..., causal_track_record=_causal_track_record, ...)
   ```
   This call site lives in `run_all_agents.py` itself, not inside `agents/ai_pm_agent.py`. Its
   output is still destined for the AI PM, so this alone would likely have been in-scope to
   remove, but it's evidence the integration surface is wider than "one file."

2. **`ascent/monitoring/weekend_runner.py:317-340`** — two scheduled jobs that call
   `ascent.causal` directly, independent of any AI PM call:
   ```python
   def _job_causal_macro_dag() -> None:
       """Run causal discovery on FRED + sector ETFs → macro_causal_dag.json."""
       from ascent.causal.causal_discovery import run_discovery
       ...
       dag = run_discovery(regime=regime)

   def _job_causal_graph_builder(portfolio_symbols: list) -> None:
       """Build/refresh Haiku causal graphs for all current holdings."""
       from ascent.causal.dag_builder import build_portfolio_graphs
       ...
       results = build_portfolio_graphs(portfolio_symbols)
   ```
   These are registered as real weekend jobs (`weekend_runner.py:410-416`,
   `once_per_weekend=True` for the DAG job). `weekend_runner.run_weekend()` is invoked live from
   `run_all_agents.py:670` (`from ascent.monitoring.weekend_runner import
   already_ran_this_weekend, run_weekend`). This is **exactly** the "periodic invocation
   separate from the AI PM" case flagged as possible in task step 6 — confirmed present, not
   hypothetical.

3. **`ascent/strategy/falsifier_registry.py:376`** —
   ```python
   def _causal_early_exits() -> list:
       try:
           from ascent.causal.tracker import check_early_exits
           return check_early_exits() or []
       except Exception:
           return []
   ```
   This feeds the falsifier registry, a **different** advisory subsystem from the AI PM
   (integrity constraint #5's third named mechanism, "the falsifier trim"). It is not gated
   behind or downstream of `agents/ai_pm_agent.py` at all.

   `ascent/strategy/` is explicitly named in this task's own global constraints as off-limits
   ("Do NOT touch `debate/`, `ascent/strategy/`, ..."). A caller inside a file I'm barred from
   touching, that is not `agents/ai_pm_agent.py` or one of its tests, is precisely the condition
   the task instructions define as a stop condition:

   > "If you find a caller outside `agents/ai_pm_agent.py` and its own tests, STOP and report
   > as a blocker rather than deleting."

Other references found, not blockers:
- `ascent/config/types.py` defines the `CausalMechanism` dataclass — this is a plain dataclass
  in the config layer, does not import `ascent.causal`, and is a shared type, not a behavioral
  caller of the package logic. No action needed regardless of outcome.
- `tests/test_dag_builder.py`, `tests/test_causal_velocity.py`, `tests/test_causal_discovery.py`,
  `tests/test_causal_compatibility.py`, `tests/test_causal_tracker.py`,
  `tests/test_ai_pm_prethesis_causal.py` — all test files, expected.

## Conclusion: BLOCKED

The background's premise — "`ascent/causal/` feeds ONLY the AI PM" — does not hold under
verification. Concretely:

- `ascent/monitoring/weekend_runner.py` runs causal discovery and per-symbol DAG building as
  **standalone scheduled jobs**, live-wired into `run_all_agents.py` via `run_weekend()`. This
  is caching/compute work independent of any single AI PM invocation.
- `ascent/strategy/falsifier_registry.py` consumes `ascent.causal.tracker.check_early_exits()`
  directly, for a **different advisory mechanism** (the falsifier trim) that this task's global
  constraints explicitly forbid touching.

Per the task's own stop condition, I have not modified or deleted anything: no changes to
`agents/ai_pm_agent.py`, no deletion of `ascent/causal/`, no test deletions, no edits to
`run_all_agents.py`. Everything reported above is read-only verification.

## What re-scoping would need to resolve

- Whether cutting `ascent/causal/` is meant to also cut the weekend jobs in
  `weekend_runner.py:_job_causal_macro_dag` / `_job_causal_graph_builder` (their sole purpose is
  populating caches that only `dag_builder.load_or_build()` in `agents/ai_pm_agent.py` reads —
  plausibly fine to remove alongside, but that's a scope decision, not mine to make here).
- Whether `ascent/strategy/falsifier_registry.py:_causal_early_exits()` is in-scope. Since
  `ascent/strategy/` is explicitly protected in this task, removing its only caller of
  `ascent.causal.tracker.check_early_exits` would require either (a) an explicit carve-out
  permitting a one-line touch to `falsifier_registry.py`, or (b) leaving `ascent/causal/tracker.py`
  (at minimum) in place and only trimming the AI PM's own integration points.
- `run_all_agents.py:1460`'s `_causal_track_record` load — likely fine to remove (feeds AI PM
  only) but is a second file beyond `agents/ai_pm_agent.py` that would need touching, which the
  task scoped to include implicitly (see step 6) but I'm flagging for the controller's explicit
  sign-off given how the other two callers turned out.

## LOC removed

0 — no deletions performed (blocked before step 3).
