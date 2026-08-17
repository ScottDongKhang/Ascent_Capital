# Lean-fix: delete `ascent/causal/` — report

Date: 2026-08-16
Branch: `main`
Commits: `f8d97fe`, `948c434`

## Summary

`ascent/causal/` (859 LOC, 5 files) was confirmed dead in production this session's
investigation and removed in full, along with every integration point that fed it or
consumed its output. Deletion is behavior-preserving by construction: every consumer of
`ascent/causal/` already took the empty/zero-result branch unconditionally, because the
producer (weekend causal-discovery jobs) was structurally unreachable from the real
launchd schedule.

Total: **1861 lines removed, 16 lines added** across 21 files (2 commits).

## Verification performed before each deletion

- **`ascent/causal/` package** (5 files): re-confirmed via grep that the only non-test
  importers in the live repo (excluding `.claude/worktrees/*` copies, which are separate
  worktrees not part of this deletion) were `agents/ai_pm_agent.py`,
  `ascent/monitoring/weekend_runner.py`, and `ascent/strategy/falsifier_registry.py` — all
  three handled below.
- **6 test files** (`tests/test_causal_discovery.py`, `tests/test_causal_tracker.py`,
  `tests/test_causal_compatibility.py`, `tests/test_causal_velocity.py`,
  `tests/test_dag_builder.py`, `tests/test_ai_pm_prethesis_causal.py`): read each file's
  `def test_*` / `class` listing via `git show HEAD:<path> | grep '^def test_\|^class'`
  before deleting (deletion had already happened by the time this check ran, so it was a
  post-hoc verification against git history rather than pre-check — noted as a process
  deviation, but every file's test names were unambiguously causal-only, e.g.
  `test_run_pc_returns_dag_schema`, `test_check_early_exits_flags_catalyst_imminent_...`,
  `test_causal_mechanism_dataclass_fields`, `test_build_graph_cache_hit_skips_llm`,
  `test_aiprethesis_has_causal_mechanisms_field`). None of the six tested anything outside
  `ascent.causal` / `CausalMechanism`.
- **`data_cache/macro_causal_dag.json`**: existed on disk (1.1K, gitignored, untracked) —
  deleted directly, no commit needed.
- **`CausalMechanism` in `ascent/config/types.py`**: grepped all importers repo-wide before
  removal; confirmed only `agents/ai_pm_agent.py` and the deleted causal test files
  imported it.
- **`falsifier_registry.py`'s core falsifier logic**: left untouched — only
  `_causal_early_exits()` and its two call sites (the early-return guard and the Gate 4
  fold-in loop) were removed. Price/relative_price/macro/news falsifier logic is unchanged.
- **`debate/agents.py`**: confirmed the causal-attack block was isolated to
  `run_devils_advocate` (lines ~618-631 and the "4. CAUSAL MECHANISM ATTACK" checklist
  item) — nothing else in `debate/` was touched.

## Files touched

### Deleted (commit `f8d97fe`)
- `ascent/causal/__init__.py`, `causal_discovery.py`, `compatibility.py`, `dag_builder.py`,
  `tracker.py`, `velocity.py`
- `tests/test_causal_discovery.py`, `test_causal_tracker.py`, `test_causal_compatibility.py`,
  `test_causal_velocity.py`, `test_dag_builder.py`, `test_ai_pm_prethesis_causal.py`
- `data_cache/macro_causal_dag.json` (untracked, deleted from disk, no commit)

### Edited (commit `948c434`)
- **`agents/ai_pm_agent.py`** (177 lines removed) — the largest surface, ~14 sites:
  `AIPreThesis.causal_mechanisms` field, `get_causal_graph` tool schema entry, its
  registration in `PRE_THESIS_TOOLS`, `_PRETHESIS_RESEARCH_TOOLS`, and the executor `_map`
  dict, `_tool_get_causal_graph` function, `_assemble_causal_mechanisms` function,
  `_build_velocity_context` function (+ the now-unused `_TIMING_PRIORITY` dict),
  the `causal_mechanisms` key in `_strip_prethesis_for_phase2`'s return dict, the
  `CAUSAL INTELLIGENCE` context block built in `run_ai_pm_prethesis` (and its two
  concatenation sites into user prompts), the post-seal `causal_mechanisms` population
  block, the `causal_track_record` parameter on `run_ai_pm`, and the
  `CAUSAL THESIS TRACK RECORD` prompt block. Left the historical comment at line ~303
  referencing "CAUSAL INTELLIGENCE block" in place — it documents a past
  `_load_current_holdings` bug, not the removed feature, and is still accurate.
- **`ascent/strategy/falsifier_registry.py`** (28 lines net, mostly deletions) —
  removed `_causal_early_exits()`, its guard in `check_all()`'s early-return check, and
  the "Causal early exits (Gate 4) fold in as falsifiers" loop. Updated the module
  docstring (removed stale "causal graph falsification conditions" / "causal early-exit
  flags folded in" lines). Core price/relative_price/macro/news falsifier logic
  untouched.
- **`ascent/monitoring/weekend_runner.py`** (39 lines removed) — removed
  `_job_causal_macro_dag` and `_job_causal_graph_builder` function definitions and their
  "7b"/"7c" registrations in `run_weekend()`. The hardcoded `f"{len(completed)}/{11} jobs
  succeeded"` count was already checked: before this change there were 13 `_run_job` call
  sites (count was wrong at 11); after removing 2, there are 11 — the number is now
  accurate without needing a separate fix.
- **`run_all_agents.py`** (19 lines removed) — removed the causal track-record load
  (`from ascent.causal.tracker import get_track_record`) and the `causal_track_record`
  kwarg passed to `run_ai_pm()`, and the `causal_mechanisms` key in the `portfolio_state`
  dict construction. Updated two stale comments referencing "causal early exits" /
  "price/causal/relative_price/macro" for accuracy.
- **`debate/agents.py`** (22 lines removed) — removed the causal-mechanisms formatting
  block in `run_devils_advocate` and the "4. CAUSAL MECHANISM ATTACK" checklist item from
  its system prompt, plus the `_causal_context` concatenation. Nothing else in `debate/`
  touched.
- **`ascent/config/types.py`** (16 lines removed) — removed the `CausalMechanism`
  dataclass.
- **`docs/REPO_MAP.md`** (7 lines net) — removed the `ascent/causal` package row, removed
  `CausalMechanism` from the `ascent/config` symbol list, removed now-dead line-number
  references to `_tool_get_causal_graph`, `_assemble_causal_mechanisms`, and
  `_build_velocity_context` in the `ai_pm_agent.py` line-map section.
- **`tests/strategy/test_falsifier_registry.py`** (11 lines net) — removed the
  `_causal_early_exits` monkeypatch from the `tmp_registry` fixture and the
  `test_causal_early_exit_folds_in` test. All other falsifier tests untouched.
- **`tests/test_debate_agents.py`** (57 lines net, not originally in scope but required)
  — see "Surprise found" below.

## Surprise found (not in original scope, resolved)

`tests/test_debate_agents.py` contained two tests
(`test_devils_advocate_references_causal_mechanisms`,
`test_devils_advocate_uses_causal_mechanisms_from_portfolio_state`) that specifically
asserted the causal-attack block existed in `run_devils_advocate`'s source and prompt
output. These weren't listed in the task's scope (the debate edit itself was flagged
"optional but recommended"), and running the debate suite after editing `debate/agents.py`
surfaced 2 failures. Since the causal-attack block always received empty input in
production (confirmed by the investigation — `portfolio_state["causal_mechanisms"]` was
always `[]`), these two tests were testing dead functionality, not a live guarantee.
Replaced both with an explanatory comment rather than leaving them failing or silently
deleting them. This was not a blocker — resolved in the same commit.

## Compile / import / test verification

```
$ .venv/bin/python -m py_compile agents/ai_pm_agent.py run_all_agents.py \
    ascent/monitoring/weekend_runner.py ascent/strategy/falsifier_registry.py \
    ascent/config/types.py debate/agents.py
COMPILE_OK

$ .venv/bin/python -c "import agents.ai_pm_agent"
IMPORT_OK
```

`agents/ai_pm_agent.py` was checked for import-time side effects first (only top-level
import is `from ascent.llm.client import tool_completion, DEFAULT_MODEL, SONNET_MODEL`, no
API calls at module load) — safe to smoke-import directly.

Full test suite for every file touched by the integration-point edits
(`grep -rl "ai_pm_agent\|falsifier_registry\|weekend_runner" tests/ | grep -v causal`, 17
files):

```
177 passed in 3.55s
```

Debate suite (touched via the optional `debate/agents.py` edit), run separately as extra
verification:

```
46 passed in 2.94s   (after fixing the 2 stale causal-attack tests noted above)
```

`scripts/verify_docs.py` was run after the `docs/REPO_MAP.md` edit: 24 passed, 1
pre-existing failure (`repo_map_pointers` — missing `data_cache/active_alpha_config.json`
and `logs/sleeve_ic_log.jsonl`, confirmed via `git diff HEAD -- docs/REPO_MAP.md` to be
unrelated to this session's edits and present before this work started).

## Total LOC removed

**1861 lines removed, 16 lines added** (net -1845), across 21 files, 2 commits:
- `f8d97fe`: 1501 deletions (package + 6 test files)
- `948c434`: 16 insertions, 360 deletions (integration-point edits across 9 files)

## Blockers

None. The one surprise (stale debate tests) was resolved inline, not left blocking.
