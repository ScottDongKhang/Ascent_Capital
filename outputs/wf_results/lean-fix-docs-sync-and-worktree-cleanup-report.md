# Doc-sync and worktree cleanup — report

Session date: 2026-08-16. All work on `main`, real commits, no `git stash`, no live-trading actions.

## Task 1: `ascent/regime/integration.py` false-claim comment

Verified before editing:
- `ascent/regime/__init__.py` exports only `RegimeEngine` — no `regime_max_weight`.
- `git show 3e125c6^:ascent/regime/__init__.py` already lacked the export — the missing export
  predates today's commit; only the comment claiming it was "still-live" is new/false.
- `regime_max_weight()` is defined at `ascent/regime/integration.py:83`.
- Sole call site: `ascent/portfolio/optimizer.py` (`from ascent.regime import
  regime_max_weight` inside a bare `try/except Exception: pass`).
- `.venv/bin/python -c "from ascent.regime import regime_max_weight"` raises `ImportError`
  both **before and after** the comment edit — confirmed identically, so the fix is comment-only.

Corrected the docstring in `ascent/regime/integration.py` to state that `regime_max_weight()`
exists but is unreachable from its only call site because of the missing export, and that the
`ImportError` is silently swallowed. Explicitly left the export/import untouched (that's a
runtime-behavior decision, not a doc fix). `ast.parse` confirmed valid syntax after the edit.

Commit: `63c2eda`

## Task 2: `docs/REPO_MAP.md` stale symbols

Verified `build_factor_constraints`, `regime_scale_weights`, `apply_guardrails` have no live
`def` anywhere in the tree (only deletion-note comments referencing them remain, in
`ascent/portfolio/mvo_optimizer.py`, `ascent/portfolio/optimizer.py`,
`ascent/regime/integration.py`, `ascent/strategy/ai_pm_guardrails.py`). Removed the three from
the `ascent/risk`, `ascent/regime`, `ascent/strategy` symbol-list rows and the two "Change the
regime model" / "Change AI PM authority / guardrails" quick-reference rows, editing each row's
list precisely (other symbols in the same rows kept). Annotated `regime_max_weight` in the
`ascent/regime` row as present-but-unreachable rather than removing it (it still exists).

Commit: `05ee910`

## Task 3: `PROJECT_STATUS.md` stale "dead code" descriptions

The exact line numbers named in the task (~823/826/827) had shifted slightly and the row for
"regime_scale_weights" doesn't literally name that symbol — instead §4.1's `risk_multiplier`
row names `apply_regime_to_portfolio`, which the `ascent/regime/integration.py` docstring
confirms was deleted alongside `regime_scale_weights` and `regime_adjust_sleeve_weights` in the
same 3e125c6 cleanup. Updated three rows in the "4.1 Risk machinery that never executes" table:
- `risk_multiplier` row: now notes `apply_regime_to_portfolio` (+ its two siblings) were
  deleted 2026-08-16, not merely imported-and-uncalled (confirmed the `main.py:420` import
  itself is gone).
- `Factor risk constraints` row: now names `build_factor_constraints` explicitly and says
  deleted 2026-08-16, noting the `factor_constraints` parameter itself stays live for a future
  builder.
- `ai_pm_guardrails.apply_guardrails` row: now says deleted 2026-08-16 along with its private
  helpers and `_LEVEL_CONFIG`, rather than "dead code" still present.

Commit: `4e93028`

## Task 4: `CURRENT_VERIFIED_NUMBERS.md` §5 stale figures

Pulled current values directly from `ascent.reporting.verified_numbers.canonical_wf()` this
session (not hardcoded from the task prompt): Sharpe 0.415, CAGR +10.2%, 165 folds, OOS window
2020-01-02 → 2026-07-15 (1641 days), artifact `wf_report_clean_2026-08-15.json`. Found the
stale lines at `CURRENT_VERIFIED_NUMBERS.md:248` and `:249` (line numbers matched the task
prompt's estimate). Updated both to cite the current §1 numbers/artifact/date instead of the
2026-06-22 / 21-fold figures, and to note that no WFE is reported for the new artifact (the
producing framework doesn't track per-fold in-sample Sharpe).

Ran `.venv/bin/python scripts/verify_docs.py` after Tasks 2–4: **24 passed, 1 failed** — the
only failure is the pre-existing, already-known `repo_map_pointers` gap (`data_cache/
active_alpha_config.json`, `logs/sleeve_ic_log.jsonl` missing on disk), unrelated to this task
and unchanged by it. No new failures introduced.

Commit: `c899ab5`

## Task 5: Worktree cleanup

**`.claude/worktrees/risk-mgmt`** (registered worktree, branch `feature/risk-management`):
- `git log feature/risk-management ^main --oneline` → empty, confirming zero unmerged commits.
- `git -C .claude/worktrees/risk-mgmt status --porcelain -uall` → only `D .DS_Store` and
  `M outputs/debate_log/verdict_2026-04-12.json` (noise, matches the task's description).
- `git worktree remove .claude/worktrees/risk-mgmt` (no force) refused with: `contains modified
  or untracked files, use --force to delete it` — exactly the confirmed-harmless noise files
  causing the refusal, as anticipated. Re-ran with `--force`; succeeded.

**`.claude/worktrees/public-showcase`** (unregistered orphan directory):
- Confirmed not in `git worktree list` output and not present under `.git/worktrees/`.
- Its `.git` file: `gitdir: /Users/scott/Downloads/ascent capital v2 up to phase 5.1/.git/
  worktrees/public-showcase` — that path does not exist (`git -C ... check-ignore` failed with
  `fatal: not a git repository`), confirming it's broken/orphaned as described.
- `git branch --list public-showcase` and `git log public-showcase --oneline -5` confirm the
  branch exists locally with commits `d3422bc`, `abaf1f3` (and earlier history) intact.
- Found `.claude/worktrees/public-showcase/private_prompts.yaml` (4.3KB, gitignored/
  proprietary). Copied it to **`/tmp/public-showcase-private_prompts.yaml.bak`** before removal
  — nothing was silently lost. Not committed anywhere.
- Removed with plain `rm -rf .claude/worktrees/public-showcase` (not `git worktree remove`,
  since it was never a registered worktree).

**Post-cleanup verification**: `git worktree list` now shows only the main worktree
(`/Users/scott/IdeaProjects/ascent-capital [main]`) — `risk-mgmt`'s registration is gone,
`public-showcase` never appeared (as expected, it was never registered). Both branches
(`feature/risk-management`, `public-showcase`) confirmed still present via `git branch --list`.

## Out-of-scope observation (not touched)

`git status` after the cleanup shows `agents/ai_pm_agent.py` modified in the main working tree
(a 177-line deletion, `causal_mechanisms` field and `get_causal_graph` tool removed). This file
was never opened, read, or edited during this session — it predates this session's work and is
unrelated to any of the five tasks here. Flagging it rather than silently leaving it, but not
touching it since it's outside this task's scope and could be someone else's in-progress work.

## Commits (this session, all on `main`)

- `63c2eda` — docs(regime): correct false "still-live" claim about regime_max_weight
- `05ee910` — docs(repo-map): remove stale symbol references deleted by 3e125c6
- `4e93028` — docs(status): mark build_factor_constraints/apply_guardrails/regime_scale_weights as deleted, not dead
- `c899ab5` — docs(numbers): sync §5 stale WF figures to the §1 canonical artifact
- Worktree removals (Task 5) are filesystem/git-metadata operations, not code commits — no
  commit associated with them.
