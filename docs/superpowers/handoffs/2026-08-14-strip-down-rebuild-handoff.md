# HANDOFF PROMPT — copy everything below the line into the new conversation

---

I'm continuing a multi-session effort on `~/IdeaProjects/ascent-capital` (a live, currently-paused
paper-trading quant platform): stripping the pipeline down to only what's proven to drive
PnL, then rebuilding it. A previous session got through 4.5 sub-projects. Don't restart —
continue from where it stopped.

## How I want you to work

- **Use as many subagents as possible.** Dispatch investigation, implementation, and review to
  subagents rather than doing deep exploration or long file reads in the main thread. Keep my
  (the controller's) context clean — that's the point of delegating.
- **Run overnight, autonomously. Don't ask for permission** on individual steps — git commits,
  worktree creation, running tests, deleting/regenerating stale data files, dispatching more
  subagents. The one exception, carried over from the prior session and not yet lifted: **do not
  reload `com.ascentcapital.eod`/`.heartbeat` (resume live trading) until the open bug below is
  actually fixed and a walk-forward validation run produces plausible numbers.** That's not a
  permission gate on everything — it's a specific, evidence-based bar for one specific action.
- Use `superpowers:subagent-driven-development` for anything that's a real implementation task
  (spec → plan → dispatch implementer → task review → fix loop → final whole-branch review →
  merge). This project has an established rhythm for this — follow it, don't shortcut it. Every
  sub-project so far went through this and it caught real bugs at nearly every stage, including
  bugs that individual task reviews missed and only a final whole-branch review found.
- I mentioned wanting "a GitHub skill to reduce tokens" — I didn't find one under that name in
  the installed skills list this session. If one exists under a different name, use it. Otherwise
  just use `gh` via Bash as normal; nothing in this effort has needed GitHub/PRs yet (everything
  has been local commits to `main`, no pushes).

## Read these first (don't re-derive what's already written down)

- `CHECKPOINTS.md` (repo root) — 5 tagged checkpoints (`checkpoint-1` through `checkpoint-5`),
  what each one did, and a live "BLOCKER" section describing the current open problem in detail.
  **Read this file's BLOCKER section before anything else.**
- Memory (if your harness has access to this project's memory store) — search for
  `strip-down-rebuild-status`, `wf-harness-bad-results-unexplained`,
  `alpha-weights-runtime-override-not-fixed`, `falsifier-trim-was-unmeasured-live-write`,
  `save-parquet-wide-cache-corruption`. These are the durable record of what was found and
  fixed, and — critically — what's still open. If you don't have memory access, the equivalent
  information is in the specs/plans/reports listed below.
- `CLAUDE.md` — durable project rules. Constraints #5 and #6 were rewritten this session to
  reflect the rebuild; read the current version, not what you might remember from training.
- `CURRENT_VERIFIED_NUMBERS.md` — the only citable source for any performance figure. This
  project has published a synthetic/wrong number as fact before; don't repeat that.

## What's already done (4.5 sub-projects, all merged to `main`, all tagged)

1. **Proof audit** (`checkpoint-1-proof-audit`, `193f445`) — built a standalone scorer
   (`ascent/analyst/proof_audit/`) measuring every alpha sleeve/agent/subsystem's real
   walk-forward IC/Sharpe or counterfactual return-delta.
2. **Extension** (`checkpoint-2`, `3e824e7`) — wired in missing data sources, fixed a `prices_live`
   timezone bug and a duplicate-row bug found along the way.
3. **Production bugfixes** (`checkpoint-3`, `db0890a`) — fixed a `signal_date` normalization bug
   and a real, live-critical `save_parquet` bug that was silently corrupting 3 specialist agents'
   price caches on every save (`prices_macro`/`international`/`alternatives`).
4. **Cache repair + final scorecard** (`checkpoint-4`, `1cb932c`) — repaired the 3 corrupted
   caches via a real yfinance re-fetch. Final proof-audit scorecard:
   `outputs/analyst/proof_audit_2026-08-13.json` — **KEEP=2** (`meanrev`, `statarb`),
   **CUT=12**, **INSUFFICIENT_DATA=9**. Only 2 of 23 components ever showed a statistically
   significant positive signal.
5. **Target architecture rebuild** (`checkpoint-5-target-architecture`, `19c3240`) — alpha stack
   reduced from 15 sleeves to those 2 (`meanrev`+`statarb`, 50/50); regime overlay and hedge
   overlay removed (both scored CUT); only `us_equities_agent` allocates live capital now
   (`macro`/`international`/`alternatives` agents excluded — code intact, just not invoked
   daily); every unproven live-write path made advisory-only — the debate judge's
   position-change, the AI PM's earned-authority blend, and (found only by that sub-project's
   own final whole-branch review, not by any task-scoped review) **the falsifier trim** — a
   real order-submitting path built entirely on unmeasured AI PM output that nearly shipped
   untouched. All three now detect, log, and record for future measurement, but don't touch
   live capital.

Every one of the above went through the full brainstorm → spec → plan → subagent-driven
implementation → task review → fix loop → final whole-branch review cycle. Several rounds
caught real bugs (a stale `shadow_promoter._SLEEVE_FLOORS` that would have re-injected cut
sleeves; the falsifier trim; two orchestrator risk gates that degenerated with only 1 agent
present). This is the process to keep using, not a one-time formality.

## What's open — THIS IS THE ACTUAL NEXT STEP

**Sub-project "4b"** (partial): a real walk-forward validation run of the rebuilt pipeline
produced implausible results (Sharpe ≈ -0.3 to -0.43, Hit Rate ≈ 4.6-4.7% over 3291 trading
days — not a real strategy's signature, a broken one). Two things happened:

1. **Found and fixed a real bug** (commits `6628948`, `31d49ee`, merged): two stale, 3-month-old
   pre-rebuild files (`data_cache/active_alpha_config.json`, `logs/sleeve_ic_log.jsonl`) were
   overriding `DEFAULT_ALPHA_WEIGHTS` at runtime — in both the backtest AND the live pipeline
   (same function, `ascent/alpha/stack.py::_load_active_alpha_weights()`). Deleted both files,
   fixed a related redistribution bug in `_get_gated_weights()`. This fix is real, independently
   verified (a final review hand-derived the math against 6 examples), and should stay — but:
2. **Re-running validation after the fix showed nearly identical bad numbers.** The fix was
   correct but was NOT the cause of the bad backtest. Full writeup, including exactly why the
   original diagnosis was incomplete (a log line I misread as evidence — `[alpha_stack]
   loaded=[...]` lists every sleeve that *computed*, not every sleeve that's *weighted* — the
   actual blend math was probably already correct even before the fix):
   `outputs/wf_results/vc-task-3-revalidation-report.md`

**The real root cause is still unknown.** One consistent, unexplained anomaly across both runs
(before and after the fix): every one of 327 `[WF] targets injected, ..., valid rows: 0` log
lines shows zero valid rows — `ascent/research/walk_forward_runner.py:349`'s per-fold training
target matrix is entirely NaN, every single fold, every single run. This feeds the `ml` sleeve
(not part of the current 2-sleeve stack either way), so it's not obviously the direct cause —
but it's the one real red flag that's persisted, and I haven't traced it further.

**Your job, if you want a clear starting point:** investigate the walk-forward harness itself —
`ascent/research/walk_forward_runner.py` and whatever `ascent.backtest.engine.BacktestEngine`
does with the combined OOS weights — to find why a 2-sleeve `meanrev`/`statarb` book produces a
4.7% hit rate. Candidates worth checking first: the return/hit-rate calculation itself, how 330
rebalance-date weights get forward-filled into 3291 daily rows, and the zero-valid-target-rows
anomaly (even though it's nominally unrelated, an entirely-broken code path running silently
across every fold in a harness is worth understanding before trusting anything else it produces).

Real run logs to work from (don't re-run from scratch until you have a hypothesis — a full run
takes several minutes):
- `outputs/wf_results/wf_run_target_architecture_2026-08-14_BROKEN.log` (before the alpha-weight fix)
- `outputs/wf_results/wf_run_target_architecture_2026-08-14_fixed_still_broken.log` (after)

## Known non-blocking follow-ups (parked, not urgent, listed in `CHECKPOINTS.md`)

- `_apply_falsifier_trim`'s suspension gate now blocks its own measurement record instead of a
  trade — self-sealing if it ever reaches 30 scored outcomes. Dormant today.
- Discovery mini-rebalance path: a `portfolio_state` variable referenced outside the `try` that
  defines it — fails safe today, one-line fix (`portfolio_state = None` init) worth landing.
- A couple of stale-comment nits (`run_all_agents.py:1249`, a file-attribution error in a
  CLAUDE.md gotcha).
- `ascent/research/wf_framework/ascent_strategy.py::_make_alpha_weights` (a DIFFERENT walk-forward
  framework, `scripts/run_ascent_wf.py`'s default path — NOT the one used for validation this
  session) force-injects `trend` at 30-50% and bypasses the IC gate entirely. Real bug, but
  outside this session's validated code path — the design spec for sub-project 4
  (`docs/superpowers/specs/2026-08-14-validation-cutover-design.md`) explicitly chose
  `walk_forward_runner.py` over this framework for exactly this class of reason. Worth fixing
  eventually if anyone ever uses `run_ascent_wf.py`'s default path.

## Non-obvious gotchas from this session, worth knowing before you hit them again

- **Never run `git stash`/`git stash pop` in this repo.** The stash stack is shared across
  worktrees and sessions. This bit a subagent twice this session (unrelated pre-existing WIP —
  a Sortino-ratio fix in `ascent/research/wf_framework/metrics.py` — got conflict-marked and had
  to be recovered via `git checkout HEAD -- <file>`, with the stash left intact). Tell every
  implementer subagent explicitly not to use it.
- Worktrees created fresh from `origin/main` (the default `baseRef`) miss same-day local commits
  not yet pushed — always `git rebase main` right after creating one, before doing anything else.
- An unrelated automated job ("chore: update performance dashboard") periodically commits to
  whatever branch/worktree is active, touching only `README.md`/`docs/index.html`. Harmless,
  ignore it, don't let a reviewer flag it as part of your diff.
- `.venv` shows as a tracked-but-broken symlink target in fresh worktrees (a handful of stub
  files are committed for some reason) — `rm -rf .venv && ln -s /Users/scott/IdeaProjects/ascent-capital/.venv .venv`
  in every new worktree before running anything, and `git checkout -- .venv` to restore the
  tracked stubs before merging (don't commit the symlink).
- Real data (`data_cache/`, `logs/`) is gitignored and worktree-local — symlink individual files
  in from the main checkout when a task needs real data (`prices_live.parquet`,
  `counterfactual_daily.jsonl`, etc.), or do data-touching work directly on `main` instead of in
  a worktree (this session did both, depending on whether the task also needed code review).
- Subagents sometimes launch a long-running command in their own background process and then
  their turn ends before it completes — you won't get the result. If a dispatched agent's final
  reply is vague about a long-running step ("I'll wait for the monitor..."), check
  `ps aux` for the actual process rather than trusting the agent finished; resume or take over
  directly if it's still running or silently died.
