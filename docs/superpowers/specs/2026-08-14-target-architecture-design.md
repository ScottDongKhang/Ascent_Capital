# Target Architecture — Design Spec

**Sub-project 2** of the strip-down/rebuild. Redesigns the live (currently paused) trading
pipeline around what sub-projects 1/1b/1c actually proved: 2 of 23 components
(`meanrev`, `statarb`) show a statistically significant positive signal. Everything else is
either measured-negative (`CUT`) or unmeasured (`INSUFFICIENT_DATA`, excluded here per an
explicit decision to revisit individually once each clears real measurement, not to guess).

**Scope, per this session's decisions:** full pipeline — alpha, portfolio construction, agent
allocation, and the AI PM/debate write paths. Not just the alpha layer.

## 1. Alpha stack: 2 sleeves only

`ascent/alpha/stack.py::DEFAULT_ALPHA_WEIGHTS` (currently 15 keys) reduces to:
```python
DEFAULT_ALPHA_WEIGHTS = {
    "meanrev": 0.50,
    "statarb": 0.50,
}
```
Equal weight — both cleared significance with comparable, not clearly differentiated, IC across
audit runs (`meanrev` 0.0229, `statarb` 0.0132-0.0243 across runs). No basis yet for a
non-equal split; revisit once more OOS evidence accumulates.

Per CLAUDE.md constraint #6, `ascent/research/self_improve.py`'s parallel
`DEFAULT_ALPHA_WEIGHTS` must match this key set exactly — update both. `self_improve.py`'s
`MIN_SLEEVE_WEIGHTS` floors several now-dropped sleeves (`trend`, `earnings`, `analyst`,
`options_flow`, `insider`, `short_interest`) — prune those entries too, they'd otherwise silently
re-introduce a floor for a sleeve with zero weight.

`DEFAULT_ALPHA_WEIGHTS_BY_REGIME` (`stack.py`, 5 regime variants) becomes dead code once §2
removes regime-conditional weight adjustment — delete it, not just leave it unreferenced.

## 2. Portfolio construction: drop regime and hedge overlays

- **Regime-conditional weight adjustment**: `regime_adjust_sleeve_weights`
  (`ascent/regime/integration.py`), called once from `build_alpha_stack()`
  (`ascent/alpha/stack.py`, gated `if regime_signal is not None:`). Delete that call and the
  regime-label resolution feeding it — `stack.py` goes back to always using the flat
  `DEFAULT_ALPHA_WEIGHTS`, no regime branch. *Not* touching `apply_regime_to_portfolio`
  (`ascent/regime/integration.py`) — that's a different function, called once from
  `ascent/main.py`'s **backtest** path, out of scope for a live-pipeline simplification.
- **Hedge overlay**: `apply_hedge_overlay` (`ascent/portfolio/hedge_overlay.py`), one call site
  in `run_all_agents.py` (~Step 5b, right after `run_orchestrator()`). Delete that call and its
  log-append. `regime_overlay` and `hedge_overlay` both scored `CUT` (no proven value) in the
  audit — same bar as the alpha sleeves.

## 3. Agent allocation: only `us_equities_agent` gets live capital

`orchestrator/central_intelligence.py`'s allocation is config-driven and already degrades
gracefully when an agent produces no output (`_compute_allocation()` only builds entries for
agents actually present; `merge_agent_outputs()` defaults an absent agent to 0.0 weight) — so
excluding `macro_agent`/`international_agent`/`alternatives_agent` from live capital does not
require restructuring the orchestrator's merge logic. It requires **not invoking those three
agents in the daily run** (`run_all_agents.py`'s orchestration step — exact call sites TBD by
implementer investigation, matching this session's established pattern of investigating before
locking a mechanism). Their code stays in the repo, unrun, not deleted — same treatment as the
9 unmeasured components. `BASE_ALLOCATION`/`STRESSED_ALLOCATION`/`CRISIS_ALLOCATION` dicts and
`_apply_crisis_veto()` in `central_intelligence.py` become inert with one agent present; simplify
them rather than leave dead branches, but do not restructure `central_intelligence.py`'s core
merge function beyond that — it already handles this case correctly.

**Why:** `macro_agent`/`international_agent` both scored `CUT` on their real universes (not
significant). `alternatives_agent` is still `INSUFFICIENT_DATA` for reasons unrelated to the
now-fixed cache bug (a distinct, unexplained signal-density issue) — excluded per the "revisit
unmeasured components individually" decision, not because it's proven negative.

## 4. AI PM / debate: remove both write paths whose measured effect scored CUT, keep advisory

The proof audit measured two **distinct** live write paths, both scoring `CUT`:

- **`earned_authority`** (p=0.35, `track_d` vs `track_astar`) measures
  `ascent/strategy/earned_authority.py`'s blend of the AI PM's own portfolio into
  `merged_weights` (`authority_blend()`/`blend()`, called from `run_all_agents.py`'s
  `update_authority(...)` and blend call sites) — a *different* mechanism from the judge's
  position-change path, easy to conflate by name. This write path's measured value-add is CUT.
- **`debate_judge_intervention`** (p=0.75, `track_b` vs `track_d`) measures the debate judge's
  bounded position-change (`apply_judge_position_change`, two call sites in `run_all_agents.py`:
  the scheduled-rebalance path and the discovery/mini-rebalance path — CLAUDE.md constraint #5),
  gated by `debate/adversarial_authority.py`'s separate authority ladder. This write path's
  measured value-add is also CUT.

**Applying the same "only keep what's proven" bar consistently:** remove both live write paths.
- Delete `run_all_agents.py`'s two `apply_judge_position_change` call sites.
- Delete `run_all_agents.py`'s AI PM blend-into-`merged_weights` call site
  (`authority_blend`/`blend()`), and the `update_authority(...)` call that feeds its ladder.
- **Do not delete** `debate/adversarial_authority.py`, `ascent/strategy/earned_authority.py`,
  `debate_runner`/`debate/agents.py`/`debate/judge.py`, or `agents/ai_pm_agent.py` themselves —
  the debate/AI-PM analysis layer keeps running, keeps producing verdicts and reasoning, keeps
  logging to `outputs/debate_log/` and `logs/ai_pm_decision_log.jsonl`, keeps being read as
  context by later pipeline stages. Only the two *write* call sites are removed. This preserves
  continued counterfactual measurement (both `earned_authority` and `debate_judge_intervention`
  tracks keep accumulating real evidence) without keeping an unproven write path live.
- `record_intervention()`/scoring calls (`run_all_agents.py`, `weekend_runner.py`) that feed
  `debate/adversarial_authority.py`'s ladder become reporting-only once nothing calls
  `apply_judge_position_change` — leave them running; they cost nothing and preserve the
  measurement trail.

**Why both, not just the judge's:** the evidence bar the user set at the start of this whole
effort was "positive IC/Sharpe in walk-forward OOS" for sleeves/agents and counterfactual return
delta for subsystems — applied mechanically to 21 of 23 components already. Treating the AI PM
blend differently from the judge write path, when both cleared the identical evidence bar with
the identical (CUT) result, would be an inconsistency this spec should not introduce silently.

**CLAUDE.md updates required:** constraint #5 (the debate write-path exception) becomes moot —
rewrite to state debate is now advisory-only with *no* live-write exception. Constraint #6
(two-file `DEFAULT_ALPHA_WEIGHTS` key-set match) stays as a rule, just now enforced on a 2-key
set instead of 15.

## Explicitly out of scope

- Deleting any of the 9 `INSUFFICIENT_DATA` components' code, or the agents/subsystems this spec
  excludes from live capital/write paths — all stay in the repo, unrun/unwired, not removed.
- `ascent/main.py`'s backtest-path `apply_regime_to_portfolio` call.
- Any change to `EVENT_TRADING_ENABLED`/`TWAP_ENABLED`/`SELF_MODIFY_ENABLED`/
  `LONG_SHORT_ENABLED` kill switches (already `False`, unrelated to this evidence).
- Resuming `com.ascentcapital.eod`/`.heartbeat` — cutover (sub-project 4) decides that, after
  this design is implemented and validated (sub-project 3).
- Re-splitting `meanrev`/`statarb`'s 50/50 weight based on IC magnitude — flagged as a future
  refinement once more OOS evidence exists, not decided here.
