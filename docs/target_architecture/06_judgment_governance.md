# Judgment & Governance Layer — Target Architecture

## Layer 1 — Department Mandate

**Purpose**: govern the boundary between the systematic book (alpha → portfolio →
walk-forward → `us_equities_agent`) and discretionary/AI judgment (`debate/`,
`agents/ai_pm_agent.py`, `agents/red_team_agent.py`), so that judgment's *live*
authority is always proportional to a measured track record — never to confidence,
seniority of model, or elapsed time alone.

**Governing principles**:
1. **Authority is earned, never granted.** Every mechanism that can move live
   weights starts at zero authority (`Shadow`, level 0, `ai_weight=0.0` — see
   `ascent/strategy/earned_authority.py:93-104`) and climbs only via
   `PROMOTION_CONFIG` gates measured against realized returns, never via a one-time
   human sign-off.
2. **Trust is asymmetric.** Promotion is slow (multi-week windows, multiple
   simultaneous gates); demotion is fast (a single day, `_apply_demotion`, can
   zero authority — see the catastrophic path at `earned_authority.py:201-207`).
   This mirrors integrity constraint #5's own philosophy: unmeasured or unproven
   mechanisms stay advisory-only until proven, and one bad result reverses the
   decision immediately.
3. **Every override is logged as a counterfactual, whether or not it executes.**
   `record_intervention(..., applied=False)`
   (`debate/adversarial_authority.py:128-182`) is the audit primitive: it makes
   "what would have happened if we'd trusted this call" a first-class, scoreable
   artifact instead of a lost opinion. This is what lets authority be revoked with
   evidence rather than vibes.
4. **The only binary, unstaged authority left is a circuit breaker, not a PM.**
   `halt_and_review` and `reduce_size` (`debate/judge.py:6,90-92`) are
   fund-halt/de-gross powers — they can only make the book *smaller*, never pick
   names, never grow. They are deliberately exempt from the staged-authority
   ladder because they are risk controls, not investment decisions.
5. **Reversibility is structural, not procedural.** Nothing judgment does writes
   to `active_alpha_config.json` or portfolio weights directly; it always passes
   through a state machine (`earned_authority.json`, `adversarial_authority.json`)
   that can be rolled back to level 0 in one write.

## Layer 2 — Roles

| Role | Current code | Function |
|---|---|---|
| **Debate Moderator / Judge** | `debate/judge.py::run_judge` | Synthesizes bull/bear debate into `proceed \| reduce_size \| halt_and_review`. Only entity with any *live* write path today (de-gross / halt). |
| **Authority Ladder Manager** | `ascent/strategy/earned_authority.py` | Owns the 6-level ladder (`Shadow->CEO`), the promotion/demotion state machine, and `ai_weight`. Currently demotion-complete, promotion-broken (buffers empty — see below). |
| **Red Team / Adversarial Reviewer** | `agents/red_team_agent.py::run_red_team` | Stress-tests the AI PM's own thesis from inside Phase 2 synthesis (`ai_pm_agent.py:2632-2634`). Advisory critique, not a separate authority track today. |
| **Outcome Tracker** | `debate/outcome_tracker.py`, `debate/adversarial_authority.py::score_pending_interventions` | Scores past interventions against realized 10-trading-day price moves; feeds win-rate back into per-type authority (`_MAX_CHANGE_PCTS`, `_rebuild_authority`). |
| **Escalation Authority** | Currently: `run_judge`'s binary verdict only. **Proposed**: a new `EscalationAuthority` role that adjudicates disagreement between the systematic book and judgment at graduated severities, not just halt/reduce. |

**New role needed**: **Authority Ladder Manager** must gain a **Promotion Auditor**
sub-responsibility — someone/something whose only job is asking "why hasn't a
promotion gate fired in N cycles" and distinguishing "correctly never earned it"
from "structurally can't fire" (buffers were empty after 19 update cycles — a
promotion-path bug, not a performance verdict).

## Layer 3 — Per-Role Responsibilities and Decision Logic

### 3.1 Authority Ladder Manager (`earned_authority.py`)

Ground truth today: `LEVEL_WEIGHTS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]`,
`PROMOTION_CONFIG` already specifies real gates per transition — e.g. (1→2): 21-day
window, Sortino edge > 0.20 vs Track A★, hit-rate ≥ 52%, profit-factor > 1.2, ≥5
decisions, fade-rate ≤ 30%, regime-gate pass — **all six must be true
simultaneously** (`all(gates.values())`, line 243). Demotion is 3-tier and
asymmetric: catastrophic (single day ≤ -9.9pp vs A★) drops straight to level 0;
hard (single day ≤ -5pp) drops one level; soft (21-day drawdown gap > 3pp) drops
one level — all with a 5-day cooldown (`_apply_demotion`, lines 205, 212, 222).

**What's broken**: "buffers empty after 19 update cycles, decision log 8x
duplicated; promotion never evaluated, only demotion can fire." `update_authority()`
requires **both** `track_d_return` and `track_astar_return` to be non-`None` on the
same call (line 166) to append to the buffer at all; if the upstream caller (the
daily pipeline / `run_all_agents.py`) doesn't reliably supply both every cycle, the
buffer never reaches `PROMOTION_CONFIG[(level, level+1)]["window"]` (21/21/42/63
days), so `len(d_buf) >= win` at line 230 never becomes true and promotion code is
dead even though the logic is correct. Demotion, by contrast, only needs `level >
0` and one bad day, so it alone stays reachable — hence "demotion-only" in
practice, not by design.

**Fix, concretely**: instrument `update_authority()` call sites (the daily
rebalance path) to assert both returns are populated every trading day the buffer
is meant to accrue, and add an explicit **stuck-buffer alarm**: if `len(d_buf) <
win` for `days_stuck >= 2 * win` (i.e., double the required window has elapsed with
no promotion decision reachable), log a `PROMOTION_PATH_STALLED` event distinct
from `is_stuck` (which today conflates "no promotion earned" with "promotion
mechanically impossible" — see `is_stuck()` at line 55, which only checks 63 days
at level with no cause diagnosis).

### 3.2 Debate Moderator / Judge (`debate/judge.py`)

Responsibility: convert bull/bear debate output into one of three verdicts.
Decision logic today is a single LLM call with a hard-coded action space
(`proceed | reduce_size | halt_and_review`, line 90-92); `halt_and_review` is
reserved for "catastrophic risk (systemic event, fund-level error)" only — i.e. it
is intentionally rare and binary, not gated by authority level because it is a
circuit breaker, not a discretionary trade.

**Proposed staged addition**: gate everything the judge recommends *besides*
halt/reduce (e.g., a hypothetical future single-position trim) behind the Authority
Ladder Manager's current `ai_weight` and the per-intervention-type authority in
`adversarial_authority.py`. Concretely: a judge recommendation to change one
position's size by up to X% requires (a) ladder level ≥ 2 (`ai_weight ≥ 0.15`)
**and** (b) that intervention type's `allowed_change_pct` from
`get_authority(intervention_type)` (`adversarial_authority.py:113-125`) is
non-zero and non-suspended. A sleeve-wide override requires ladder level ≥ 4
(`Director`, `ai_weight=0.50`) — i.e., a materially longer, harder-earned track
record, matching the spec's "Q consecutive rebalances" requirement via
`PROMOTION_CONFIG[(3,4)]["window"]=42` and `[(4,5)]["window"]=63`.

### 3.3 Red Team / Adversarial Reviewer (`agents/red_team_agent.py`)

Today: called only from inside `run_ai_pm`'s Phase 2 (`ai_pm_agent.py:2632`), a
single critique pass with no independent scoring track. **Proposed**: give
red-team verdicts their own `intervention_type` bucket in `adversarial_authority.
py`'s `_ALL_TYPES` (e.g., `"red_team_veto"`), scored the same way as
`adversarial_thesis`/`regime_sizing` via `score_pending_interventions()`, so a
red-team override earns authority independently rather than being permanently
subordinate to whatever the AI PM decides in the same call.

### 3.4 Outcome Tracker (`adversarial_authority.py::score_pending_interventions`,
`debate/outcome_tracker.py`)

Decision logic already concrete and correct in shape: 14-calendar-day lookback
(`OUTCOME_WINDOW_DAYS=10` trading-day target, line 31), correctness = "trimmed
symbol underperformed SPY by >1pp" (lines 250-259), win-rate tiers
`MIN_SAMPLE_SUSPEND=30`: `<40% -> suspended`, `50-70% -> medium (2%)`, `>70% ->
high (4%)`, `<30 scored -> low (1%)` (lines 33-45, 298-312). This is the
*per-intervention-type* authority ladder, parallel to but structurally separate
from the *AI-PM-wide* ladder in `earned_authority.py`. **Gap**: these two ladders
are never cross-checked — a judge could theoretically be "high" authority on
`regime_sizing` while the AI PM as a whole is still `Shadow` level. Layer 4
formalizes the required consumption order to close this.

### 3.5 Escalation Authority (new)

When the systematic book and judgment disagree beyond what
`reduce_size`/`halt_and_review` cover, escalation should route to a
**severity-graded** response, not silence:
- **Severity 1** (single position, judgment's authorized change_pct covers it):
  apply automatically, log via `record_intervention(applied=True)`.
- **Severity 2** (exceeds authorized change_pct, or ladder level too low):
  auto-downgrade to `record_intervention(applied=False)` — proposal only, exactly
  today's behavior for position-change/blend/falsifier-trim.
- **Severity 3** (judge itself calls `halt_and_review`): existing binary halt,
  unstaged. **Correction (adversarial verification pass, cross-checked against
  `debate_runner.py:400-411` and `run_all_agents.py::check_halt_state()`
  lines 371-402): this already *does* require human unlock.** The halt write
  sets `requires_override: True`, and `check_halt_state()` will not clear the
  halt unless a human manually creates `execution/halt_override.json` with a
  valid `override_date` — matching `00_institutional_audit.md`'s correct
  characterization of the same code. An earlier draft of this document stated
  the opposite; that was a factual error in the document, not a gap in the
  code. No change is needed to the halt mechanism itself — it already is the
  one place judgment's authority is hard-capped by a human checkpoint.

## Layer 4 — Interfaces / Data Contracts

```
record_intervention(date, symbol, intervention_type, from_weight, to_weight,
                     prediction, regime, applied: bool)
    -> appends logs/adversarial_interventions.jsonl
    -> increments data_cache/adversarial_authority.json[type].n_interventions

score_pending_interventions()
    reads:  logs/adversarial_interventions.jsonl (rows >=14 cal days old, unmeasured)
            yfinance prices for {symbols} U {SPY}
    emits:  per-row {outcome_measured, outcome_correct, symbol_10d_return, spy_10d_return}
            -> _rebuild_authority() writes data_cache/adversarial_authority.json
              {type: {n_scored, win_rate, allowed_change_pct, suspended}}
    consumed by: debate/judge.py via format_authority_for_judge(regime) (injected into judge prompt)
                 agents/ai_pm_agent.py (indirectly, via judge verdict)

update_authority(track_d_return, track_astar_return, n_decisions_evaluated,
                  hit_rate, profit_factor, fade_rate, regime_gate_pass)
    reads:  data_cache/earned_authority.json (state), rolling 63-day D/A* buffers
    emits:  state{level, title, ai_weight, days_at_level, in_cooldown, cooldown_until}
            -> data_cache/earned_authority.json
    consumed by: run_all_agents.py daily pipeline (gates how much AI PM output can move weights)
                 debate/judge.py (should read ai_weight as a ceiling on any staged authority,
                                   not just adversarial_authority.py's per-type win rate -- GAP today)

rebuild_buffers_from_counterfactual()
    reads:  ascent.monitoring.ai_pm_counterfactual.load_daily_records()
    emits:  reconciled track_d_returns / track_astar_returns buffers
    role:   Promotion Auditor's repair tool -- run when buffers are suspected stale/empty
```

**New contract to add**: `AuthorityLadderManager.get_gated_authority(
intervention_type: str) -> {max_change_pct: float, requires_escalation_above:
float}` — a single function that intersects `earned_authority.get_state()
["ai_weight"]` (macro ceiling) with `adversarial_authority.get_authority(
intervention_type)["allowed_change_pct"]` (micro ceiling per call type), so
`debate/judge.py` has one call site instead of reading two disjoint JSON files
with no reconciliation logic between them (today's actual gap).

## Layer 5 — Concrete Implementation Mapping

**What's real today**: `debate/judge.py::run_judge` (binary halt/reduce, live);
`debate/adversarial_authority.py::record_intervention` + `score_pending_
interventions` (fully functional logging + scoring loop, all advisory);
`ascent/strategy/earned_authority.py::update_authority` (demotion fully reachable,
three-tier, asymmetric).

**What's broken**: promotion in `earned_authority.py` is logically complete
(`PROMOTION_CONFIG`, `all(gates.values())`, lines 227-263) but practically dead
because the buffer-population precondition at line 166 (`track_d_return is None or
track_astar_return is None -> skip`) isn't reliably satisfied by upstream callers —
this is a wiring bug in the daily pipeline, not the ladder module.

**Exact changes needed**:
1. In whatever calls `update_authority()` inside `run_all_agents.py` / the EOD
   runner — audit that both `track_d_return` and `track_astar_return` are
   computed and passed every trading day, not just on days both happen to be
   available. Add a `PROMOTION_PATH_STALLED` warning distinct from `is_stuck()`.
2. Add `AuthorityLadderManager.get_gated_authority()` (new function in
   `earned_authority.py`) implementing the Layer 4 contract, and call it from
   `debate/judge.py::run_judge` before line 90's verdict options are constructed,
   so any future non-binary judge action is capped by the intersection of both
   ladders.
3. Add `"red_team_veto"` to `_ALL_TYPES` in `adversarial_authority.py` (line 45)
   and a `record_intervention` call site inside `agents/red_team_agent.py::
   run_red_team`, so red-team critiques accrue their own scored track record
   instead of being folded silently into the AI PM's synthesis.
4. Gate point for any future live-write reinstatement (position-change,
   earned-authority blend, falsifier trim): the check belongs immediately before
   the (currently-deleted) live-write call, and must read
   `get_gated_authority(intervention_type).max_change_pct > 0` — reinstatement
   per integrity constraint #5 requires this gate to exist and be exercised in
   dry-run/advisory mode for a full promotion window (per `PROMOTION_CONFIG`)
   before any `dry_run=False` path is restored.
