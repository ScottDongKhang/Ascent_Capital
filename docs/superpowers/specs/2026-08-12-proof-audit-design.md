# Proof Audit — Design Spec

**Sub-project 1 of 4** in the Ascent Capital strip-down/rebuild:
1. **Proof audit** (this spec) — score every live component, produce a keep/cut list
2. Target architecture design — minimal pipeline built only from what survives #1
3. Subagent-driven rebuild — decompose #2, dispatch via `superpowers:subagent-driven-development`
4. Validation & cutover — backtest + paper-shadow new system, then switch

**Context.** The live daily pipeline (`run_all_agents.py`) accumulated components over ~4 months
of plan docs without a mechanical bar for keeping any of them. An audit
(2026-08-12, this session) found several confirmed-dead pieces (event agent, TWAP executor,
self_improve, 13 orphan `ascent/monitoring/` modules, ~460+200+345 LOC) and several live-but-
unmeasured pieces (openbb, MiroFish, causal intelligence — genuinely wired, no cited PnL
contribution). This spec covers only the **audit that produces the keep/cut list** — not the
deletions or the rebuild themselves.

**Live trading is paused** for the duration of this rebuild: `com.ascentcapital.eod.plist` and
`com.ascentcapital.heartbeat.plist` were unloaded via `launchctl unload` on 2026-08-12. Reload
with `launchctl load ~/Library/LaunchAgents/com.ascentcapital.eod.plist` (+ heartbeat) to resume.
MiroFish and LiteLLM proxy launchd jobs were left running (no order-submission path).

---

## Architecture

Two independent scorers, one output artifact. Both are new code — the existing
`ascent/research/walk_forward_runner.py` / `wf_framework/` is deliberately **not** reused here,
since the point of this audit is not to inherit trust from machinery that hasn't itself been
re-verified.

- **Path A — walk-forward IC/Sharpe scorer.** For each alpha sleeve (from
  `ascent/alpha/stack.py::DEFAULT_ALPHA_WEIGHTS`) and each of the 4 specialist agents
  (`agents/us_equities_agent.py`, `macro_agent.py`, `international_agent.py`,
  `alternatives_agent.py`), compute out-of-sample IC and Sharpe per fold. Universe per fold comes
  from `get_universe_on_date()` — no look-ahead (integrity constraint #1 in `CLAUDE.md`).
- **Path B — counterfactual return-delta scorer.** For non-sleeve subsystems (regime overlay,
  hedge overlay, earned-authority ladder, debate judge intervention), compare with-component vs.
  without-component returns. Reuses the five-track counterfactual data via
  `ascent/analyst/catalog/registry.py` (tracks `astar`/`a`/`b`/`c`/`d`) where an existing track
  already isolates the component; builds a new synthetic track where one doesn't exist yet
  (e.g. hedge overlay has no existing track).
- **Output.** Both paths write rows into one scorecard: `outputs/analyst/proof_audit_<date>.json`.
  One row per component: `name`, `method` (`wf_ic` | `counterfactual`), `metric` (IC/Sharpe or
  return delta), `p_value`, `sample_size`, `verdict`.

## Components

The component list is a **pinned fixture**, not dynamically discovered — an audit that silently
skips a component because it wasn't detected is worse than no audit. Covers:

- Every key in `ascent/alpha/stack.py::DEFAULT_ALPHA_WEIGHTS` (Path A)
- The 4 specialist agents by module name (Path A)
- Named subsystems: regime overlay, hedge overlay, earned-authority ladder, debate judge
  intervention (Path B)

Adding a component later means editing the fixture explicitly, not relying on discovery.

## Verdict rule

Three-way, never silently defaults:

- **KEEP** — statistically significant positive (p < 0.05, meets minimum sample threshold)
- **CUT** — statistically significant negative, or indistinguishable from zero
- **INSUFFICIENT_DATA** — sample too small to call either way (e.g. earned-authority's empty
  promotion buffers — this must not be reported as CUT just because the buffer is empty; it's a
  separate known issue tracked in `docs/superpowers/plans/2026-07-31-phase0-naming-and-liveness.md`)

## Testing & error handling

- Golden-path tests use **synthetic fixture data with a planted, known correlation** — this pins
  the IC/delta math itself, not any claim about real market behavior.
- A real-data run against the live artifacts is a separate manual step; it is not asserted to
  produce a specific outcome in a unit test.
- Missing or corrupt source data for one component fails **only that component's row** as
  `INSUFFICIENT_DATA` — one bad input never crashes the whole audit run.

## Explicitly out of scope

- Deleting any code (sub-project 3's job, after the target architecture in sub-project 2)
- Reusing or fixing `ascent/research/walk_forward_runner.py` — Path A is fresh, minimal, standalone
- Scoring debate mechanics beyond the judge's single bounded position-change (integrity
  constraint #5 stays intact regardless of audit outcome)
- Resuming the launchd scheduler — stays paused until sub-project 4 (cutover)
