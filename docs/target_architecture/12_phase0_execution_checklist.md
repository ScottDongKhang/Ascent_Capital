# Phase 0 Execution Checklist — Verified Against Current Source

Re-confirmed against the working tree (overnight pass, 2026-08-20) with exact
line numbers, so this is directly actionable without re-deriving anything.

## 1. CLAUDE.md stale claim

- **File**: `CLAUDE.md:12` — reads `"requires calling all four agents
  (\"Required: run_quant_agent for all four agents\")"`.
- **Reality**: `agents/ai_pm_agent.py:972` and `:1290` both read `"Required:
  run_quant_agent for us_equities — it is the only agent whose output feeds
  live..."` — macro/international/alternatives are optional, context-only.
- **Fix**: edit `CLAUDE.md:12`'s sentence to match the current prompt text.
  Zero code risk, one line.

## 2. Audit trail gaps

- **Confirmed existing infrastructure**: `compliance/audit_trail.py` (hash-
  chained, `record_event()`), already imported in `ascent/execution/
  eod_runner.py:37`, called at `:352`.
- **Confirmed gap**: `run_all_agents.py::check_halt_state()` (def at `:360`,
  called at `:1803`) reads/writes `HALT_STATE_PATH = execution/halt_state.json`
  (`:45`) and `HALT_OVERRIDE_PATH = execution/halt_override.json` (`:211`) with
  **no `_audit(...)` call anywhere in that function** — halt and override
  events are print-logged (`:383`, `:1805`) but not written to the hash-chained
  trail.
- **Fix**: add `_audit("halt_triggered", ...)` inside `check_halt_state()`
  where the halt is detected, and `_audit("halt_overridden", ...)` where a
  valid override is found and applied — mirrors the existing
  `_audit("order_submitted", ...)` pattern one-for-one. Needs the same import
  (`from compliance.audit_trail import record_event as _audit`) added to
  `run_all_agents.py` (currently only imported in `eod_runner.py`).

## 3. Duplicate order-submission paths

- **Confirmed**: `def run_eod(...)` at `ascent/execution/eod_runner.py:103`,
  `def run_eod_with_weights(...)` at `:766` — two separate function
  definitions, not one shared with parameters, exactly as `03` described.
- **Fix scope**: extract the shared kill-switch-check → order-loop →
  audit-log sequence into one internal helper both functions call, rather
  than rewriting either entrypoint's external signature (both are called from
  multiple sites in `run_all_agents.py` and shouldn't change shape).

## 4. Silent try/except around regime risk multiplier

- **Confirmed**: `ascent/portfolio/optimizer.py:713-717` —
  ```python
  if regime_signal is not None:
      try:
          from ascent.regime import regime_max_weight
          max_weight = regime_max_weight(max_weight, regime_signal)
      except Exception:
          pass
  ```
  A bare `except Exception: pass` — any import or runtime failure in
  `regime_max_weight` silently no-ops, and the surrounding docstring
  (`optimizer.py:711-712`) still claims "max_weight is tightened based on the
  current regime," which is misleading if the call is failing silently.
- **Fix**: first determine whether `regime_max_weight` currently succeeds or
  is actually failing (add a one-time diagnostic log inside the `except`
  block before deciding). If it's working, narrow the `except Exception` to
  the specific expected failure mode (e.g. `ImportError` only) so a real bug
  doesn't get swallowed. If it's not working, decide explicitly: fix it, or
  remove the regime-tightening call and its docstring claim entirely rather
  than leave code that reads as a safeguard but isn't one.

## Suggested next diagnostic (cheap, before writing any code)

Add a temporary log line inside the `except Exception as e: print(f"[regime_max_weight
failed] {e}")` to observe on the next local run whether this path is silently
failing in practice — this determines whether item 4 is a "fix" or a
"delete," which changes the Phase 0 estimate from S to possibly M if the
underlying regime integration needs real repair.
