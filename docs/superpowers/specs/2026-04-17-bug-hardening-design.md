# Bug Hardening — Pre-Phase-2 Baseline
**Date:** 2026-04-17  
**Approach:** Single hardening branch, fixes grouped by layer, merge as one clean baseline commit.

---

## Scope

11 bugs found in a systematic audit of the live execution, debate, and runner layers. All are guard clauses, index checks, or log improvements — no architectural changes.

---

## Layer 1: Execution (`ascent/execution/eod_runner.py`)

### E1 — Empty weights after pipeline filter (Critical)
**Location:** ~line 65, `run_eod_with_weights()`  
**Bug:** `valid_rows = target_weights_all[(target_weights_all > 0).any(axis=1)]` followed immediately by `valid_rows.index[-1]` with no empty check. If the pipeline returns all-zero weights, `valid_rows` is empty and `index[-1]` raises `IndexError` — crash with no useful message.  
**Fix:** Add explicit guard before index access:
```python
if valid_rows.empty:
    raise ValueError("Pipeline returned no positive-weight positions — aborting EOD run")
```
**Why explicit raise:** fail loudly so the error appears in `eod_log.jsonl` with context, not as a bare crash.

---

### E2 — DataFrame `.get()` call (Medium)
**Location:** ~line 138  
**Bug:** `current_positions.get("weight", [0.0] * len(current_positions))` — DataFrames do not have a `.get()` method. Raises `AttributeError` at runtime whenever current positions are non-empty.  
**Fix:**
```python
current_weights_list = (
    current_positions["weight"].tolist()
    if "weight" in current_positions.columns
    else [0.0] * len(current_positions)
)
```

---

### E3 — Empty cost features dict passed to order engine (Medium)
**Location:** ~line 789  
**Bug:** `compute_orders(..., features=_cost_features or None)` — an empty dict `{}` is falsy, so `or None` works correctly. But if `_cost_features` is a partially-populated dict (has some keys but not the ones the order engine expects), the `or None` check passes it through silently.  
**Fix:** Check for required keys explicitly:
```python
_required_cost_keys = {"dollar_volume"}
features_arg = _cost_features if (_cost_features and _required_cost_keys.issubset(_cost_features)) else None
compute_orders(..., features=features_arg)
```

---

### E4 — Pivot silently empty on missing `dollar_volume` column (Low)
**Location:** ~line 781  
**Bug:** `_prices_raw.pivot_table(index=..., columns=..., values="dollar_volume")` produces an empty DataFrame silently if `dollar_volume` is absent. Cost filtering is inactive with no warning.  
**Fix:** Add column check before pivoting:
```python
if "dollar_volume" not in _prices_raw.columns:
    print("[EodRunner] WARNING: dollar_volume missing from prices cache — cost features disabled")
    _cost_features = {}
else:
    _dv = _prices_raw.pivot_table(...)
```

---

### E5 — Silent regime extraction failure in EOD runner (Low)
**Location:** ~line 104, `run_eod_with_weights()`  
**Bug:** `regime_engine.get_signal()` failure is caught, prints a warning, and sets `posture="unknown"` — execution continues in a degraded state with no regime context. No entry in `eod_log.jsonl`, so the next morning there's no record that regime was unknown during order submission.  
**Fix:** Log the degraded state to `eod_log.jsonl` alongside the print, so it's auditable:
```python
except Exception as e:
    print(f"[EodRunner] WARNING: regime signal unavailable ({e}) — proceeding with posture=unknown")
    _log_warning("regime_unavailable", str(e))  # write to eod_log
```

---

## Layer 2: Debate (`debate/debate_runner.py`)

### D1 — `portfolio_state["weights"]` None-safety (Medium)
**Location:** ~line 140  
**Bug:** `portfolio_state.get("weights", {}).items()` — if weights is explicitly `None` (not missing), `.get()` returns `None` and `.items()` raises `AttributeError`.  
**Fix:**
```python
(portfolio_state.get("weights") or {}).items()
```

---

### D2 — Quant context exception log not actionable (Medium)
**Location:** ~lines 191–202  
**Bug:** Outer `except Exception as e` prints `"Quant context skipped: {e}"` regardless of cause. A missing import looks identical to a missing cache file.  
**Fix:** Distinguish causes:
```python
except FileNotFoundError:
    print("[Debate] Quant context skipped: prices_live cache not found")
except ImportError as e:
    print(f"[Debate] Quant context skipped: import error — {e}")
except Exception as e:
    print(f"[Debate] Quant context skipped: {type(e).__name__}: {e}")
```
Non-fatal in all cases — debate proceeds without quant data.

---

## Layer 3: Runner (`run_all_agents.py`)

### R1 — Empty list IndexError on regime signal (Critical)
**Location:** ~line 392  
**Bug:** `(_rdata[-1] if isinstance(_rdata, list) else _rdata).get("label", "unknown")` — if `regime_signal.json` contains `[]`, `_rdata[-1]` raises `IndexError` before `.get()` is reached. This crashes the debate setup block, losing regime context for the entire session.  
**Fix:**
```python
_sig = (_rdata[-1] if (isinstance(_rdata, list) and _rdata) else _rdata) or {}
_saved_regime = _sig.get("label", "unknown")
```

---

### R2 — Hardcoded allocation dict diverges from orchestrator output (Medium)
**Location:** ~line 400  
**Bug:** `portfolio_state["allocation"]` is set from a static dict `{"us_equities": 0.60, "macro": 0.15, ...}`. If the orchestrator computes different allocations (skill-adjusted, regime-adjusted), the debate agents see stale allocation numbers.  
**Fix:** Pull from orchestrator output when available:
```python
_base_alloc = {"us_equities": 0.60, "macro": 0.15, "international": 0.15, "alternatives": 0.10}
portfolio_state["allocation"] = getattr(orchestrator_result, "allocation", None) or _base_alloc
```
Where `orchestrator_result` is whatever `merge_agent_outputs()` returns. If it doesn't expose allocation yet, fall back to static — but add a TODO to wire it in.

---

### R3 — Bare except on regime staleness check (Low)
**Location:** ~line 43, `_is_regime_stale()`  
**Bug:** `except Exception: return True` silently swallows permission errors, corrupt JSON, and all other failures — all treated as "stale" with no log entry.  
**Fix:**
```python
except Exception as e:
    print(f"[Runner] Regime staleness check failed ({type(e).__name__}: {e}) — treating as stale")
    return True
```

---

### R4 — `AgentOutput.n_positions` schema drift (Low)
**Location:** `run_all_agents.py` ~line 457, `ascent/config/types.py`  
**Bug:** `ao.n_positions if hasattr(ao, "n_positions") else len(getattr(ao, "target_weights", {}))` — this guard exists because `n_positions` is missing from some `AgentOutput` instances. Root cause: `AgentOutput` dataclass doesn't define `n_positions` with a default, so agents that don't set it crash on access.  
**Fix:** Add `n_positions: int = 0` to `AgentOutput` in `ascent/config/types.py`. Remove the `hasattr` guard — use `ao.n_positions` directly (defaults to 0 if not set by agent).

---

## Implementation Plan

**Branch:** `feature/bug-hardening`  
**Order:** E1 → E2 → E3 → E4 → D1 → D2 → R1 → R2 → R3 → R4  
**Tests:** Add tests where fix is non-obvious (E1 empty weights, R1 empty list). Skip test overhead for pure log/guard additions.  
**Commit strategy:** One commit per layer (execution, debate, runner), then merge to main.

---

## Success Criteria

- All 11 issues addressed
- `pytest tests/ -q` still passes (139 tests)
- No new bare `except` blocks introduced
- `eod_runner.py` raises explicitly on empty weights (E1) — verifiable via test
- `AgentOutput` dataclass has `n_positions: int = 0` (R4) — `hasattr` guard removed
