# Position-Level Stop-Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exit any position that has fallen more than a configured threshold below its entry price, and block re-entry for a cooldown window — implementing Han, Zhou & Zhu (2014), "Taming Momentum Crashes: A Simple Stop-Loss Strategy."

**Architecture:** One shared pure module (`ascent/portfolio/stop_loss.py`) holds the rule. Production reads entry prices from the live Alpaca book; research reconstructs them from the price panel. Both call the same functions, following the precedent set by `ascent/portfolio/exposure.py` after research and production silently diverged on overlays. Stopped weight goes to **cash**, not to other names, and the stop is applied **last** — after every cap and overlay — so nothing re-inflates it.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, Alpaca REST (`ascent/execution/alpaca_broker.py`).

## Global Constraints

- **HARD PREREQUISITE:** `docs/superpowers/plans/2026-07-27-backtest-cash-bucket-fix.md` must be landed first. Until it is, `BacktestEngine` re-levers any sub-1.0 gross book back to 1.0 the next day, so a stop-to-cash rule is erased in research and cannot be validated. Do not start this plan before that one is merged and its tests pass.
- Always use `.venv/bin/python`. Never the system Python.
- Use `import logging`; **never** `from loguru import logger` (loguru is not installed).
- Config access via `get_config()` — never `Config()` directly.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- New config flags default to **`False`** (disabled). This ships the machinery dark; a separate, deliberate decision enables it after WF validation (Task 7).
- Any sizing change must be mirrored in **both** `ascent/main.py` and `ascent/research/wf_framework/ascent_strategy.py`, or research and production diverge silently.
- Risk overlays never raise. On any internal failure, log a warning and return the input unchanged (fail-open), matching `enforce_cluster_cap`.
- `ascent/main.py` must keep returning a 10-tuple.
- Threshold default **0.10** and cooldown default **30 calendar days** (≈21 trading days) are the paper's values. Both are config-tunable.

---

## Evidence and design decisions

**Paper result (Han, Zhou & Zhu 2014):** at a 10% stop level, maximum monthly loss of the equal-weighted momentum strategy fell from −49.79% to −11.36% and the Sharpe ratio more than doubled, while average monthly return *rose* (1.01% → 1.73%). For momentum specifically the stop is not a defensive tax — the tail losses it prevents dominate the upside it occasionally cuts short.

**Measured on this book (2026-07-27, actual prices, entry from the 2026-06-29 rebalance):**

| Stop | ALGM | MRNA | Portfolio saved |
|---|---|---|---|
| 10% | exits 07-02 @ 55.49 → −16.40% (vs −30.65% held) | exits 07-17 @ 61.82 → −11.31% (vs −22.42% held) | **+1.037pp** |
| 15% | exits 07-02 @ 55.49 → −16.40% | exits 07-22 @ 58.07 → −16.69% | +0.817pp |
| 20% | exits 07-07 @ 51.55 → −22.33% | never exits → −22.42% | +0.340pp |

10% recovers about a third of the −3.40% episode. It does **not** prevent the first cliff (ALGM gapped from −10% to −16% in one session), and this plan does not claim it does.

**Decisions locked in (each is config-tunable; these are the defaults):**

1. **Exit to cash, not redistribute.** Redistributing a stopped name's weight across the remaining momentum book re-risks into the same factor that just broke. Cash is the faithful reading of the paper and the actual risk reduction. `redistribute=True` exists for research comparison.
2. **Applied last, after caps and overlays.** `_water_fill_cap`, `enforce_cluster_cap`, `enforce_risk_budget_cap`, and `apply_exposure_overlays` all renormalize or redistribute. Any of them running after the stop would refill the stopped name.
3. **Cooldown is mandatory.** Without it the next rebalance simply re-buys the stopped name, because its momentum rank has not updated yet. 30 calendar days, tracked in a JSON state file.
4. **Cooldown measured in calendar days, not trading days.** `bdate_range(end="today")` returns empty on weekends (known gotcha), and a trading-calendar dependency here buys nothing.
5. **Missing data never triggers a stop.** No entry price or no current price → not breached, log a warning. Fail-open, consistent with the rest of the risk layer.

---

## File Structure

- `ascent/portfolio/stop_loss.py` — **new**. The whole rule: breach detection, weight application, cooldown state, and the research panel driver. Single source of truth for both callers.
- `ascent/execution/alpaca_broker.py` — `get_positions()` extended to surface `avg_entry_price` and `unrealized_plpc` (it currently returns only symbol/qty/market_value/current_price/weight, so production has no entry price to compare against).
- `ascent/config/settings.py` — four new `BacktestConfig` fields.
- `run_all_agents.py` — production wiring, immediately before order submission.
- `ascent/research/wf_framework/ascent_strategy.py` — research wiring, parity with production.
- `data_cache/stop_loss_state.json` — runtime cooldown state (created on first stop; not committed).
- `tests/portfolio/test_stop_loss.py` — **new**. Core rule + cooldown + panel + the ALGM/MRNA regression scenario.
- `tests/execution/test_alpaca_positions_entry.py` — **new**. Broker field contract.

---

### Task 1: Core stop-loss rule (pure functions)

**Files:**
- Create: `ascent/portfolio/stop_loss.py`
- Create: `tests/portfolio/test_stop_loss.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `STOP_THRESHOLD: float = 0.10`
  - `COOLDOWN_DAYS: int = 30`
  - `compute_stop_breaches(entry_prices: pd.Series, current_prices: pd.Series, threshold: float = STOP_THRESHOLD) -> pd.Series` (bool, indexed by symbol)
  - `apply_stop_loss(weights: pd.Series, breached: pd.Series, redistribute: bool = False) -> pd.Series`

- [ ] **Step 1: Write the failing test**

Create `tests/portfolio/test_stop_loss.py`:

```python
# tests/portfolio/test_stop_loss.py
"""
Position-level stop-loss — Han, Zhou & Zhu (2014).

Exit a name that has fallen more than `threshold` below its entry price;
block re-entry for a cooldown window. Stopped weight goes to cash by
default rather than being redistributed into the remaining momentum book.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.stop_loss import (
    STOP_THRESHOLD,
    compute_stop_breaches,
    apply_stop_loss,
)


class TestComputeStopBreaches:
    def test_name_below_threshold_is_breached(self):
        entry = pd.Series({"ALGM": 66.37, "TLT": 87.41})
        now = pd.Series({"ALGM": 55.49, "TLT": 87.00})  # ALGM -16.4%, TLT -0.5%
        out = compute_stop_breaches(entry, now, threshold=0.10)
        assert out["ALGM"] is np.True_ or out["ALGM"] == True  # noqa: E712
        assert not out["TLT"]

    def test_exactly_at_threshold_is_breached(self):
        """-10.0% with a 10% stop breaches (inclusive), no float-edge escape."""
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 90.0})
        assert compute_stop_breaches(entry, now, threshold=0.10)["A"]

    def test_just_inside_threshold_is_not_breached(self):
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 90.5})  # -9.5%
        assert not compute_stop_breaches(entry, now, threshold=0.10)["A"]

    def test_gain_is_never_breached(self):
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 156.6})
        assert not compute_stop_breaches(entry, now, threshold=0.10)["A"]

    def test_missing_current_price_is_not_breached(self, caplog):
        """Fail-open: unknown price must never trigger a forced exit."""
        entry = pd.Series({"A": 100.0, "B": 100.0})
        now = pd.Series({"A": 50.0})  # B absent
        with caplog.at_level("WARNING"):
            out = compute_stop_breaches(entry, now, threshold=0.10)
        assert out["A"]
        assert not out["B"]
        assert any("B" in rec.message for rec in caplog.records)

    def test_missing_entry_price_is_not_breached(self):
        entry = pd.Series({"A": 100.0})
        now = pd.Series({"A": 50.0, "B": 1.0})
        out = compute_stop_breaches(entry, now, threshold=0.10)
        assert out["A"]
        assert not out["B"]

    def test_non_positive_entry_price_is_not_breached(self):
        entry = pd.Series({"A": 0.0, "B": -5.0})
        now = pd.Series({"A": 1.0, "B": 1.0})
        out = compute_stop_breaches(entry, now, threshold=0.10)
        assert not out.any()

    def test_empty_input_returns_empty(self):
        out = compute_stop_breaches(pd.Series(dtype=float), pd.Series(dtype=float))
        assert out.empty
        assert out.dtype == bool


class TestApplyStopLoss:
    def test_breached_name_goes_to_zero_and_gross_falls(self):
        w = pd.Series({"ALGM": 0.04, "MRNA": 0.04, "TLT": 0.08})
        breached = pd.Series({"ALGM": True, "MRNA": False, "TLT": False})
        out = apply_stop_loss(w, breached)
        assert out["ALGM"] == 0.0
        assert out["MRNA"] == pytest.approx(0.04)
        assert out["TLT"] == pytest.approx(0.08)
        # Freed weight becomes cash, it does NOT refill the book.
        assert out.sum() == pytest.approx(0.12)

    def test_redistribute_true_preserves_gross(self):
        w = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10})
        breached = pd.Series({"A": True, "B": False, "C": False})
        out = apply_stop_loss(w, breached, redistribute=True)
        assert out["A"] == 0.0
        assert out.sum() == pytest.approx(w.sum())
        assert out["B"] == pytest.approx(0.15)
        assert out["C"] == pytest.approx(0.15)

    def test_no_breach_is_exact_noop(self):
        w = pd.Series({"A": 0.10, "B": 0.10})
        breached = pd.Series({"A": False, "B": False})
        pd.testing.assert_series_equal(apply_stop_loss(w, breached), w.astype(float))

    def test_all_breached_goes_fully_to_cash(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        breached = pd.Series({"A": True, "B": True})
        out = apply_stop_loss(w, breached)
        assert out.sum() == pytest.approx(0.0)

    def test_all_breached_with_redistribute_does_not_divide_by_zero(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        breached = pd.Series({"A": True, "B": True})
        out = apply_stop_loss(w, breached, redistribute=True)
        assert out.sum() == pytest.approx(0.0)
        assert not out.isna().any()

    def test_symbol_missing_from_breach_series_is_kept(self):
        w = pd.Series({"A": 0.10, "B": 0.10})
        breached = pd.Series({"A": True})  # B unknown
        out = apply_stop_loss(w, breached)
        assert out["A"] == 0.0
        assert out["B"] == pytest.approx(0.10)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_stop_loss.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ascent.portfolio.stop_loss'`.

- [ ] **Step 3: Write the implementation**

Create `ascent/portfolio/stop_loss.py`:

```python
# ascent/portfolio/stop_loss.py
"""
Position-level stop-loss — single source of truth.

Implements Han, Zhou & Zhu (2014), "Taming Momentum Crashes: A Simple
Stop-Loss Strategy": exit a position that has fallen more than `threshold`
below its entry price, and block re-entry for a cooldown window.

Both the production path (run_all_agents.py, entry prices from the live
Alpaca book) and the walk-forward framework (ascent/research/wf_framework/
ascent_strategy.py, entry prices reconstructed from the price panel) MUST
go through this module. See ascent/portfolio/exposure.py for the precedent:
research and production previously carried separate overlay implementations
and silently diverged.

Design notes:
  * Stopped weight goes to CASH by default. Redistributing into the
    remaining book re-risks into the same factor that just broke.
  * The stop must be applied LAST, after every cap and overlay, because
    _water_fill_cap / enforce_cluster_cap / enforce_risk_budget_cap /
    apply_exposure_overlays all renormalize and would refill the name.
  * Missing data never triggers a stop (fail-open), matching
    enforce_cluster_cap's never-raise contract.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STOP_THRESHOLD = 0.10   # Han, Zhou & Zhu (2014) headline stop level
COOLDOWN_DAYS  = 30     # calendar days, ~21 trading days


def compute_stop_breaches(
    entry_prices: pd.Series,
    current_prices: pd.Series,
    threshold: float = STOP_THRESHOLD,
) -> pd.Series:
    """
    Boolean Series (indexed by symbol) marking positions that have fallen
    `threshold` or more below their entry price.

    A breach requires a positive entry price and a positive current price.
    Anything unresolvable (missing on either side, non-positive, NaN) is
    reported as NOT breached and logged — an unknown price must never force
    a liquidation.

    The comparison is inclusive: exactly -threshold breaches.
    """
    idx = entry_prices.index.union(current_prices.index)
    if len(idx) == 0:
        return pd.Series(dtype=bool)

    entry = pd.to_numeric(entry_prices.reindex(idx), errors="coerce")
    now   = pd.to_numeric(current_prices.reindex(idx), errors="coerce")

    resolvable = entry.notna() & now.notna() & (entry > 0) & (now > 0)
    unresolved = list(idx[~resolvable])
    if unresolved:
        log.warning(
            "[StopLoss] Cannot evaluate stop for %s (missing/invalid entry or "
            "current price) — treating as NOT breached", unresolved,
        )

    pct = pd.Series(np.nan, index=idx, dtype=float)
    pct.loc[resolvable] = now.loc[resolvable] / entry.loc[resolvable] - 1.0

    # Inclusive at the threshold; 1e-12 absorbs float representation error.
    breached = resolvable & (pct <= -abs(threshold) + 1e-12)
    return breached.fillna(False).astype(bool)


def apply_stop_loss(
    weights: pd.Series,
    breached: pd.Series,
    redistribute: bool = False,
) -> pd.Series:
    """
    Zero out breached names.

    redistribute=False (default): freed weight becomes cash, gross exposure
    falls. This is the actual risk reduction and the faithful reading of the
    paper.

    redistribute=True: freed weight is spread pro-rata across survivors,
    preserving gross. Provided for research comparison only.
    """
    if weights is None or len(weights) == 0:
        return weights

    w = weights.astype(float).copy()
    mask = breached.reindex(w.index).fillna(False).astype(bool)
    if not mask.any():
        return w

    freed = float(w[mask].sum())
    out = w.copy()
    out[mask] = 0.0

    if redistribute:
        survivors = ~mask
        surv_total = float(out[survivors].sum())
        if surv_total > 0:
            out[survivors] = out[survivors] / surv_total * (surv_total + freed)
        # else: every name breached — everything is already cash, nothing to
        # redistribute into. Returning all-zeros is correct.

    log.info(
        "[StopLoss] Stopped %d position(s) %s — %.4f of gross moved to %s",
        int(mask.sum()), list(w.index[mask]), freed,
        "survivors" if redistribute else "cash",
    )
    return out
```

- [ ] **Step 4: Verify the file parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/stop_loss.py').read())"`
Expected: no output.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/portfolio/test_stop_loss.py -v`
Expected: all tests in `TestComputeStopBreaches` and `TestApplyStopLoss` pass.

- [ ] **Step 6: Commit**

```bash
git add ascent/portfolio/stop_loss.py tests/portfolio/test_stop_loss.py
git commit -m "feat(portfolio): add position-level stop-loss core rule

Han, Zhou & Zhu (2014). compute_stop_breaches + apply_stop_loss, shared by
production and research. Stopped weight goes to cash; missing data never
triggers a stop (fail-open)."
```

---

### Task 2: Cooldown state (re-entry block)

**Files:**
- Modify: `ascent/portfolio/stop_loss.py`
- Modify: `tests/portfolio/test_stop_loss.py`

**Interfaces:**
- Consumes: `STOP_THRESHOLD`, `COOLDOWN_DAYS` from Task 1.
- Produces:
  - `DEFAULT_STATE_PATH: str = "data_cache/stop_loss_state.json"`
  - `load_stop_state(path: str = DEFAULT_STATE_PATH) -> dict[str, str]` — `{symbol: ISO date of last stop}`
  - `record_stops(symbols: list[str], today: str, path: str = DEFAULT_STATE_PATH) -> dict[str, str]`
  - `blocked_symbols(state: dict[str, str], today: str, cooldown_days: int = COOLDOWN_DAYS) -> set[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/portfolio/test_stop_loss.py`:

```python
import json

from ascent.portfolio.stop_loss import (
    COOLDOWN_DAYS,
    load_stop_state,
    record_stops,
    blocked_symbols,
)


class TestCooldownState:
    def test_record_then_load_roundtrip(self, tmp_path):
        p = str(tmp_path / "state.json")
        record_stops(["ALGM", "MRNA"], "2026-07-02", path=p)
        state = load_stop_state(p)
        assert state == {"ALGM": "2026-07-02", "MRNA": "2026-07-02"}

    def test_record_overwrites_older_stop_for_same_symbol(self, tmp_path):
        p = str(tmp_path / "state.json")
        record_stops(["ALGM"], "2026-07-02", path=p)
        record_stops(["ALGM"], "2026-07-20", path=p)
        assert load_stop_state(p)["ALGM"] == "2026-07-20"

    def test_missing_state_file_is_empty_not_an_error(self, tmp_path):
        assert load_stop_state(str(tmp_path / "nope.json")) == {}

    def test_corrupt_state_file_is_empty_not_an_error(self, tmp_path, caplog):
        p = tmp_path / "state.json"
        p.write_text("{not valid json")
        with caplog.at_level("WARNING"):
            assert load_stop_state(str(p)) == {}
        assert caplog.records

    def test_symbol_is_blocked_inside_cooldown(self):
        state = {"ALGM": "2026-07-02"}
        assert "ALGM" in blocked_symbols(state, "2026-07-10", cooldown_days=30)

    def test_symbol_is_free_after_cooldown(self):
        state = {"ALGM": "2026-07-02"}
        assert "ALGM" not in blocked_symbols(state, "2026-08-02", cooldown_days=30)

    def test_boundary_day_is_free(self):
        """Exactly cooldown_days later the name is tradeable again."""
        state = {"A": "2026-07-01"}
        assert "A" not in blocked_symbols(state, "2026-07-31", cooldown_days=30)

    def test_unparseable_date_does_not_block(self, caplog):
        """Fail-open: bad state must not permanently freeze a symbol out."""
        with caplog.at_level("WARNING"):
            out = blocked_symbols({"A": "garbage"}, "2026-07-10")
        assert "A" not in out
        assert caplog.records

    def test_empty_state_blocks_nothing(self):
        assert blocked_symbols({}, "2026-07-10") == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_stop_loss.py::TestCooldownState -v`
Expected: FAIL — `ImportError: cannot import name 'load_stop_state'`.

- [ ] **Step 3: Write the implementation**

Append to `ascent/portfolio/stop_loss.py`:

```python
import json
from datetime import date, timedelta
from pathlib import Path

DEFAULT_STATE_PATH = "data_cache/stop_loss_state.json"


def load_stop_state(path: str = DEFAULT_STATE_PATH) -> dict:
    """
    Load {symbol: ISO date of last stop}. A missing or corrupt file yields
    an empty state — a broken state file must never block trading.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"expected a dict, got {type(data).__name__}")
        return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        log.warning("[StopLoss] Could not read stop state %s (%s) — treating "
                    "as empty", path, exc)
        return {}


def record_stops(symbols: list, today: str,
                 path: str = DEFAULT_STATE_PATH) -> dict:
    """
    Record `symbols` as stopped on `today` (ISO YYYY-MM-DD) and persist.
    Re-stopping a symbol refreshes its date. Returns the new state.
    """
    state = load_stop_state(path)
    for s in symbols:
        state[str(s)] = str(today)
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception as exc:
        log.warning("[StopLoss] Could not persist stop state to %s: %s",
                    path, exc)
    return state


def blocked_symbols(state: dict, today: str,
                    cooldown_days: int = COOLDOWN_DAYS) -> set:
    """
    Symbols still inside their re-entry cooldown as of `today`.

    Cooldown is measured in CALENDAR days: bdate_range(end="today") returns
    empty on weekends (known repo gotcha) and a trading-calendar dependency
    buys nothing here. The boundary is exclusive — exactly `cooldown_days`
    after the stop, the symbol is tradeable again.

    An unparseable date does not block (fail-open): a corrupt entry must not
    freeze a symbol out of the book forever.
    """
    if not state:
        return set()
    try:
        today_d = date.fromisoformat(str(today))
    except Exception as exc:
        log.warning("[StopLoss] Bad 'today' value %r (%s) — blocking nothing",
                    today, exc)
        return set()

    out = set()
    for sym, stopped_on in state.items():
        try:
            d = date.fromisoformat(str(stopped_on))
        except Exception:
            log.warning("[StopLoss] Unparseable stop date %r for %s — not "
                        "blocking", stopped_on, sym)
            continue
        if today_d < d + timedelta(days=int(cooldown_days)):
            out.add(sym)
    return out
```

- [ ] **Step 4: Verify the file parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/stop_loss.py').read())"`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/portfolio/test_stop_loss.py -v`
Expected: all pass, including the new `TestCooldownState`.

- [ ] **Step 6: Add the state file to .gitignore**

Append to `.gitignore`:

```
data_cache/stop_loss_state.json
```

- [ ] **Step 7: Commit**

```bash
git add ascent/portfolio/stop_loss.py tests/portfolio/test_stop_loss.py .gitignore
git commit -m "feat(portfolio): add stop-loss re-entry cooldown state

Without a cooldown the next rebalance simply re-buys the stopped name,
because its momentum rank has not updated. 30 calendar days, JSON-backed,
fail-open on missing or corrupt state."
```

---

### Task 3: Research panel driver + ALGM/MRNA regression scenario

**Files:**
- Modify: `ascent/portfolio/stop_loss.py`
- Modify: `tests/portfolio/test_stop_loss.py`

**Interfaces:**
- Consumes: `compute_stop_breaches`, `apply_stop_loss`, `STOP_THRESHOLD`, `COOLDOWN_DAYS`.
- Produces: `apply_stop_loss_panel(weights: pd.DataFrame, close: pd.DataFrame, threshold: float = STOP_THRESHOLD, cooldown_days: int = COOLDOWN_DAYS, redistribute: bool = False) -> tuple[pd.DataFrame, list[dict]]`
  - Returns `(stopped_weights, events)` where each event is
    `{"date": str, "symbol": str, "entry_price": float, "exit_price": float, "pct_from_entry": float}`.

**Why a separate panel function:** research has no broker to ask for entry prices, so it must reconstruct them by walking the weights panel forward and recording the close on the day each name enters the book. This is stateful across dates and cannot be expressed as a vectorized row operation.

- [ ] **Step 1: Write the failing test**

Append to `tests/portfolio/test_stop_loss.py`:

```python
from ascent.portfolio.stop_loss import apply_stop_loss_panel


class TestStopLossPanel:
    def test_position_is_stopped_and_stays_out_for_cooldown(self):
        dates = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        )
        # A crashes 20% on day 3; B is flat.
        close = pd.DataFrame(
            {"A": [100.0, 99.0, 80.0, 81.0], "B": [50.0, 50.0, 50.0, 50.0]},
            index=dates,
        )
        w = pd.DataFrame(0.5, index=dates, columns=["A", "B"])

        out, events = apply_stop_loss_panel(
            w, close, threshold=0.10, cooldown_days=30
        )

        assert out.loc[dates[0], "A"] == pytest.approx(0.5)   # entry day
        assert out.loc[dates[1], "A"] == pytest.approx(0.5)   # -1%, fine
        assert out.loc[dates[2], "A"] == 0.0                  # -20%, stopped
        assert out.loc[dates[3], "A"] == 0.0                  # cooldown
        # B untouched throughout — no redistribution.
        assert (out["B"] == pytest.approx(0.5)).all()

        assert len(events) == 1
        assert events[0]["symbol"] == "A"
        assert events[0]["entry_price"] == pytest.approx(100.0)
        assert events[0]["pct_from_entry"] == pytest.approx(-0.20)

    def test_gross_falls_when_a_name_is_stopped(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        close = pd.DataFrame({"A": [100.0, 80.0], "B": [50.0, 50.0]}, index=dates)
        w = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
        out, _ = apply_stop_loss_panel(w, close, threshold=0.10)
        assert out.loc[dates[0]].sum() == pytest.approx(1.0)
        assert out.loc[dates[1]].sum() == pytest.approx(0.5)  # A -> cash

    def test_reentry_allowed_after_cooldown_resets_entry_price(self):
        dates = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-03-01", "2026-03-02"]
        )
        close = pd.DataFrame({"A": [100.0, 80.0, 40.0, 39.0]}, index=dates)
        w = pd.DataFrame(1.0, index=dates, columns=["A"])
        out, events = apply_stop_loss_panel(w, close, threshold=0.10,
                                            cooldown_days=30)
        assert out.loc[dates[1], "A"] == 0.0     # stopped
        assert out.loc[dates[2], "A"] == 1.0     # cooldown expired, re-entered
        # Re-entry price is 40.0, so -2.5% on the next day is NOT a breach.
        assert out.loc[dates[3], "A"] == 1.0
        assert len(events) == 1

    def test_name_that_exits_naturally_clears_its_entry(self):
        """Weight -> 0 by the strategy, then back in later at a new price."""
        dates = pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03"]
        )
        close = pd.DataFrame({"A": [100.0, 100.0, 91.0]}, index=dates)
        w = pd.DataFrame({"A": [1.0, 0.0, 1.0]}, index=dates)
        out, events = apply_stop_loss_panel(w, close, threshold=0.10)
        # Re-entry on day 3 at 91.0 is a fresh entry, not -9% from 100.
        assert out.loc[dates[2], "A"] == 1.0
        assert events == []

    def test_disabled_threshold_zero_is_a_noop(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        close = pd.DataFrame({"A": [100.0, 1.0]}, index=dates)
        w = pd.DataFrame(1.0, index=dates, columns=["A"])
        out, events = apply_stop_loss_panel(w, close, threshold=0.0)
        pd.testing.assert_frame_equal(out, w.astype(float))
        assert events == []

    def test_empty_weights_returns_empty(self):
        out, events = apply_stop_loss_panel(pd.DataFrame(), pd.DataFrame())
        assert out.empty
        assert events == []


class TestAlgmMrnaRegression:
    """
    The 2026-06-29 -> 2026-07-24 episode that motivated this work.

    Real closes. Entry prices are the 2026-06-29 rebalance closes. Held to
    the end, ALGM returned -30.65% and MRNA -22.42%. A 10% stop should exit
    ALGM on 2026-07-02 and MRNA on 2026-07-17.
    """

    ALGM = [66.370003, 69.620003, 63.200001, 55.485001, 56.560001, 51.549999,
            51.465000, 57.380001, 54.869999, 50.855000, 52.320000, 50.029999,
            47.119999, 46.480000, 46.360001, 49.349998, 49.869999, 50.070000,
            46.029999]
    MRNA = [69.699997, 70.029999, 72.500000, 79.760002, 81.800003, 79.769997,
            73.800003, 76.559998, 68.269997, 67.010002, 67.440002, 68.279999,
            63.150002, 61.820000, 59.490002, 59.660000, 58.070000, 57.020000,
            54.070000]
    DATES = ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
             "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
             "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
             "2026-07-16", "2026-07-17", "2026-07-20", "2026-07-21",
             "2026-07-22", "2026-07-23", "2026-07-24"]

    def _panel(self):
        idx = pd.to_datetime(self.DATES)
        close = pd.DataFrame({"ALGM": self.ALGM, "MRNA": self.MRNA}, index=idx)
        w = pd.DataFrame(0.04086962656941537, index=idx,
                         columns=["ALGM", "MRNA"])
        return w, close

    def test_ten_percent_stop_exits_both_on_the_expected_dates(self):
        w, close = self._panel()
        out, events = apply_stop_loss_panel(w, close, threshold=0.10,
                                            cooldown_days=30)
        by_sym = {e["symbol"]: e for e in events}
        assert set(by_sym) == {"ALGM", "MRNA"}
        assert str(pd.Timestamp(by_sym["ALGM"]["date"]).date()) == "2026-07-02"
        assert str(pd.Timestamp(by_sym["MRNA"]["date"]).date()) == "2026-07-17"
        # Both stay out for the remainder (cooldown covers the window).
        assert out.iloc[-1].sum() == pytest.approx(0.0)

    def test_ten_percent_stop_saves_about_one_percentage_point(self):
        """
        Measured 2026-07-27: +1.037pp vs holding. Assert the magnitude, with
        tolerance for the exact fill convention.
        """
        w, close = self._panel()
        _, events = apply_stop_loss_panel(w, close, threshold=0.10)
        weight = 0.04086962656941537
        held = {"ALGM": close["ALGM"].iloc[-1] / close["ALGM"].iloc[0] - 1,
                "MRNA": close["MRNA"].iloc[-1] / close["MRNA"].iloc[0] - 1}
        saved = sum(
            (e["pct_from_entry"] - held[e["symbol"]]) * weight for e in events
        )
        assert saved == pytest.approx(0.01037, abs=0.002)

    def test_twenty_percent_stop_saves_much_less(self):
        """Threshold matters: a 20% stop recovers roughly a third as much."""
        w, close = self._panel()
        _, events = apply_stop_loss_panel(w, close, threshold=0.20)
        weight = 0.04086962656941537
        held = {"ALGM": close["ALGM"].iloc[-1] / close["ALGM"].iloc[0] - 1,
                "MRNA": close["MRNA"].iloc[-1] / close["MRNA"].iloc[0] - 1}
        saved = sum(
            (e["pct_from_entry"] - held[e["symbol"]]) * weight for e in events
        )
        assert saved == pytest.approx(0.0034, abs=0.002)
        assert saved < 0.01037
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_stop_loss.py::TestStopLossPanel -v`
Expected: FAIL — `ImportError: cannot import name 'apply_stop_loss_panel'`.

- [ ] **Step 3: Write the implementation**

Append to `ascent/portfolio/stop_loss.py`:

```python
def apply_stop_loss_panel(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    threshold: float = STOP_THRESHOLD,
    cooldown_days: int = COOLDOWN_DAYS,
    redistribute: bool = False,
) -> tuple:
    """
    Apply the stop-loss rule across a (dates x symbols) weights panel.

    Research counterpart to the production path: there is no broker to ask
    for entry prices, so they are reconstructed by walking the panel forward
    and recording the close on the day each name enters the book.

    Entry price is the close on the entry date. In the live path the fill is
    the next open, so research is a close-to-close approximation of the same
    rule — a deliberate, documented simplification for a risk overlay.

    threshold <= 0 disables the rule entirely (exact no-op).

    Returns (stopped_weights, events).
    """
    if weights is None or weights.empty or threshold <= 0:
        return (weights.astype(float) if weights is not None
                and not weights.empty else weights), []

    out = weights.astype(float).copy()
    px = close.reindex(index=out.index, columns=out.columns)

    entry: dict = {}          # symbol -> entry price
    blocked_until: dict = {}  # symbol -> pd.Timestamp
    events: list = []

    for dt in out.index:
        row = out.loc[dt]
        held = [s for s in out.columns if float(row.get(s, 0.0)) > 0.0]

        # 1. Names still inside their cooldown cannot be re-entered.
        for s in held:
            until = blocked_until.get(s)
            if until is not None and dt < until:
                out.loc[dt, s] = 0.0
        row = out.loc[dt]
        held = [s for s in out.columns if float(row.get(s, 0.0)) > 0.0]

        # 2. Names that left the book (by the strategy or by cooldown) lose
        #    their recorded entry, so a later re-entry prices fresh.
        for s in list(entry):
            if s not in held:
                entry.pop(s, None)

        # 3. New entries record their price and cannot breach on day one.
        fresh = []
        for s in held:
            if s not in entry:
                p = px.at[dt, s] if s in px.columns else np.nan
                if pd.notna(p) and float(p) > 0:
                    entry[s] = float(p)
                    fresh.append(s)

        # 4. Evaluate breaches for names held since a prior date.
        seasoned = [s for s in held if s in entry and s not in fresh]
        if seasoned:
            breached = compute_stop_breaches(
                pd.Series({s: entry[s] for s in seasoned}),
                pd.Series({s: px.at[dt, s] for s in seasoned}),
                threshold=threshold,
            )
            hits = [s for s in seasoned if bool(breached.get(s, False))]
            if hits:
                for s in hits:
                    exit_px = float(px.at[dt, s])
                    events.append({
                        "date":           str(dt),
                        "symbol":         s,
                        "entry_price":    entry[s],
                        "exit_price":     exit_px,
                        "pct_from_entry": exit_px / entry[s] - 1.0,
                    })
                    blocked_until[s] = dt + pd.Timedelta(days=int(cooldown_days))
                    entry.pop(s, None)
                out.loc[dt] = apply_stop_loss(
                    out.loc[dt],
                    pd.Series(True, index=hits),
                    redistribute=redistribute,
                )

    return out, events
```

- [ ] **Step 4: Verify the file parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/stop_loss.py').read())"`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/portfolio/test_stop_loss.py -v`
Expected: all pass, including both `TestAlgmMrnaRegression` cases.

If `test_ten_percent_stop_saves_about_one_percentage_point` is off, print the events and compare against the measured table in this plan's evidence section before adjusting anything — the fixture prices are real and the expected value was measured, so a mismatch means the entry/exit convention in the implementation drifted, not that the target is wrong.

- [ ] **Step 6: Commit**

```bash
git add ascent/portfolio/stop_loss.py tests/portfolio/test_stop_loss.py
git commit -m "feat(portfolio): add stop-loss panel driver for research

Reconstructs entry prices by walking the weights panel, applies the stop
with cooldown, and emits an event log. Includes the ALGM/MRNA 2026-06-29
regression scenario on real closes (+1.04pp at a 10% stop)."
```

---

### Task 4: Config flags

**Files:**
- Modify: `ascent/config/settings.py:146-155` (the risk-aware construction block)
- Create: `tests/test_stop_loss_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: four `BacktestConfig` fields —
  `stop_loss_enabled: bool = False`, `stop_loss_threshold: float = 0.10`,
  `stop_loss_cooldown_days: int = 30`, `stop_loss_redistribute: bool = False`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stop_loss_config.py`:

```python
# tests/test_stop_loss_config.py
"""Stop-loss config surface. Ships DISABLED pending WF validation."""
from ascent.config.settings import get_config


def test_stop_loss_flags_exist_with_paper_defaults():
    bt = get_config().backtest
    assert bt.stop_loss_enabled is False, (
        "stop-loss must ship disabled until walk-forward validation "
        "(Task 7 of the position-stop-loss plan) says otherwise"
    )
    assert bt.stop_loss_threshold == 0.10       # Han, Zhou & Zhu (2014)
    assert bt.stop_loss_cooldown_days == 30     # ~21 trading days
    assert bt.stop_loss_redistribute is False   # freed weight -> cash
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_stop_loss_config.py -v`
Expected: FAIL — `AttributeError: 'BacktestConfig' object has no attribute 'stop_loss_enabled'`.

- [ ] **Step 3: Add the fields**

In `ascent/config/settings.py`, immediately after the existing `risk_budget_per_name` field (currently line 154), add:

```python
    # Position-level stop-loss (Han, Zhou & Zhu 2014). Exits a name that has
    # fallen `stop_loss_threshold` below its entry price and blocks re-entry
    # for `stop_loss_cooldown_days` calendar days. Freed weight goes to cash
    # unless `stop_loss_redistribute` is set.
    # DISABLED by default — enable only after the walk-forward validation in
    # docs/superpowers/plans/2026-07-27-position-stop-loss.md (Task 7).
    stop_loss_enabled: bool = False
    stop_loss_threshold: float = 0.10
    stop_loss_cooldown_days: int = 30
    stop_loss_redistribute: bool = False
```

- [ ] **Step 4: Verify the file parses and the test passes**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/config/settings.py').read())"
.venv/bin/python -m pytest tests/test_stop_loss_config.py -v
```
Expected: no parse output; 1 passed.

- [ ] **Step 5: Commit**

```bash
git add ascent/config/settings.py tests/test_stop_loss_config.py
git commit -m "feat(config): add stop-loss flags, disabled by default"
```

---

### Task 5: Surface entry prices from the live broker

**Files:**
- Modify: `ascent/execution/alpaca_broker.py:41-68` (`get_positions`)
- Create: `tests/execution/test_alpaca_positions_entry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `get_positions()` returns a DataFrame with the existing columns
  `["symbol", "qty", "market_value", "current_price", "weight"]` **plus**
  `["avg_entry_price", "unrealized_plpc"]`, both `float`.

**Why:** verified on 2026-07-27, `get_positions()` returns only symbol/qty/market_value/current_price/weight. Production has no entry price to compare against, so the stop cannot run without this.

**Verified against the real code on 2026-07-27:** `get_positions()` fetches with an inline `requests.get(...)` (line 47) — there is no `_get_json` helper, so the test patches `ab.requests.get`. It also has an **early-return branch at line 52** with a hardcoded column list for the empty-book case; that list must gain the two new columns too, or `"avg_entry_price" in pos.columns` fails whenever the book is empty.

- [ ] **Step 1: Create the test package marker**

```bash
mkdir -p tests/execution && touch tests/execution/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/execution/test_alpaca_positions_entry.py`:

```python
# tests/execution/test_alpaca_positions_entry.py
"""
get_positions() must surface entry price so the stop-loss can evaluate
drawdown-from-entry on the live book.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from ascent.execution import alpaca_broker as ab

_FAKE = [
    {"symbol": "ALGM", "qty": "66.51392", "market_value": "3062.97",
     "current_price": "46.05", "avg_entry_price": "66.370003",
     "unrealized_plpc": "-0.30646"},
    {"symbol": "TLT", "qty": "108.481866", "market_value": "9090.78",
     "current_price": "83.80", "avg_entry_price": "87.41",
     "unrealized_plpc": "-0.04130"},
]


def _mock_get(payload):
    """Patch the inline requests.get used by get_positions()."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return patch.object(ab.requests, "get", return_value=resp)


def test_positions_include_entry_price_and_plpc():
    with _mock_get(_FAKE):
        df = ab.get_positions()
    assert {"avg_entry_price", "unrealized_plpc"} <= set(df.columns)
    row = df.set_index("symbol").loc["ALGM"]
    assert float(row["avg_entry_price"]) == 66.370003
    assert float(row["unrealized_plpc"]) < 0


def test_existing_columns_are_unchanged():
    with _mock_get(_FAKE):
        df = ab.get_positions()
    for col in ("symbol", "qty", "market_value", "current_price", "weight"):
        assert col in df.columns


def test_missing_entry_price_becomes_nan_not_an_error():
    partial = [{"symbol": "X", "qty": "1", "market_value": "10",
                "current_price": "10"}]
    with _mock_get(partial):
        df = ab.get_positions()
    assert pd.isna(df.set_index("symbol").loc["X", "avg_entry_price"])


def test_empty_book_still_exposes_the_new_columns():
    """
    get_positions() early-returns a hardcoded column list when the account
    holds nothing. The stop-loss checks `"avg_entry_price" in pos.columns`,
    so that branch must carry the new columns as well.
    """
    with _mock_get([]):
        df = ab.get_positions()
    assert df.empty
    assert {"avg_entry_price", "unrealized_plpc"} <= set(df.columns)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/execution/test_alpaca_positions_entry.py -v`
Expected: FAIL on the missing `avg_entry_price` column.

- [ ] **Step 4: Extend `get_positions`**

In `ascent/execution/alpaca_broker.py`:

1. Update the empty-book early return (line 52) to include the new columns:

```python
        return pd.DataFrame(columns=[
            "symbol", "qty", "market_value", "current_price", "weight",
            "avg_entry_price", "unrealized_plpc",
        ])
```

2. In the row-building loop, add the two optional fields. Use `.get()` plus `pd.to_numeric(..., errors="coerce")` so an absent or unparseable key becomes `NaN` instead of raising — unlike the existing required fields, these must never break a position fetch:

```python
        rows.append({
            "symbol":          p["symbol"],
            "qty":             float(p["qty"]),
            "market_value":    float(p["market_value"]),
            "current_price":   float(p["current_price"]),
            "avg_entry_price": pd.to_numeric(p.get("avg_entry_price"),
                                             errors="coerce"),
            "unrealized_plpc": pd.to_numeric(p.get("unrealized_plpc"),
                                             errors="coerce"),
        })
```

3. Update the docstring's column list.

Leave the existing `weight` computation exactly as it is.

- [ ] **Step 5: Verify parse, run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/execution/alpaca_broker.py').read())"
.venv/bin/python -m pytest tests/execution/test_alpaca_positions_entry.py -v
```

- [ ] **Step 6: Verify against the real paper account (read-only)**

```bash
.venv/bin/python -c "
from ascent.execution import alpaca_broker as ab
df = ab.get_positions()
print(df[['symbol','avg_entry_price','current_price','unrealized_plpc']].head())
"
```
Expected: real entry prices. Cross-check one row: `current_price/avg_entry_price - 1` should be within rounding of `unrealized_plpc`. This is a read-only call; it submits nothing.

- [ ] **Step 7: Commit**

```bash
git add ascent/execution/alpaca_broker.py tests/execution/
git commit -m "feat(execution): surface avg_entry_price and unrealized_plpc

get_positions() previously returned no entry price, so the position-level
stop-loss had nothing to measure drawdown-from-entry against."
```

---

### Task 6: Wire into production and research

**Files:**
- Modify: `run_all_agents.py` (immediately before order submission)
- Modify: `ascent/research/wf_framework/ascent_strategy.py` (after the risk-budget cap block, ~line 250-260)
- Create: `tests/strategy/test_stop_loss_wiring.py`

**Interfaces:**
- Consumes: `compute_stop_breaches`, `apply_stop_loss`, `apply_stop_loss_panel`, `load_stop_state`, `record_stops`, `blocked_symbols` (Tasks 1-3); the config flags (Task 4); `get_positions()` with entry prices (Task 5).
- Produces: `_apply_stop_loss_to_book(target_weights: dict, today: str) -> tuple[dict, list[str]]` in `run_all_agents.py`, returning the adjusted book and the list of stopped symbols.

- [ ] **Step 1: Write the failing test**

Create `tests/strategy/test_stop_loss_wiring.py`:

```python
# tests/strategy/test_stop_loss_wiring.py
"""
Production wiring: the stop must run on the LIVE book, after every cap and
overlay, and must not be undone by redistribution.
"""
from unittest.mock import patch

import pandas as pd
import pytest


def _live_book():
    return pd.DataFrame([
        {"symbol": "ALGM", "qty": 66.5, "market_value": 3062.97,
         "current_price": 46.05, "weight": 0.029,
         "avg_entry_price": 66.370003, "unrealized_plpc": -0.30646},
        {"symbol": "TLT", "qty": 108.5, "market_value": 9090.78,
         "current_price": 83.80, "weight": 0.086,
         "avg_entry_price": 87.41, "unrealized_plpc": -0.04130},
    ])


class TestStopLossWiring:
    def test_disabled_flag_is_a_noop(self):
        import run_all_agents as raa
        from ascent.config.settings import get_config
        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", False):
            out, stopped = raa._apply_stop_loss_to_book(target, "2026-07-27")
        assert out == target
        assert stopped == []

    def test_breached_name_is_zeroed_and_others_untouched(self, tmp_path):
        import run_all_agents as raa
        from ascent.config.settings import get_config
        from ascent.portfolio import stop_loss as sl

        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", True), \
             patch.object(cfg.backtest, "stop_loss_threshold", 0.10), \
             patch.object(sl, "DEFAULT_STATE_PATH", str(tmp_path / "s.json")), \
             patch("ascent.execution.alpaca_broker.get_positions",
                   return_value=_live_book()):
            out, stopped = raa._apply_stop_loss_to_book(target, "2026-07-27")

        assert stopped == ["ALGM"]
        assert out["ALGM"] == 0.0
        assert out["TLT"] == pytest.approx(0.08)   # NOT refilled
        assert sum(out.values()) == pytest.approx(0.08)

    def test_blocked_symbol_is_not_re_added(self, tmp_path):
        import run_all_agents as raa
        from ascent.config.settings import get_config
        from ascent.portfolio import stop_loss as sl

        state = str(tmp_path / "s.json")
        sl.record_stops(["ALGM"], "2026-07-20", path=state)

        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", True), \
             patch.object(sl, "DEFAULT_STATE_PATH", state), \
             patch("ascent.execution.alpaca_broker.get_positions",
                   return_value=pd.DataFrame(columns=[
                       "symbol", "qty", "market_value", "current_price",
                       "weight", "avg_entry_price", "unrealized_plpc"])):
            out, _ = raa._apply_stop_loss_to_book(target, "2026-07-27")

        assert out["ALGM"] == 0.0      # inside cooldown
        assert out["TLT"] == pytest.approx(0.08)

    def test_broker_failure_leaves_book_unchanged(self):
        """Fail-open: a broker outage must not silently liquidate the book."""
        import run_all_agents as raa
        from ascent.config.settings import get_config
        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", True), \
             patch("ascent.execution.alpaca_broker.get_positions",
                   side_effect=RuntimeError("broker down")):
            out, stopped = raa._apply_stop_loss_to_book(target, "2026-07-27")
        assert out == target
        assert stopped == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/strategy/test_stop_loss_wiring.py -v`
Expected: FAIL — `AttributeError: module 'run_all_agents' has no attribute '_apply_stop_loss_to_book'`.

- [ ] **Step 3: Add the production helper**

In `run_all_agents.py`, add near the other portfolio helpers:

```python
def _apply_stop_loss_to_book(target_weights: dict, today: str) -> tuple:
    """
    Apply the position-level stop-loss to a target book, using entry prices
    from the live Alpaca positions.

    Must be called LAST, after every cap and overlay: _water_fill_cap,
    enforce_cluster_cap, enforce_risk_budget_cap and apply_exposure_overlays
    all renormalize and would refill a stopped name.

    Fail-open: any failure (broker down, missing prices) returns the input
    unchanged. A monitoring failure must never liquidate the book.

    Returns (adjusted_weights, stopped_symbols).
    """
    # Local imports match this file's style: run_all_agents.py has no
    # module-level `logging` import (verified 2026-07-27 — the only use is a
    # function-local `import logging as _logging` at line 2616).
    import logging
    import pandas as pd
    from ascent.config.settings import get_config

    cfg = get_config()
    if not getattr(cfg.backtest, "stop_loss_enabled", False):
        return target_weights, []

    try:
        from ascent.portfolio.stop_loss import (
            DEFAULT_STATE_PATH, compute_stop_breaches, apply_stop_loss,
            load_stop_state, record_stops, blocked_symbols,
        )
        from ascent.execution import alpaca_broker

        w = pd.Series(target_weights, dtype=float)

        # 1. Names still inside their re-entry cooldown never get re-bought.
        state = load_stop_state(DEFAULT_STATE_PATH)
        blocked = blocked_symbols(
            state, today, cooldown_days=cfg.backtest.stop_loss_cooldown_days
        )
        if blocked:
            hit = [s for s in w.index if s in blocked]
            if hit:
                logging.info("[StopLoss] Cooldown blocks re-entry: %s", hit)
                w.loc[hit] = 0.0

        # 2. Evaluate live positions against their entry prices.
        pos = alpaca_broker.get_positions()
        stopped: list = []
        if pos is not None and not pos.empty and "avg_entry_price" in pos.columns:
            idx = pos.set_index("symbol")
            breached = compute_stop_breaches(
                idx["avg_entry_price"].astype(float),
                idx["current_price"].astype(float),
                threshold=cfg.backtest.stop_loss_threshold,
            )
            stopped = [s for s in breached.index if bool(breached[s])]
            if stopped:
                w = apply_stop_loss(
                    w,
                    pd.Series(True, index=[s for s in stopped if s in w.index]),
                    redistribute=cfg.backtest.stop_loss_redistribute,
                )
                record_stops(stopped, today, path=DEFAULT_STATE_PATH)
                logging.warning(
                    "[StopLoss] Stopped out %s at a %.0f%% threshold",
                    stopped, cfg.backtest.stop_loss_threshold * 100,
                )

        return w.to_dict(), stopped

    except Exception as exc:
        logging.warning(
            "[StopLoss] Skipped (%s) — book unchanged. Fail-open by design.",
            exc,
        )
        return target_weights, []
```

- [ ] **Step 4: Call it from the submission path**

Find the point where the final book is handed to `run_eod_with_weights(...)`. Insert immediately before that call, after all caps and overlays:

```python
    target_weights, _stopped_syms = _apply_stop_loss_to_book(
        target_weights, today_str
    )
```

Use whatever the surrounding code already calls the date string; do not introduce a new date source. Confirm with `grep -n "run_eod_with_weights" run_all_agents.py` that every submission path is covered — if there is more than one call site (there is at least the scheduled-rebalance path and the discovery mini-rebalance path), each needs the stop applied.

- [ ] **Step 5: Verify parse and run the wiring tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read())"
.venv/bin/python -m pytest tests/strategy/test_stop_loss_wiring.py -v
```

- [ ] **Step 6: Mirror into the walk-forward framework**

In `ascent/research/wf_framework/ascent_strategy.py`, after the risk-budget cap block, add:

```python
        # Position-level stop-loss — parity with production (see
        # docs/superpowers/plans/2026-07-27-position-stop-loss.md). Applied
        # on the DAILY panel, not the rebalance rows, because a stop has to
        # be able to fire between rebalances.
        try:
            from ascent.config.settings import get_config as _get_cfg2
            _sl = _get_cfg2().backtest
            _sl_on = getattr(_sl, "stop_loss_enabled", False)
        except Exception:
            _sl_on = False

        if _sl_on and _close_panel is not None and not weights_at_rebal.empty:
            try:
                from ascent.portfolio.stop_loss import apply_stop_loss_panel
                _daily = weights_at_rebal.reindex(
                    _close_panel.index, method="ffill"
                ).dropna(how="all")
                _daily, _events = apply_stop_loss_panel(
                    _daily, _close_panel,
                    threshold=_sl.stop_loss_threshold,
                    cooldown_days=_sl.stop_loss_cooldown_days,
                    redistribute=_sl.stop_loss_redistribute,
                )
                weights_at_rebal = _daily
                if _events:
                    # ascent_strategy.py has NO module-level `log` (verified
                    # 2026-07-27) — use a self-contained local logger.
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "[StopLoss/WF] %d stop events", len(_events))
            except Exception as _sl_e:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "[StopLoss/WF] skipped: %s", _sl_e)
```

Note this converts the returned frame from rebalance rows to daily rows. Confirm the downstream overlay calls (`_apply_200ma_overlay`, `_apply_vol_target`) and the engine accept a daily frame — they index by date and should. Run the WF smoke test after this change.

- [ ] **Step 7: Verify parse and run the broader suites**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/research/wf_framework/ascent_strategy.py').read())"
.venv/bin/python -m pytest tests/portfolio tests/strategy tests/backtest -v 2>&1 | tail -30
```
Expected: all green. Because `stop_loss_enabled` defaults to `False`, nothing behavioural should change for any existing test.

- [ ] **Step 8: Commit**

```bash
git add run_all_agents.py ascent/research/wf_framework/ascent_strategy.py tests/strategy/test_stop_loss_wiring.py
git commit -m "feat: wire position stop-loss into production and research

Applied last, after all caps and overlays, on the live book in production
and on the daily panel in walk-forward. Fail-open on broker failure.
Inert until stop_loss_enabled is turned on."
```

---

### Task 7: Validate on walk-forward, then decide whether to enable

**Files:**
- Create: `outputs/wf_results/wf_report_stoploss_2026-07-27.json`
- Modify: `ascent/config/settings.py` (only if the evidence supports enabling)
- Modify: `CURRENT_VERIFIED_NUMBERS.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-6, plus the corrected engine from the cash-bucket-fix plan.
- Produces: a decision, recorded in writing, on whether `stop_loss_enabled` becomes `True`.

**This task is a gate, not a formality.** The paper's result is not evidence about *this* book. `enforce_risk_budget_cap`, the cluster cap, and the momentum sleeve all interact with the stop, and the honest possibility is that it costs more in whipsaw than it saves in tails.

- [ ] **Step 1: Confirm the prerequisite actually landed**

```bash
.venv/bin/python -m pytest tests/backtest/test_engine_cash_bucket.py -v
```
Expected: 4 passed. If these fail, **stop** — the WF run will silently re-lever away every stop and the comparison is meaningless.

- [ ] **Step 2: Run walk-forward with the stop disabled (baseline)**

Same entrypoint and settings as `outputs/wf_results/wf_report_cashfix_2026-07-27.json`. If that artifact already exists from the prerequisite plan, reuse it rather than re-running.

- [ ] **Step 3: Run walk-forward with the stop enabled at 10%**

Temporarily set `stop_loss_enabled=True` in the run's config (do not commit that yet). Output to `outputs/wf_results/wf_report_stoploss_2026-07-27.json`.

- [ ] **Step 4: Run the threshold sweep**

Repeat at `stop_loss_threshold` in `{0.05, 0.10, 0.15, 0.20}`. Record Sharpe, CAGR, max drawdown, turnover, and stop-event count for each.

Turnover matters as much as return here: a stop that fires constantly pays spread and impact on every exit and re-entry. If turnover rises sharply while Sharpe is flat, the stop is churning, not protecting.

- [ ] **Step 5: Apply the decision rule**

Enable (`stop_loss_enabled = True`) only if **all** of these hold against the disabled baseline:
- max drawdown improves (less negative) by at least 2pp, **and**
- Sharpe does not fall by more than 0.05, **and**
- average turnover does not more than double.

Otherwise leave it disabled and write down why. A negative result is a real result — the machinery stays in the tree, dark, and the finding gets recorded.

Beware of tuning to the ALGM/MRNA episode. The threshold sweep exists to show the shape of the response, not to pick the value that best fits one month of one book. If 10% is within noise of the best value, keep 10% — it is the pre-registered value from the paper.

- [ ] **Step 6: Record the outcome**

Add a session-log entry to `CLAUDE.md` with the sweep table and the decision. If enabled, update `CURRENT_VERIFIED_NUMBERS.md` with the new WF figures and note that the stop is live.

- [ ] **Step 7: Commit**

```bash
git add -A outputs/wf_results/ CURRENT_VERIFIED_NUMBERS.md CLAUDE.md ascent/config/settings.py
git commit -m "test(portfolio): walk-forward validation of position stop-loss

Threshold sweep {5,10,15,20}% against the cash-bucket-corrected engine.
Decision and evidence recorded in CLAUDE.md."
```

---

## Self-Review

- **Spec coverage:** breach rule (Task 1), cash-vs-redistribute (Task 1), cooldown (Task 2), research entry-price reconstruction (Task 3), the measured ALGM/MRNA target (Task 3), config surface (Task 4), the missing broker field discovered on 2026-07-27 (Task 5), both wiring paths with the apply-last ordering constraint (Task 6), and an explicit enable/reject gate (Task 7).
- **Placeholders:** none. Every code step carries literal code; Task 5 Step 2 and Task 6 Step 4 tell the implementer to read the surrounding code and adapt rather than guessing at a call signature.
- **Type consistency:** `compute_stop_breaches` returns a bool `pd.Series` and is consumed as one in `apply_stop_loss`, `apply_stop_loss_panel`, and `_apply_stop_loss_to_book`. `apply_stop_loss_panel` returns `(DataFrame, list[dict])` and is unpacked that way at both call sites. `DEFAULT_STATE_PATH` is the patch target in tests and the default argument everywhere.
- **Known limitation, stated not hidden:** a 10% stop does not prevent a gap-through. ALGM fell from −10% to −16.4% in a single session, so the realized exit was −16.4%, not −10%. The plan claims +1.04pp on this episode, which is what was measured, not the −10% the threshold nominally implies.
