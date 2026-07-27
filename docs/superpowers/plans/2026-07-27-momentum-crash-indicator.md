# Momentum Crash Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut portfolio exposure when the market is in the specific state that precedes momentum crashes — a prolonged decline followed by a rebound — implementing the conditional mechanism from Daniel & Moskowitz (2016), "Momentum Crashes."

**Architecture:** A third, independent exposure overlay alongside the 200MA cut and vol targeting, living in `ascent/portfolio/exposure.py`. It is a pure function of the SPY series (bear state AND rebound state, both causal) returning a per-date multiplier, composed multiplicatively with the existing two. Ships disabled.

**Tech Stack:** Python 3.12, pandas, numpy, pytest.

## Global Constraints

- **PREREQUISITE:** `docs/superpowers/plans/2026-07-27-backtest-cash-bucket-fix.md` must land first. Without it, `BacktestEngine` re-levers a de-risked book to gross 1.0 the next day and this overlay cannot be validated.
- **SEQUENCING:** this plan and `2026-07-27-strategy-own-vol-targeting.md` both modify `ascent/portfolio/exposure.py`, specifically both touch `apply_exposure_overlays`. **Land the vol-targeting plan first**, then this one — this plan's Task 2 assumes `vol_reference` and `close` already exist on that signature. If vol-targeting is skipped, drop those two parameters from Task 2's diff.
- Always use `.venv/bin/python`. Never the system Python.
- Use `import logging`; **never** `from loguru import logger` (loguru is not installed).
- Config access via `get_config()` — never `Config()` directly.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- Ships **disabled** (`momentum_crash_overlay_enabled = False`). Enabling is a separate evidence-gated decision (Task 4).
- Causal: only SPY data strictly before each decision date.
- Overlays never raise. On failure, return an all-ones scale and log.
- Mirror into **both** `ascent/main.py` and `ascent/research/wf_framework/ascent_strategy.py`.

---

## Evidence, and an honest statement of what this does not fix

**Daniel & Moskowitz (2016), Journal of Financial Economics:** momentum crashes are not random. They cluster in a specific, identifiable state — panic periods following market declines, when the market rebounds. In that state the momentum portfolio's short leg (or, for a long-only book, the low-momentum names it has rotated out of) rallies violently, and high-momentum winners carry elevated beta acquired during the decline. Their dynamic strategy scales exposure by forecast momentum mean and variance, more than doubling the static momentum factor's Sharpe ratio and beating the constant-volatility variant of Barroso & Santa-Clara.

**This plan implements the conditional state indicator, not the full forecasting model.** Daniel & Moskowitz scale continuously by a forecast conditional Sharpe ratio. That requires estimating conditional mean and variance of the momentum factor, which is a research project with real overfitting risk on this book's short history. The bear-AND-rebound state indicator is the robust, low-parameter core of the same insight, and it is what this plan builds. Extending to the full dynamic weighting is deliberately deferred.

**It would not have prevented the ALGM/MRNA loss.** The regime engine labelled the entire 2026-06/07 window `calm_bull`; there was no two-year market decline, so the bear condition would not have fired. This overlay protects against a rarer and much larger failure mode than the one that just happened — the 2009-style momentum crash where a factor loses tens of percent in weeks. Of the three risk plans it is the least likely to help with the immediate problem and the most likely to matter in the worst case. Sequence it last; do not skip it.

---

## File Structure

- `ascent/portfolio/exposure.py` — add `momentum_crash_scale()`; compose it into `apply_exposure_overlays()`.
- `ascent/config/settings.py` — two new `BacktestConfig` fields.
- `ascent/main.py` — pass the flags through.
- `ascent/research/wf_framework/ascent_strategy.py` — mirror.
- `tests/portfolio/test_momentum_crash_overlay.py` — **new**.

---

### Task 1: The crash-state indicator

**Files:**
- Modify: `ascent/portfolio/exposure.py`
- Create: `tests/portfolio/test_momentum_crash_overlay.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CRASH_BEAR_LOOKBACK: int = 504`, `CRASH_REBOUND_LOOKBACK: int = 21`, `CRASH_MULTIPLIER: float = 0.50`
  - `momentum_crash_scale(spy_close: pd.Series, dates: pd.Index, bear_lookback: int = CRASH_BEAR_LOOKBACK, rebound_lookback: int = CRASH_REBOUND_LOOKBACK, multiplier: float = CRASH_MULTIPLIER) -> pd.Series`

- [ ] **Step 1: Write the failing test**

Create `tests/portfolio/test_momentum_crash_overlay.py`:

```python
# tests/portfolio/test_momentum_crash_overlay.py
"""
Momentum crash indicator — Daniel & Moskowitz (2016).

Momentum crashes cluster in panic states: a prolonged market decline
followed by a rebound. In that state, cut exposure.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.exposure import (
    CRASH_MULTIPLIER,
    momentum_crash_scale,
)


def _series(values, start="2023-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _path(n_down: int, down_rate: float, n_up: int, up_rate: float):
    """A decline of n_down days then a rebound of n_up days."""
    vals = [100.0]
    for _ in range(n_down):
        vals.append(vals[-1] * (1 + down_rate))
    for _ in range(n_up):
        vals.append(vals[-1] * (1 + up_rate))
    return _series(vals)


class TestMomentumCrashScale:
    def test_bear_plus_rebound_cuts_exposure(self):
        """The crash state: 2y cumulative negative, recent window positive."""
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), bear_lookback=504,
                                   rebound_lookback=21, multiplier=0.5)
        assert out.iloc[0] == pytest.approx(0.5)

    def test_bear_without_rebound_does_not_cut(self):
        """Still falling is not the crash state — the 200MA cut owns that."""
        spy = _path(n_down=600, down_rate=-0.001, n_up=0, up_rate=0.0)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), bear_lookback=504,
                                   rebound_lookback=21, multiplier=0.5)
        assert out.iloc[0] == pytest.approx(1.0)

    def test_rebound_without_bear_does_not_cut(self):
        """A rally in an ongoing bull is not a crash state."""
        spy = _path(n_down=0, down_rate=0.0, n_up=600, up_rate=0.001)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), bear_lookback=504,
                                   rebound_lookback=21, multiplier=0.5)
        assert out.iloc[0] == pytest.approx(1.0)

    def test_calm_bull_is_never_cut(self):
        """Regression for the 2026-06/07 window: this must NOT fire."""
        rng = np.random.default_rng(4)
        spy = _series(100 * np.cumprod(
            1 + rng.normal(0.0004, 0.008, 700)))
        out = momentum_crash_scale(spy, spy.index[-30:], multiplier=0.5)
        assert (out == 1.0).all()

    def test_is_causal_future_data_does_not_leak(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        d = spy.index[400]
        base = momentum_crash_scale(spy, pd.Index([d])).iloc[0]

        tampered = spy.copy()
        tampered.iloc[401:] = tampered.iloc[401:] * 5.0
        after = momentum_crash_scale(tampered, pd.Index([d])).iloc[0]
        assert base == pytest.approx(after)

    def test_insufficient_history_returns_one(self):
        spy = _series([100.0, 99.0, 101.0])
        out = momentum_crash_scale(spy, spy.index, bear_lookback=504)
        assert (out == 1.0).all()

    def test_multiplier_is_configurable(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), multiplier=0.25)
        assert out.iloc[0] == pytest.approx(0.25)

    def test_multiplier_one_is_a_noop(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        out = momentum_crash_scale(spy, spy.index[-5:], multiplier=1.0)
        assert (out == 1.0).all()

    def test_empty_dates_returns_empty(self):
        spy = _path(n_down=10, down_rate=-0.001, n_up=5, up_rate=0.002)
        assert momentum_crash_scale(spy, pd.Index([])).empty

    def test_duplicate_index_entries_are_tolerated(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        dupd = pd.concat([spy, spy.iloc[-3:]]).sort_index()
        out = momentum_crash_scale(dupd, pd.Index([spy.index[-1]]))
        assert out.iloc[0] in (0.5, CRASH_MULTIPLIER)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_momentum_crash_overlay.py -v`
Expected: FAIL — `ImportError: cannot import name 'CRASH_MULTIPLIER'`.

- [ ] **Step 3: Write the implementation**

Append to `ascent/portfolio/exposure.py`:

```python
CRASH_BEAR_LOOKBACK    = 504   # ~2 trading years
CRASH_REBOUND_LOOKBACK = 21    # ~1 trading month
CRASH_MULTIPLIER       = 0.50


def momentum_crash_scale(
    spy_close: pd.Series,
    dates: pd.Index,
    bear_lookback: int = CRASH_BEAR_LOOKBACK,
    rebound_lookback: int = CRASH_REBOUND_LOOKBACK,
    multiplier: float = CRASH_MULTIPLIER,
) -> pd.Series:
    """
    Per-date exposure multiplier for the momentum-crash state.

    Daniel & Moskowitz (2016): momentum crashes cluster in panic periods —
    a prolonged market decline followed by a rebound. Winners carry beta
    acquired during the decline, and the rebound repriced them violently.

    Fires `multiplier` when BOTH hold, using only data strictly before `d`:
      * bear:    cumulative SPY return over the trailing `bear_lookback` < 0
      * rebound: SPY return over the trailing `rebound_lookback` > 0
    Otherwise 1.0.

    Deliberately NOT the full dynamic model. Daniel & Moskowitz scale
    continuously by a forecast conditional Sharpe ratio; estimating that on
    this book's history is an overfitting risk. This is the low-parameter
    state indicator at the core of the same result.

    Distinct from the 200MA cut, which fires on "market is below trend".
    This fires on "market fell for a long time and is now bouncing" — the
    200MA filter is typically OFF in exactly that state, which is the point.
    """
    if len(dates) == 0:
        return pd.Series(dtype=float)
    if multiplier >= 1.0:
        return pd.Series(1.0, index=dates)

    try:
        s = spy_close.sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
    except Exception as exc:
        log.warning("[Exposure] momentum_crash_scale: bad SPY series (%s) "
                    "— no cut applied", exc)
        return pd.Series(1.0, index=dates)

    scales = []
    for d in dates:
        past = s[s.index < d]
        if len(past) < bear_lookback + 1:
            scales.append(1.0)
            continue

        bear_window = past.iloc[-(bear_lookback + 1):]
        bear_ret = float(bear_window.iloc[-1] / bear_window.iloc[0] - 1.0)

        reb_window = past.iloc[-(rebound_lookback + 1):]
        reb_ret = float(reb_window.iloc[-1] / reb_window.iloc[0] - 1.0)

        scales.append(float(multiplier) if (bear_ret < 0.0 and reb_ret > 0.0)
                      else 1.0)

    out = pd.Series(scales, index=dates)
    n_cut = int((out < 1.0).sum())
    if n_cut:
        log.info("[Exposure] Momentum-crash state on %d/%d dates — exposure "
                 "x%.2f (Daniel & Moskowitz 2016)", n_cut, len(out), multiplier)
    return out
```

- [ ] **Step 4: Verify parse and run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/exposure.py').read())"
.venv/bin/python -m pytest tests/portfolio/test_momentum_crash_overlay.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ascent/portfolio/exposure.py tests/portfolio/test_momentum_crash_overlay.py
git commit -m "feat(portfolio): add momentum-crash state indicator

Daniel & Moskowitz (2016): cut exposure when the market has declined over
~2y AND is rebounding. Causal, low-parameter, not yet wired in."
```

---

### Task 2: Compose into `apply_exposure_overlays`

**Files:**
- Modify: `ascent/portfolio/exposure.py` (`apply_exposure_overlays` + module docstring)
- Modify: `ascent/config/settings.py`
- Modify: `tests/portfolio/test_momentum_crash_overlay.py`
- Create: `tests/test_momentum_crash_config.py`

**Interfaces:**
- Consumes: `momentum_crash_scale`.
- Produces: `apply_exposure_overlays(..., crash_overlay_enabled: bool = False, crash_multiplier: float = CRASH_MULTIPLIER)`; `meta` gains `"crash_cut_dates": int`. Config gains `momentum_crash_overlay_enabled: bool = False` and `momentum_crash_multiplier: float = 0.50`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/portfolio/test_momentum_crash_overlay.py`:

```python
from ascent.portfolio.exposure import apply_exposure_overlays


class TestCrashOverlayComposition:
    def _crash_market(self):
        vals = [100.0]
        for _ in range(560):
            vals.append(vals[-1] * 0.999)
        for _ in range(25):
            vals.append(vals[-1] * 1.004)
        idx = pd.bdate_range("2023-01-01", periods=len(vals))
        return pd.Series(vals, index=idx, dtype=float)

    def test_disabled_by_default_changes_nothing(self):
        spy = self._crash_market()
        w = pd.DataFrame(0.5, index=spy.index[-30:], columns=["A", "B"])
        a, meta_a = apply_exposure_overlays(w, spy)
        b, _ = apply_exposure_overlays(w, spy, crash_overlay_enabled=False)
        pd.testing.assert_frame_equal(a, b)
        assert meta_a["crash_cut_dates"] == 0

    def test_enabled_cuts_exposure_in_the_crash_state(self):
        spy = self._crash_market()
        w = pd.DataFrame(0.5, index=spy.index[-30:], columns=["A", "B"])
        off, _ = apply_exposure_overlays(w, spy, rebalance_only=False)
        on, meta = apply_exposure_overlays(
            w, spy, crash_overlay_enabled=True, crash_multiplier=0.5,
            rebalance_only=False,
        )
        assert meta["crash_cut_dates"] > 0
        assert on.iloc[-1].sum() < off.iloc[-1].sum()

    def test_composes_multiplicatively_with_other_overlays(self):
        """Halving via the crash overlay must halve the final book."""
        spy = self._crash_market()
        w = pd.DataFrame(0.5, index=spy.index[-30:], columns=["A", "B"])
        off, _ = apply_exposure_overlays(w, spy, rebalance_only=False)
        on, _ = apply_exposure_overlays(
            w, spy, crash_overlay_enabled=True, crash_multiplier=0.5,
            rebalance_only=False,
        )
        ratio = on.iloc[-1].sum() / off.iloc[-1].sum()
        assert ratio == pytest.approx(0.5, rel=1e-9)
```

Create `tests/test_momentum_crash_config.py`:

```python
# tests/test_momentum_crash_config.py
"""Momentum-crash overlay config. Ships DISABLED pending validation."""
from ascent.config.settings import get_config


def test_crash_overlay_ships_disabled():
    bt = get_config().backtest
    assert bt.momentum_crash_overlay_enabled is False
    assert bt.momentum_crash_multiplier == 0.50
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/portfolio/test_momentum_crash_overlay.py::TestCrashOverlayComposition tests/test_momentum_crash_config.py -v
```
Expected: `TypeError: ... unexpected keyword argument 'crash_overlay_enabled'` and `AttributeError` on the config field.

- [ ] **Step 3: Add the config fields**

In `ascent/config/settings.py`, after the other risk-construction fields:

```python
    # Momentum-crash overlay (Daniel & Moskowitz 2016). Cuts exposure when
    # the market has declined over ~2y AND is rebounding — the state in which
    # momentum crashes cluster. DISABLED by default; see
    # docs/superpowers/plans/2026-07-27-momentum-crash-indicator.md.
    momentum_crash_overlay_enabled: bool = False
    momentum_crash_multiplier: float = 0.50
```

- [ ] **Step 4: Compose the overlay**

Add to the end of `apply_exposure_overlays`'s signature:

```python
    crash_overlay_enabled: bool = False,
    crash_multiplier: float = CRASH_MULTIPLIER,
```

After `combined = ma_scale * v_scale`, insert:

```python
    if crash_overlay_enabled:
        c_scale = momentum_crash_scale(spy_close, dates,
                                       multiplier=crash_multiplier)
    else:
        c_scale = pd.Series(1.0, index=dates)
    combined = combined * c_scale
```

Add to `meta`:

```python
        "crash_cut_dates": int((c_scale < 1.0).sum()),
```

Update the module docstring's overlay list from two entries to three:

```
  3. momentum_crash_scale() — 2y decline + rebound (Daniel & Moskowitz) → x0.50
```

- [ ] **Step 5: Verify parse and run the whole portfolio suite**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/exposure.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('ascent/config/settings.py').read())"
.venv/bin/python -m pytest tests/portfolio tests/test_momentum_crash_config.py -v
```
Expected: all pass, `tests/portfolio/test_exposure.py` still unedited.

- [ ] **Step 6: Commit**

```bash
git add ascent/portfolio/exposure.py ascent/config/settings.py tests/portfolio/test_momentum_crash_overlay.py tests/test_momentum_crash_config.py
git commit -m "feat(portfolio): compose momentum-crash overlay into exposure stack

Third multiplier alongside the 200MA cut and vol targeting. Disabled by
default, so behaviour is unchanged."
```

---

### Task 3: Wire production and research

**Files:**
- Modify: `ascent/main.py:825` (the `apply_exposure_overlays` call)
- Modify: `ascent/research/wf_framework/ascent_strategy.py` (`_apply_200ma_overlay` / `_apply_vol_target` neighbourhood, ~lines 289-380)
- Create: `tests/portfolio/test_crash_overlay_parity.py`

**Interfaces:**
- Consumes: the config fields and the composed overlay from Task 2.
- Produces: no new public symbols — production and research read the same two config values.

- [ ] **Step 1: Write the failing parity test**

Create `tests/portfolio/test_crash_overlay_parity.py`:

```python
# tests/portfolio/test_crash_overlay_parity.py
"""
Research and production must read the SAME crash-overlay config.

Precedent: ascent/portfolio/exposure.py exists because research had vol
targeting and production did not, and the two silently diverged.
"""
import inspect

from ascent.config.settings import get_config


def test_both_paths_reference_the_config_flag():
    import ascent.main as main_mod
    from ascent.research.wf_framework import ascent_strategy

    prod = inspect.getsource(main_mod)
    research = inspect.getsource(ascent_strategy)

    assert "momentum_crash_overlay_enabled" in prod, (
        "ascent/main.py must pass the crash-overlay flag through"
    )
    assert "momentum_crash_overlay_enabled" in research, (
        "the WF strategy must read the same flag or research and production "
        "diverge silently"
    )


def test_config_values_are_sane():
    bt = get_config().backtest
    assert 0.0 < bt.momentum_crash_multiplier <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_crash_overlay_parity.py -v`
Expected: FAIL — the flag name appears in neither module.

- [ ] **Step 3: Wire production**

In `ascent/main.py`, extend the existing `apply_exposure_overlays(...)` call:

```python
            crash_overlay_enabled=cfg.backtest.momentum_crash_overlay_enabled,
            crash_multiplier=cfg.backtest.momentum_crash_multiplier,
```

- [ ] **Step 4: Wire research**

In `ascent/research/wf_framework/ascent_strategy.py`, after the existing `_apply_vol_target` call (~line 295), add a third overlay step:

```python
        # --- Step 6: Momentum-crash overlay (parity with production) ---
        weights_at_rebal = self._apply_momentum_crash_overlay(
            data, weights_at_rebal
        )
```

and add the method, modelled on the existing `_apply_200ma_overlay`:

```python
    def _apply_momentum_crash_overlay(self, data, weights):
        """
        Daniel & Moskowitz (2016) crash-state cut. Delegates to
        ascent/portfolio/exposure.py — single source of truth shared with
        production. See docs/superpowers/plans/2026-07-27-momentum-crash-indicator.md.
        """
        if weights is None or weights.empty:
            return weights
        try:
            from ascent.config.settings import get_config as _gc
            _bt = _gc().backtest
            if not getattr(_bt, "momentum_crash_overlay_enabled", False):
                return weights
            mult = float(getattr(_bt, "momentum_crash_multiplier", 0.50))
        except Exception:
            return weights

        try:
            from ascent.portfolio.exposure import momentum_crash_scale
            spy = self._benchmark_close(data)
            if spy is None or spy.empty:
                return weights
            scale = momentum_crash_scale(spy, weights.index, multiplier=mult)
            return weights.mul(scale, axis=0)
        except Exception as exc:
            # ascent_strategy.py has NO module-level `log` (verified
            # 2026-07-27) — use a self-contained local logger.
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "[Exposure/WF] momentum-crash overlay skipped: %s", exc)
            return weights
```

`self._benchmark_close(data)` is a placeholder for however `_apply_200ma_overlay` already obtains the SPY close series in this class. **Read that method first** (`sed -n '305,340p' ascent/research/wf_framework/ascent_strategy.py`) and reuse its exact mechanism — do not add a second way of loading SPY.

- [ ] **Step 5: Verify parse and run**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/main.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('ascent/research/wf_framework/ascent_strategy.py').read())"
.venv/bin/python -m pytest tests/portfolio tests/test_momentum_crash_config.py -v
```
Expected: all pass. Disabled by default, so no behaviour changes.

- [ ] **Step 6: Commit**

```bash
git add ascent/main.py ascent/research/wf_framework/ascent_strategy.py tests/portfolio/test_crash_overlay_parity.py
git commit -m "feat: wire momentum-crash overlay through production and research

Both paths read the same config flags. Inert until enabled."
```

---

### Task 4: Historical firing audit, then the enable decision

**Files:**
- Create: `outputs/wf_results/wf_report_crashoverlay_2026-07-27.json`
- Modify: `ascent/config/settings.py` (only if the evidence supports enabling)
- Modify: `CLAUDE.md`

**The specific risk here is a rule that never fires.** With `bear_lookback=504` on a 2021-2026 OOS window, the crash state may occur on very few dates — possibly zero. A rule that never fires is neither validated nor refuted, and enabling it on that basis would be false comfort. Audit the firing rate **before** running the walk-forward.

- [ ] **Step 1: Confirm the prerequisite landed**

```bash
.venv/bin/python -m pytest tests/backtest/test_engine_cash_bucket.py -v
```
Expected: 4 passed.

- [ ] **Step 2: Count firing dates over the OOS window**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from ascent.data.store.parquet import load_parquet, has_data
from ascent.portfolio.exposure import momentum_crash_scale

assert has_data("prices_live"), "no prices_live cache"
df = load_parquet("prices_live")
spy = (df[df["symbol"] == "SPY"]
       .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
       .sort_values("date").drop_duplicates("date", keep="last")
       .set_index("date")["close"].astype(float))

dates = spy.index[spy.index >= "2021-01-01"]
scale = momentum_crash_scale(spy, dates, multiplier=0.5)
fired = scale[scale < 1.0]
print(f"OOS dates: {len(dates)}   crash-state dates: {len(fired)} "
      f"({len(fired)/max(len(dates),1):.1%})")
if len(fired):
    print("first:", fired.index.min().date(), " last:", fired.index.max().date())
PY
```

Record the output verbatim in the session log.

- [ ] **Step 3: Branch on the firing rate**

- **0 firing dates:** the rule is untestable on this window. Do **not** enable. Record the result, leave the code in the tree, and note that validation needs a longer history (the mechanism is documented back to 1932 in the paper, so a pre-2021 sample would test it). Skip to Step 6.
- **1-5 firing dates:** too few to distinguish signal from luck. Do not enable on this evidence alone; report the per-episode effect descriptively and move on.
- **>5 firing dates:** proceed to the walk-forward comparison.

- [ ] **Step 4: Run the walk-forward comparison**

Baseline (`momentum_crash_overlay_enabled=False`) versus treatment (`True`, multiplier 0.50), changing nothing else. Output the treatment run to `outputs/wf_results/wf_report_crashoverlay_2026-07-27.json`.

- [ ] **Step 5: Apply the decision rule**

Enable only if, against the baseline:
- max drawdown improves by at least 3pp, **and**
- Sharpe does not fall by more than 0.05, **and**
- the improvement is not driven by a single episode — check that removing the largest single firing window still leaves drawdown improved.

That last condition matters most: with a rare indicator, one lucky episode can carry the whole result.

- [ ] **Step 6: Record the outcome and commit**

Add a `CLAUDE.md` session-log entry with the firing count, the comparison (or the reason it was not run), and the decision.

```bash
git add -A outputs/wf_results/ CLAUDE.md ascent/config/settings.py
git commit -m "test(portfolio): momentum-crash overlay firing audit and decision"
```

---

## Self-Review

- **Spec coverage:** the bear-AND-rebound state indicator (Task 1), composition into the existing overlay stack with a disabled default (Task 2), both wiring paths with an explicit parity guard (Task 3), and a firing-rate audit gating the enable decision (Task 4).
- **Placeholders:** none. Task 3 Step 4 flags `self._benchmark_close(data)` as a placeholder **in prose** and instructs the implementer to read `_apply_200ma_overlay` and reuse its mechanism, rather than leaving an undefined call to be guessed at.
- **Type consistency:** `momentum_crash_scale(spy_close: pd.Series, dates: pd.Index) -> pd.Series` is consumed as a Series in `apply_exposure_overlays` (multiplied into `combined`) and in the WF method (`weights.mul(scale, axis=0)`). `CRASH_MULTIPLIER` is the default in both the function and the `apply_exposure_overlays` parameter.
- **Known limitation, stated up front:** this would not have fired during the 2026-06/07 episode, and it may not fire at all on the current OOS window. Task 4 Step 3 makes "zero firings, do not enable" an explicit, acceptable outcome rather than a reason to loosen the thresholds until something happens.
