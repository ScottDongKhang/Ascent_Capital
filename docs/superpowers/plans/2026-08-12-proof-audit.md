# Proof Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone scorer that produces a KEEP / CUT / INSUFFICIENT_DATA verdict, with
a p-value and sample size, for every alpha sleeve, specialist-agent alpha builder, and named
subsystem in the live Ascent Capital pipeline — so the rebuild (sub-projects 2-4) only carries
forward what's proven.

**Architecture:** Two independent scorers writing rows into one scorecard JSON. Path A
(`wf_scorer.py`) computes out-of-sample daily cross-sectional IC (Spearman) between each pure
alpha function's signal and forward returns, using point-in-time universe per date — no
walk-forward-framework reuse, no retraining loop. Path B (`counterfactual_scorer.py`) computes
return deltas from the existing five-track counterfactual data plus new synthetic with/without
tracks for subsystems that don't have one yet. `scorecard.py` applies one shared verdict rule to
both paths' outputs.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`, pandas, scipy.stats (t-test), pytest.

## Global Constraints

- Always use `.venv/bin/python`. Never bare `python`.
- No look-ahead: forward returns and universe membership must respect
  `get_universe_on_date(date, universe_df=None)` (`ascent/data/universe.py:533`) — a signal
  computed on date T is only ever compared against the return from T to T+1, never against
  information that includes T+1 in its inputs.
- `import logging`; never `from loguru import logger`.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- This plan creates **no new production write paths** — nothing here changes `ai_weight`,
  `PROMOTION_CONFIG`, `DEFAULT_ALPHA_WEIGHTS`, or any execution code. Output is read-only:
  `outputs/analyst/proof_audit_<date>.json`.
- Component list in `components.py` is a **pinned fixture** — never populated by dynamically
  scanning `ascent/alpha/stack.py` at runtime. Adding a component means editing the fixture.
- Verdict is always one of `KEEP` / `CUT` / `INSUFFICIENT_DATA` — a component with too little
  data must never silently resolve to `KEEP` or `CUT`.

---

## Known scope limits (decided during design, not deferred TBDs)

Three alpha sleeves — `ml`, `llm_fundamental`, `narrative` — are excluded from Path A's live
re-simulation. `ml` needs a model retrained per fold (that's the walk-forward-framework's job,
which this plan deliberately doesn't reuse or trust); `llm_fundamental` and `narrative` call an
LLM, so re-running them historically is both expensive and non-deterministic — a re-run today
would not reproduce the signal that was actually live on a past date. These three are registered
in `components.py` with `method="deferred"` and always score `INSUFFICIENT_DATA` with reason
`"requires live-logged signal history, not re-simulation — out of scope for this audit"`. This is
a disclosed scope decision, not a gap to silently paper over.

`agents.us_equities_agent` is not scored as its own row: it calls `ascent/main.py::run_pipeline`,
which is exactly the alpha-stack pipeline Path A already scores sleeve-by-sleeve. Scoring it again
as a fourth "agent" would double-count the same signal under a different name. It's registered in
`components.py` with `method="covered_by_sleeves"` and a `verdict()` that always returns
`INSUFFICIENT_DATA` with reason `"covered by per-sleeve rows; not scored standalone"` — visible in
the scorecard, not silently dropped from the component list.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ascent/analyst/proof_audit/__init__.py` | package marker | 1 |
| `ascent/analyst/proof_audit/components.py` | pinned component fixture | 1 |
| `ascent/analyst/proof_audit/stats.py` | IC/t-test/Sharpe math, no I/O | 2 |
| `ascent/analyst/proof_audit/forward_returns.py` | point-in-time forward-return + fold assembly | 3 |
| `ascent/analyst/proof_audit/sleeve_signals.py` | registry of pure sleeve-signal callables | 4 |
| `ascent/analyst/proof_audit/agent_signals.py` | registry of macro/international/alternatives builders | 5 |
| `ascent/analyst/proof_audit/wf_scorer.py` | Path A: ties signals + forward returns + stats together | 4, 5 |
| `ascent/analyst/proof_audit/counterfactual_scorer.py` | Path B: subsystem return-delta scoring | 6 |
| `ascent/analyst/proof_audit/scorecard.py` | verdict rule + JSON writer | 7 |
| `ascent/analyst/proof_audit/run.py` | CLI entrypoint, wires A + B → scorecard | 7 |
| `tests/analyst/proof_audit/*` | one test file per module above | 1-8 |

---

## Task 1: Package skeleton and pinned component fixture

**Files:**
- Create: `ascent/analyst/proof_audit/__init__.py` (empty)
- Create: `ascent/analyst/proof_audit/components.py`
- Test: `tests/analyst/proof_audit/__init__.py` (empty), `tests/analyst/proof_audit/test_components.py`

**Interfaces:**
- Produces: `Component` dataclass (`name: str`, `kind: str`, `method: str`), `COMPONENTS: list[Component]`, `get_component(name: str) -> Component` (raises `KeyError` on unknown name).

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/proof_audit/test_components.py`:

```python
"""The component list is a pinned fixture -- every entry must resolve to a known method."""
import pytest

from ascent.analyst.proof_audit.components import COMPONENTS, get_component

VALID_METHODS = {"wf_ic", "counterfactual", "deferred", "covered_by_sleeves"}

EXPECTED_SLEEVES = {
    "trend", "meanrev", "volatility", "statarb", "fundamental",
    "earnings", "analyst", "options_flow", "insider", "short_interest",
    "altdata", "earnings_tone",
}
EXPECTED_DEFERRED_SLEEVES = {"ml", "llm_fundamental", "narrative"}
EXPECTED_AGENTS = {"macro_agent", "international_agent", "alternatives_agent", "us_equities_agent"}
EXPECTED_SUBSYSTEMS = {
    "regime_overlay", "hedge_overlay", "earned_authority", "debate_judge_intervention",
}


def test_every_component_has_valid_method():
    for c in COMPONENTS:
        assert c.method in VALID_METHODS, f"{c.name} has unknown method {c.method!r}"


def test_names_are_unique():
    names = [c.name for c in COMPONENTS]
    assert len(names) == len(set(names))


def test_expected_sleeves_present_as_wf_ic():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_SLEEVES:
        assert by_name[name].kind == "alpha_sleeve"
        assert by_name[name].method == "wf_ic"


def test_deferred_sleeves_present():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_DEFERRED_SLEEVES:
        assert by_name[name].method == "deferred"


def test_agents_present():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_AGENTS:
        assert by_name[name].kind == "agent"
    assert by_name["us_equities_agent"].method == "covered_by_sleeves"
    for name in EXPECTED_AGENTS - {"us_equities_agent"}:
        assert by_name[name].method == "wf_ic"


def test_subsystems_present_as_counterfactual():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_SUBSYSTEMS:
        assert by_name[name].kind == "subsystem"
        assert by_name[name].method == "counterfactual"


def test_get_component_raises_on_unknown():
    with pytest.raises(KeyError):
        get_component("does_not_exist")


def test_get_component_returns_match():
    c = get_component("trend")
    assert c.name == "trend"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_components.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ascent.analyst.proof_audit'`

- [ ] **Step 3: Create the package and fixture**

```bash
mkdir -p ascent/analyst/proof_audit tests/analyst/proof_audit
touch ascent/analyst/proof_audit/__init__.py tests/analyst/proof_audit/__init__.py
```

Create `ascent/analyst/proof_audit/components.py`:

```python
"""Pinned component fixture for the proof audit.

Never populate this by scanning ascent/alpha/stack.py or agents/ at runtime -- an audit that
silently drops a component because discovery missed it is worse than no audit. Add a component
by editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    name: str
    kind: str    # "alpha_sleeve" | "agent" | "subsystem"
    method: str  # "wf_ic" | "counterfactual" | "deferred" | "covered_by_sleeves"


COMPONENTS: list[Component] = [
    # -- Alpha sleeves: pure functions of `features`, re-simulated day by day (Task 4) --
    Component("trend", "alpha_sleeve", "wf_ic"),
    Component("meanrev", "alpha_sleeve", "wf_ic"),
    Component("volatility", "alpha_sleeve", "wf_ic"),
    Component("statarb", "alpha_sleeve", "wf_ic"),
    Component("fundamental", "alpha_sleeve", "wf_ic"),
    Component("earnings", "alpha_sleeve", "wf_ic"),
    Component("analyst", "alpha_sleeve", "wf_ic"),
    Component("options_flow", "alpha_sleeve", "wf_ic"),
    Component("insider", "alpha_sleeve", "wf_ic"),
    Component("short_interest", "alpha_sleeve", "wf_ic"),
    Component("altdata", "alpha_sleeve", "wf_ic"),
    Component("earnings_tone", "alpha_sleeve", "wf_ic"),
    # -- Alpha sleeves excluded from re-simulation: retrained model or LLM-driven --
    Component("ml", "alpha_sleeve", "deferred"),
    Component("llm_fundamental", "alpha_sleeve", "deferred"),
    Component("narrative", "alpha_sleeve", "deferred"),
    # -- Specialist agents --
    Component("us_equities_agent", "agent", "covered_by_sleeves"),
    Component("macro_agent", "agent", "wf_ic"),
    Component("international_agent", "agent", "wf_ic"),
    Component("alternatives_agent", "agent", "wf_ic"),
    # -- Named subsystems: scored by counterfactual return delta (Task 6) --
    Component("regime_overlay", "subsystem", "counterfactual"),
    Component("hedge_overlay", "subsystem", "counterfactual"),
    Component("earned_authority", "subsystem", "counterfactual"),
    Component("debate_judge_intervention", "subsystem", "counterfactual"),
]

_BY_NAME = {c.name: c for c in COMPONENTS}


def get_component(name: str) -> Component:
    if name not in _BY_NAME:
        raise KeyError(f"unknown component {name!r}; known: {sorted(_BY_NAME)}")
    return _BY_NAME[name]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/components.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_components.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/analyst/proof_audit/__init__.py ascent/analyst/proof_audit/components.py \
        tests/analyst/proof_audit/__init__.py tests/analyst/proof_audit/test_components.py
git commit -m "feat(proof-audit): pinned component fixture for keep/cut scoring

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Statistics core (IC, t-test, Sharpe) — no I/O

**Files:**
- Create: `ascent/analyst/proof_audit/stats.py`
- Test: `tests/analyst/proof_audit/test_stats.py`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces: `ICResult` dataclass (`ic_mean: float`, `ic_t: float`, `p_value: float`,
  `sharpe: float`, `n: int`); `score_ic_series(daily_ic: list[float], daily_ls_return: list[float]) -> ICResult`.
  Task 4/5 build `daily_ic` and `daily_ls_return`; Task 7's verdict rule consumes `ICResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/proof_audit/test_stats.py`:

```python
"""Pin the IC/t-test/Sharpe math against a synthetic series with a known planted mean.

Real market data is never used here -- this only checks the arithmetic.
"""
import math

import pytest

from ascent.analyst.proof_audit.stats import score_ic_series


def test_positive_planted_ic_is_significant():
    # 60 days of IC centered at 0.05 with small noise -> should be clearly significant.
    daily_ic = [0.05 + 0.01 * math.sin(i) for i in range(60)]
    daily_ls_return = [0.001 + 0.0002 * math.sin(i) for i in range(60)]
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.ic_mean == pytest.approx(0.05, abs=0.01)
    assert result.p_value < 0.05
    assert result.n == 60


def test_zero_mean_ic_is_not_significant():
    daily_ic = [0.01 * math.sin(i) for i in range(60)]  # oscillates around 0
    daily_ls_return = [0.0001 * math.sin(i) for i in range(60)]
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.p_value > 0.05


def test_negative_planted_ic_is_significant_and_negative():
    daily_ic = [-0.05 + 0.01 * math.sin(i) for i in range(60)]
    daily_ls_return = [-0.001 + 0.0002 * math.sin(i) for i in range(60)]
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.ic_mean < 0
    assert result.p_value < 0.05


def test_sharpe_is_annualized():
    # constant positive daily return with zero variance is degenerate (std=0);
    # use a small planted variance instead so Sharpe is finite and computable.
    daily_ic = [0.03] * 40
    daily_ls_return = [0.001, 0.0008] * 20
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.sharpe > 0
    assert math.isfinite(result.sharpe)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        score_ic_series([0.01, 0.02], [0.001, 0.002])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        score_ic_series([0.01] * 10, [0.001] * 9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ascent.analyst.proof_audit.stats'`

- [ ] **Step 3: Implement**

Create `ascent/analyst/proof_audit/stats.py`:

```python
"""IC / significance / Sharpe math. Pure functions, no file or network I/O.

IC-t convention matches the one already used elsewhere in this repo (e.g. the fundamental
sleeve's disable comment: "IC=-0.015, IC-t=-4.75 across 31 live days").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as _scipy_stats

MIN_SAMPLE = 10
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ICResult:
    ic_mean: float
    ic_t: float
    p_value: float
    sharpe: float
    n: int


def score_ic_series(daily_ic: list[float], daily_ls_return: list[float]) -> ICResult:
    """Score a per-date IC series and a parallel long-short daily-return series.

    daily_ic[i] and daily_ls_return[i] must both describe the same trading date i --
    callers are responsible for that alignment (Task 3/4/5 build both from one date loop).
    """
    if len(daily_ic) != len(daily_ls_return):
        raise ValueError(
            f"daily_ic ({len(daily_ic)}) and daily_ls_return ({len(daily_ls_return)}) "
            "must be the same length"
        )
    n = len(daily_ic)
    if n < MIN_SAMPLE:
        raise ValueError(f"need at least {MIN_SAMPLE} points, got {n}")

    ic_mean = sum(daily_ic) / n
    t_stat, p_value = _scipy_stats.ttest_1samp(daily_ic, popmean=0.0)

    ret_mean = sum(daily_ls_return) / n
    variance = sum((r - ret_mean) ** 2 for r in daily_ls_return) / (n - 1)
    ret_std = math.sqrt(variance) if variance > 0 else float("nan")
    sharpe = (
        (ret_mean / ret_std) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if ret_std and not math.isnan(ret_std) and ret_std > 0
        else 0.0
    )

    return ICResult(
        ic_mean=float(ic_mean),
        ic_t=float(t_stat),
        p_value=float(p_value),
        sharpe=float(sharpe),
        n=n,
    )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/stats.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_stats.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/analyst/proof_audit/stats.py tests/analyst/proof_audit/test_stats.py
git commit -m "feat(proof-audit): IC/t-test/Sharpe scoring core, no I/O

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Point-in-time forward returns and date folds

**Files:**
- Create: `ascent/analyst/proof_audit/forward_returns.py`
- Test: `tests/analyst/proof_audit/test_forward_returns.py`

**Interfaces:**
- Consumes: `get_universe_on_date(date, universe_df=None)` from `ascent/data/universe.py:533`
- Produces: `forward_return_matrix(prices: pandas.DataFrame) -> pandas.DataFrame` (next-day simple
  return, same index/columns as `prices`, last row all-NaN because it has no next day);
  `eligible_dates(prices: pandas.DataFrame, min_universe_size: int = 20) -> list[pandas.Timestamp]`
  (dates whose point-in-time universe has at least `min_universe_size` symbols, excluding the
  final date). Task 4/5 iterate `eligible_dates(...)` and read rows out of
  `forward_return_matrix(...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/proof_audit/test_forward_returns.py`:

```python
"""Forward returns must be strictly next-day -- never same-day, never look back."""
import pandas as pd
import pytest

from ascent.analyst.proof_audit.forward_returns import (
    eligible_dates,
    forward_return_matrix,
)


def _toy_prices():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"AAA": [100, 102, 101, 105, 110], "BBB": [50, 49, 51, 52, 53]},
        index=dates,
    )


def test_forward_return_is_next_day_not_same_day():
    prices = _toy_prices()
    fwd = forward_return_matrix(prices)
    expected_day0_aaa = (102 - 100) / 100
    assert fwd.iloc[0]["AAA"] == pytest.approx(expected_day0_aaa)


def test_last_row_is_nan_no_lookahead():
    prices = _toy_prices()
    fwd = forward_return_matrix(prices)
    assert fwd.iloc[-1].isna().all()


def test_index_and_columns_match_input():
    prices = _toy_prices()
    fwd = forward_return_matrix(prices)
    assert list(fwd.index) == list(prices.index)
    assert list(fwd.columns) == list(prices.columns)


def test_eligible_dates_excludes_final_date(monkeypatch):
    prices = _toy_prices()

    def fake_universe(date, universe_df=None):
        return ["AAA", "BBB"] * 15  # 30 symbols, always eligible

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date", fake_universe
    )
    dates = eligible_dates(prices, min_universe_size=20)
    assert prices.index[-1] not in dates
    assert len(dates) == 4


def test_eligible_dates_respects_min_universe_size(monkeypatch):
    prices = _toy_prices()

    def fake_universe(date, universe_df=None):
        return ["AAA"]  # only 1 symbol -- never eligible at threshold 20

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date", fake_universe
    )
    dates = eligible_dates(prices, min_universe_size=20)
    assert dates == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_forward_returns.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ascent/analyst/proof_audit/forward_returns.py`:

```python
"""Point-in-time forward returns and eligible-date folds.

No look-ahead: forward_return_matrix's row for date T is the return realized from T to the NEXT
row in the index, never from T-1 to T. eligible_dates additionally filters to dates where the
point-in-time universe (ascent/data/universe.py::get_universe_on_date) is large enough for a
cross-sectional IC to be meaningful.
"""
from __future__ import annotations

import pandas as pd

from ascent.data.universe import get_universe_on_date


def forward_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Next-row simple return per column. Last row is NaN -- it has no next day."""
    return prices.pct_change().shift(-1)


def eligible_dates(
    prices: pd.DataFrame, min_universe_size: int = 20
) -> list[pd.Timestamp]:
    """Dates with a next-day return available AND a large-enough point-in-time universe."""
    if len(prices.index) < 2:
        return []
    candidate_dates = prices.index[:-1]  # last date has no forward return
    out = []
    for d in candidate_dates:
        universe = get_universe_on_date(d)
        if len(universe) >= min_universe_size:
            out.append(d)
    return out
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/forward_returns.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_forward_returns.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/analyst/proof_audit/forward_returns.py tests/analyst/proof_audit/test_forward_returns.py
git commit -m "feat(proof-audit): point-in-time forward returns and eligible-date folds

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: Sleeve signal registry and Path A scoring for pure-function sleeves

**Files:**
- Create: `ascent/analyst/proof_audit/sleeve_signals.py`
- Create: `ascent/analyst/proof_audit/wf_scorer.py`
- Test: `tests/analyst/proof_audit/test_sleeve_signals.py`, `tests/analyst/proof_audit/test_wf_scorer.py`

**Interfaces:**
- Consumes: `Component`/`COMPONENTS` (Task 1), `ICResult`/`score_ic_series` (Task 2),
  `forward_return_matrix`/`eligible_dates` (Task 3)
- Produces: `SLEEVE_SIGNAL_FUNCS: dict[str, Callable[[dict], pandas.DataFrame]]`;
  `score_sleeve(name: str, features: dict, prices: pandas.DataFrame) -> ICResult`. Task 7 calls
  `score_sleeve` for every `Component` with `kind == "alpha_sleeve"` and `method == "wf_ic"`.

- [ ] **Step 1: Write the failing test for the registry**

Create `tests/analyst/proof_audit/test_sleeve_signals.py`:

```python
"""Every wf_ic-method sleeve in components.py must have a real registered signal function."""
from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.sleeve_signals import SLEEVE_SIGNAL_FUNCS


def test_every_wf_ic_sleeve_is_registered():
    wf_ic_sleeves = {
        c.name for c in COMPONENTS if c.kind == "alpha_sleeve" and c.method == "wf_ic"
    }
    assert wf_ic_sleeves.issubset(set(SLEEVE_SIGNAL_FUNCS))


def test_registered_funcs_are_callable():
    for fn in SLEEVE_SIGNAL_FUNCS.values():
        assert callable(fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_sleeve_signals.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the sleeve signal registry**

Create `ascent/analyst/proof_audit/sleeve_signals.py`:

```python
"""Registry of pure alpha-sleeve signal functions used by the proof audit.

Each entry takes the same `features: dict[str, pandas.DataFrame]` shape ascent/alpha/stack.py
passes to its sleeve functions, and returns a date x symbol DataFrame of raw (pre-normalization)
signal values. ml, llm_fundamental and narrative are deliberately absent -- see the "Known scope
limits" section of the proof-audit plan.

The volatility formula is duplicated from ascent/alpha/stack.py's inline vol-regime block (not
imported, since it isn't a standalone function there) -- kept in sync manually; it is 3 lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ascent.alpha.trend import trend_alpha
from ascent.alpha.meanrev import meanrev_alpha
from ascent.alpha.statarb import statarb_alpha
from ascent.alpha.fundamental import fundamental_alpha
from ascent.alpha.earnings import earnings_alpha
from ascent.alpha.analyst import analyst_alpha
from ascent.alpha.options_flow import options_flow_alpha
from ascent.alpha.insider import insider_alpha
from ascent.alpha.short_interest import short_interest_alpha
from ascent.alpha.altdata_alpha import altdata_alpha
from ascent.alpha.earnings_tone import earnings_tone_alpha
from ascent.alpha.stack import _load_sector_map


def _volatility_signal(features: dict) -> pd.DataFrame:
    if "vol_of_vol_21d" in features and "vol_trend_10d" in features:
        vov = features["vol_of_vol_21d"].copy().replace(0, np.nan)
        vtrnd = features["vol_trend_10d"].copy()
        return -vtrnd / (vov + 1e-6)
    if "vol_21d" in features:
        return -features["vol_21d"].copy()
    return pd.DataFrame()


def _statarb_signal(features: dict) -> pd.DataFrame:
    return statarb_alpha(features, sector_map=_load_sector_map())


SLEEVE_SIGNAL_FUNCS = {
    "trend": trend_alpha,
    "meanrev": meanrev_alpha,
    "volatility": _volatility_signal,
    "statarb": _statarb_signal,
    "fundamental": fundamental_alpha,
    "earnings": earnings_alpha,
    "analyst": analyst_alpha,
    "options_flow": options_flow_alpha,
    "insider": insider_alpha,
    "short_interest": short_interest_alpha,
    "altdata": lambda features: altdata_alpha(features=features),
    "earnings_tone": earnings_tone_alpha,
}
```

- [ ] **Step 4: Run the registry test**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/sleeve_signals.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_sleeve_signals.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Write the failing test for wf_scorer**

Create `tests/analyst/proof_audit/test_wf_scorer.py`:

```python
"""score_sleeve wires signal + forward returns + stats.py together correctly."""
import pandas as pd
import pytest

from ascent.analyst.proof_audit.wf_scorer import score_sleeve


def _planted_features_and_prices(n_days=40, n_symbols=25):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rng_signal = pd.DataFrame(
        [[(i + j) % 7 - 3 for j in range(n_symbols)] for i in range(n_days)],
        index=dates, columns=symbols,
    ).astype(float)
    # Prices constructed so tomorrow's return is proportional to today's signal
    # (planted positive IC) plus small noise.
    prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    prices.iloc[0] = 100.0
    for i in range(1, n_days):
        planted_ret = 0.01 * rng_signal.iloc[i - 1] / (rng_signal.iloc[i - 1].abs().max() or 1)
        prices.iloc[i] = prices.iloc[i - 1] * (1 + planted_ret)
    features = {"toy_signal": rng_signal}
    return features, prices


def test_score_sleeve_detects_planted_positive_ic(monkeypatch):
    features, prices = _planted_features_and_prices()

    def fake_signal_func(features):
        return features["toy_signal"]

    monkeypatch.setitem(
        __import__(
            "ascent.analyst.proof_audit.sleeve_signals", fromlist=["SLEEVE_SIGNAL_FUNCS"]
        ).SLEEVE_SIGNAL_FUNCS,
        "trend",
        fake_signal_func,
    )
    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date",
        lambda date, universe_df=None: list(prices.columns),
    )
    result = score_sleeve("trend", features, prices)
    assert result.ic_mean > 0
    assert result.n > 0


def test_score_sleeve_unknown_name_raises():
    with pytest.raises(KeyError):
        score_sleeve("not_a_sleeve", {}, pd.DataFrame())
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_wf_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ascent.analyst.proof_audit.wf_scorer'`

- [ ] **Step 7: Implement wf_scorer**

Create `ascent/analyst/proof_audit/wf_scorer.py`:

```python
"""Path A: walk-forward IC/Sharpe scoring for pure alpha-sleeve and agent signal functions."""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from ascent.analyst.proof_audit.forward_returns import eligible_dates, forward_return_matrix
from ascent.analyst.proof_audit.sleeve_signals import SLEEVE_SIGNAL_FUNCS
from ascent.analyst.proof_audit.stats import ICResult, score_ic_series

N_LEGS = 5  # top/bottom quintile for the long-short daily return


def _daily_ic_and_ls_return(signal_row: pd.Series, forward_row: pd.Series) -> tuple[float, float] | None:
    both = pd.DataFrame({"signal": signal_row, "fwd": forward_row}).dropna()
    if len(both) < N_LEGS * 2:
        return None
    ic, _ = spearmanr(both["signal"], both["fwd"])
    if ic != ic:  # NaN check without importing math for one use
        return None
    ranked = both.sort_values("signal")
    bottom = ranked.iloc[:N_LEGS]["fwd"].mean()
    top = ranked.iloc[-N_LEGS:]["fwd"].mean()
    ls_return = top - bottom
    return float(ic), float(ls_return)


def score_signal_matrix(signal: pd.DataFrame, prices: pd.DataFrame) -> ICResult:
    """Shared core: score any date x symbol signal matrix against prices."""
    fwd = forward_return_matrix(prices)
    dates = eligible_dates(prices)
    daily_ic, daily_ls = [], []
    for d in dates:
        if d not in signal.index or d not in fwd.index:
            continue
        pair = _daily_ic_and_ls_return(signal.loc[d], fwd.loc[d])
        if pair is None:
            continue
        ic, ls = pair
        daily_ic.append(ic)
        daily_ls.append(ls)
    return score_ic_series(daily_ic, daily_ls)


def score_sleeve(name: str, features: dict, prices: pd.DataFrame) -> ICResult:
    if name not in SLEEVE_SIGNAL_FUNCS:
        raise KeyError(f"unknown sleeve {name!r}; known: {sorted(SLEEVE_SIGNAL_FUNCS)}")
    signal = SLEEVE_SIGNAL_FUNCS[name](features)
    return score_signal_matrix(signal, prices)
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/wf_scorer.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_wf_scorer.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add ascent/analyst/proof_audit/sleeve_signals.py ascent/analyst/proof_audit/wf_scorer.py \
        tests/analyst/proof_audit/test_sleeve_signals.py tests/analyst/proof_audit/test_wf_scorer.py
git commit -m "feat(proof-audit): sleeve signal registry and Path A walk-forward scoring

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: Agent signal registry (macro / international / alternatives)

**Files:**
- Create: `ascent/analyst/proof_audit/agent_signals.py`
- Modify: `ascent/analyst/proof_audit/wf_scorer.py` — add `score_agent`
- Test: `tests/analyst/proof_audit/test_agent_signals.py`

**Interfaces:**
- Consumes: `score_signal_matrix` (Task 4, unchanged); each agent module's private
  `_build_features` and `_build_*_alpha` functions
  (`agents/macro_agent.py:93,127`, `agents/international_agent.py:101,122`,
  `agents/alternatives_agent.py:85,105`)
- Produces: `AGENT_SIGNAL_FUNCS: dict[str, Callable[[pandas.DataFrame], pandas.DataFrame]]`
  (each takes a prices DataFrame, builds its own features internally, and returns a signal
  matrix); `score_agent(name: str, prices: pandas.DataFrame) -> ICResult`

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/proof_audit/test_agent_signals.py`:

```python
"""Each wf_ic-method agent in components.py must have a real registered signal function."""
from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.agent_signals import AGENT_SIGNAL_FUNCS


def test_every_wf_ic_agent_is_registered():
    wf_ic_agents = {c.name for c in COMPONENTS if c.kind == "agent" and c.method == "wf_ic"}
    assert wf_ic_agents == set(AGENT_SIGNAL_FUNCS)


def test_registered_funcs_are_callable():
    for fn in AGENT_SIGNAL_FUNCS.values():
        assert callable(fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_agent_signals.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ascent/analyst/proof_audit/agent_signals.py`:

```python
"""Registry of the 3 non-us-equities specialist agents' internal alpha builders.

us_equities_agent is intentionally absent -- its signal is exactly the sleeve stack Task 4
already scores; see the "Known scope limits" section of the proof-audit plan.

Each agent module keeps its alpha builder as a module-private function (`_build_*_alpha`) taking
the prices/features it fetches itself. We call those private functions directly rather than
duplicating their feature-engineering logic -- duplicating it would silently drift from what the
live agent actually runs.
"""
from __future__ import annotations

import pandas as pd

from agents.macro_agent import _build_macro_features, _build_trend_alpha as _macro_trend_alpha
from agents.international_agent import (
    _build_features as _international_features,
    _build_trend_alpha as _international_trend_alpha,
)
from agents.alternatives_agent import (
    _build_features as _alternatives_features,
    _build_alternatives_alpha,
)


def _macro_signal(prices: pd.DataFrame) -> pd.DataFrame:
    features = _build_macro_features(prices)
    return _macro_trend_alpha(features, prices)


def _international_signal(prices: pd.DataFrame) -> pd.DataFrame:
    features = _international_features(prices)
    return _international_trend_alpha(features)


def _alternatives_signal(prices: pd.DataFrame) -> pd.DataFrame:
    features = _alternatives_features(prices)
    return _build_alternatives_alpha(features, prices)


AGENT_SIGNAL_FUNCS = {
    "macro_agent": _macro_signal,
    "international_agent": _international_signal,
    "alternatives_agent": _alternatives_signal,
}
```

- [ ] **Step 4: Add score_agent to wf_scorer.py**

In `ascent/analyst/proof_audit/wf_scorer.py`, add at the end:

```python
from ascent.analyst.proof_audit.agent_signals import AGENT_SIGNAL_FUNCS


def score_agent(name: str, prices: pd.DataFrame) -> ICResult:
    if name not in AGENT_SIGNAL_FUNCS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENT_SIGNAL_FUNCS)}")
    signal = AGENT_SIGNAL_FUNCS[name](prices)
    return score_signal_matrix(signal, prices)
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/agent_signals.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/wf_scorer.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_agent_signals.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ascent/analyst/proof_audit/agent_signals.py ascent/analyst/proof_audit/wf_scorer.py \
        tests/analyst/proof_audit/test_agent_signals.py
git commit -m "feat(proof-audit): score the 3 non-us-equities specialist agents

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: Counterfactual scorer (Path B) for named subsystems

**Files:**
- Create: `ascent/analyst/proof_audit/counterfactual_scorer.py`
- Test: `tests/analyst/proof_audit/test_counterfactual_scorer.py`

**Interfaces:**
- Consumes: `ascent.analyst.catalog.registry` (`names()`, `describe()`, `load()`,
  `Series` — already in the repo at `ascent/analyst/catalog/registry.py`), `ICResult`/`score_ic_series` (Task 2, reused for its t-test machinery on a return-delta series instead of an IC series — the field names `ic_mean`/`ic_t` are reinterpreted as "delta_mean"/"delta_t" by the caller, not renamed, to avoid a second near-duplicate dataclass)
- Produces: `score_subsystem(name: str) -> ICResult`; `SUBSYSTEM_TRACK_PAIRS: dict[str, tuple[str, str]]`
  mapping subsystem name to `(with_component_track, without_component_track)` canonical series names.

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/proof_audit/test_counterfactual_scorer.py`:

```python
"""Path B: subsystems are scored by with-vs-without return delta on the counterfactual tracks."""
import pandas as pd
import pytest

from ascent.analyst.proof_audit.counterfactual_scorer import (
    SUBSYSTEM_TRACK_PAIRS,
    score_subsystem,
)


def test_all_named_subsystems_have_track_pairs():
    from ascent.analyst.proof_audit.components import COMPONENTS

    counterfactual_subsystems = {
        c.name for c in COMPONENTS if c.kind == "subsystem" and c.method == "counterfactual"
    }
    assert counterfactual_subsystems == set(SUBSYSTEM_TRACK_PAIRS)


def test_score_subsystem_detects_planted_positive_delta(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    with_track = pd.Series([0.002] * 30, index=dates)
    without_track = pd.Series([0.001] * 30, index=dates)

    def fake_load(name):
        return with_track if name == "counterfactual.track_d" else without_track

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.counterfactual_scorer.registry.load", fake_load
    )
    result = score_subsystem("earned_authority")
    assert result.ic_mean > 0
    assert result.n == 30


def test_score_subsystem_unknown_name_raises():
    with pytest.raises(KeyError):
        score_subsystem("not_a_subsystem")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_counterfactual_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ascent/analyst/proof_audit/counterfactual_scorer.py`:

```python
"""Path B: counterfactual return-delta scoring for named subsystems.

Reuses the canonical counterfactual tracks (ascent/analyst/catalog/registry.py) where an
existing track already isolates the subsystem being tested. earned_authority and
debate_judge_intervention map onto the existing Track D (pure AI PM) vs Track A* (pure quant)
pair -- that pair is exactly "with AI-layer influence" vs "without it", which is what those two
subsystems inject. regime_overlay and hedge_overlay don't have an existing isolating track, so
they map onto the same pair as a documented approximation pending sub-project 2's design work to
build a dedicated synthetic track for each -- recorded here, not hidden.

score_subsystem reuses ICResult/score_ic_series from stats.py: for this path "ic_mean"/"ic_t"
hold the mean/t-stat of the daily WITH-minus-WITHOUT return delta, not a rank correlation. This
is a deliberate field reuse (avoids a near-duplicate dataclass), not a naming mismatch.
"""
from __future__ import annotations

from ascent.analyst.catalog import registry
from ascent.analyst.proof_audit.stats import ICResult, score_ic_series

# (with_component_track, without_component_track), both canonical names from registry.py
SUBSYSTEM_TRACK_PAIRS: dict[str, tuple[str, str]] = {
    "earned_authority": ("counterfactual.track_d", "counterfactual.track_astar"),
    "debate_judge_intervention": ("counterfactual.track_b", "counterfactual.track_d"),
    "regime_overlay": ("counterfactual.track_d", "counterfactual.track_astar"),
    "hedge_overlay": ("counterfactual.track_d", "counterfactual.track_astar"),
}


def score_subsystem(name: str) -> ICResult:
    if name not in SUBSYSTEM_TRACK_PAIRS:
        raise KeyError(f"unknown subsystem {name!r}; known: {sorted(SUBSYSTEM_TRACK_PAIRS)}")
    with_name, without_name = SUBSYSTEM_TRACK_PAIRS[name]
    with_series = registry.load(with_name)
    without_series = registry.load(without_name)
    aligned = with_series.to_frame("with").join(without_series.to_frame("without"), how="inner")
    delta = (aligned["with"] - aligned["without"]).tolist()
    # score_ic_series expects a parallel (ic, ls_return) pair per date; for a return-delta
    # series there is only one meaningful series, so we pass it as both arguments.
    return score_ic_series(delta, delta)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/counterfactual_scorer.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_counterfactual_scorer.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/analyst/proof_audit/counterfactual_scorer.py \
        tests/analyst/proof_audit/test_counterfactual_scorer.py
git commit -m "feat(proof-audit): counterfactual return-delta scoring for named subsystems

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 7: Scorecard verdict rule, JSON writer, and CLI

**Files:**
- Create: `ascent/analyst/proof_audit/scorecard.py`
- Create: `ascent/analyst/proof_audit/run.py`
- Test: `tests/analyst/proof_audit/test_scorecard.py`

**Interfaces:**
- Consumes: `Component`/`COMPONENTS`/`get_component` (Task 1), `ICResult` (Task 2),
  `score_sleeve`/`score_agent` (Task 4/5), `score_subsystem` (Task 6)
- Produces: `verdict(result: ICResult, min_sample: int = 30) -> str` (returns
  `"KEEP"`/`"CUT"`/`"INSUFFICIENT_DATA"`); `ScorecardRow` dataclass
  (`component: str, kind: str, method: str, metric: float, p_value: float, sample_size: int, verdict: str`);
  `write_scorecard(rows: list[ScorecardRow], out_path: pathlib.Path) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/proof_audit/test_scorecard.py`:

```python
"""Verdict rule is three-way and never silently defaults."""
import json

from ascent.analyst.proof_audit.scorecard import ScorecardRow, verdict, write_scorecard
from ascent.analyst.proof_audit.stats import ICResult


def test_significant_positive_is_keep():
    result = ICResult(ic_mean=0.03, ic_t=3.5, p_value=0.001, sharpe=1.2, n=50)
    assert verdict(result) == "KEEP"


def test_significant_negative_is_cut():
    result = ICResult(ic_mean=-0.02, ic_t=-3.1, p_value=0.002, sharpe=-0.8, n=50)
    assert verdict(result) == "CUT"


def test_not_significant_is_cut():
    result = ICResult(ic_mean=0.001, ic_t=0.4, p_value=0.7, sharpe=0.05, n=50)
    assert verdict(result) == "CUT"


def test_below_min_sample_is_insufficient_data():
    result = ICResult(ic_mean=0.05, ic_t=2.0, p_value=0.01, sharpe=1.0, n=5)
    assert verdict(result, min_sample=30) == "INSUFFICIENT_DATA"


def test_write_scorecard_round_trips(tmp_path):
    rows = [
        ScorecardRow(
            component="trend", kind="alpha_sleeve", method="wf_ic",
            metric=0.03, p_value=0.001, sample_size=50, verdict="KEEP",
        ),
        ScorecardRow(
            component="ml", kind="alpha_sleeve", method="deferred",
            metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
        ),
    ]
    out = tmp_path / "scorecard.json"
    write_scorecard(rows, out)
    loaded = json.loads(out.read_text())
    assert len(loaded) == 2
    assert loaded[0]["component"] == "trend"
    assert loaded[0]["verdict"] == "KEEP"
    assert loaded[1]["metric"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/proof_audit/test_scorecard.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scorecard.py**

Create `ascent/analyst/proof_audit/scorecard.py`:

```python
"""Verdict rule and scorecard I/O. The only place KEEP/CUT/INSUFFICIENT_DATA is decided."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from ascent.analyst.proof_audit.stats import ICResult

SIGNIFICANCE_P = 0.05
DEFAULT_MIN_SAMPLE = 30


def verdict(result: ICResult, min_sample: int = DEFAULT_MIN_SAMPLE) -> str:
    """Three-way, never a silent default.

    INSUFFICIENT_DATA takes priority over the significance test -- a small sample that happens
    to look significant is not trustworthy enough to KEEP or CUT on.
    """
    if result.n < min_sample:
        return "INSUFFICIENT_DATA"
    if result.p_value < SIGNIFICANCE_P and result.ic_mean > 0:
        return "KEEP"
    return "CUT"


@dataclass(frozen=True)
class ScorecardRow:
    component: str
    kind: str
    method: str
    metric: float | None
    p_value: float | None
    sample_size: int
    verdict: str


def write_scorecard(rows: list[ScorecardRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in rows]
    out_path.write_text(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Run scorecard tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/scorecard.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/test_scorecard.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Implement the CLI**

Create `ascent/analyst/proof_audit/run.py`:

```python
#!/usr/bin/env python
"""Run the full proof audit: Path A (sleeves + agents) + Path B (subsystems) -> scorecard.

    .venv/bin/python -m ascent.analyst.proof_audit.run

Requires: data_cache prices_live parquet (for features/prices) and
logs/counterfactual_daily.jsonl (for the subsystem tracks). Missing inputs fail that
component's row as INSUFFICIENT_DATA, not the whole run -- see the plan's Task 7 for the
per-component try/except boundary.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.counterfactual_scorer import score_subsystem
from ascent.analyst.proof_audit.scorecard import DEFAULT_MIN_SAMPLE, ScorecardRow, verdict, write_scorecard
from ascent.analyst.proof_audit.stats import ICResult
from ascent.analyst.proof_audit.wf_scorer import score_agent, score_sleeve

log = logging.getLogger(__name__)

_DEFERRED_REASON = {
    "deferred": "requires live-logged signal history, not re-simulation -- out of scope for this audit",
    "covered_by_sleeves": "covered by per-sleeve rows; not scored standalone",
}


def _row_for_deferred(name: str, kind: str, method: str) -> ScorecardRow:
    log.info("proof_audit: %s (%s) skipped -- %s", name, method, _DEFERRED_REASON[method])
    return ScorecardRow(
        component=name, kind=kind, method=method,
        metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
    )


def _row_from_result(name: str, kind: str, method: str, result: ICResult) -> ScorecardRow:
    return ScorecardRow(
        component=name, kind=kind, method=method,
        metric=result.ic_mean, p_value=result.p_value, sample_size=result.n,
        verdict=verdict(result, min_sample=DEFAULT_MIN_SAMPLE),
    )


def run(features: dict, prices, out_path: Path | None = None) -> list[ScorecardRow]:
    if out_path is None:
        out_path = Path("outputs/analyst") / f"proof_audit_{date.today().isoformat()}.json"

    rows: list[ScorecardRow] = []
    for c in COMPONENTS:
        try:
            if c.method in _DEFERRED_REASON:
                rows.append(_row_for_deferred(c.name, c.kind, c.method))
            elif c.kind == "alpha_sleeve":
                rows.append(_row_from_result(c.name, c.kind, c.method, score_sleeve(c.name, features, prices)))
            elif c.kind == "agent":
                rows.append(_row_from_result(c.name, c.kind, c.method, score_agent(c.name, prices)))
            elif c.kind == "subsystem":
                rows.append(_row_from_result(c.name, c.kind, c.method, score_subsystem(c.name)))
        except Exception as exc:
            log.warning("proof_audit: %s failed (%s) -- marking INSUFFICIENT_DATA", c.name, exc)
            rows.append(ScorecardRow(
                component=c.name, kind=c.kind, method=c.method,
                metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
            ))

    write_scorecard(rows, out_path)
    log.info("proof_audit: wrote %d rows to %s", len(rows), out_path)
    return rows


if __name__ == "__main__":
    import argparse

    from ascent.data.store.parquet import load_parquet
    from ascent.features import build_features  # existing feature builder used by main.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    prices = load_parquet("prices_live")
    features = build_features(prices)
    run(features, prices, out_path=args.out)
```

- [ ] **Step 6: Verify `ascent.features.build_features` exists with this signature**

```bash
.venv/bin/python -c "from ascent.features import build_features; import inspect; print(inspect.signature(build_features))"
```

If the signature differs, adjust the `if __name__ == "__main__":` block in `run.py` to match —
this block is a convenience CLI wrapper, not covered by a unit test, so it must be checked by hand.

- [ ] **Step 7: Run syntax check and full proof_audit test suite**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/run.py').read())"
.venv/bin/python -m pytest tests/analyst/proof_audit/ -v
```

Expected: all tests across Tasks 1-7 PASS (28 tests total).

- [ ] **Step 8: Commit**

```bash
git add ascent/analyst/proof_audit/scorecard.py ascent/analyst/proof_audit/run.py \
        tests/analyst/proof_audit/test_scorecard.py
git commit -m "feat(proof-audit): verdict rule, scorecard writer, and CLI entrypoint

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 8: Real-data run and human-readable summary

**Files:**
- Create: `scripts/run_proof_audit.py` (thin wrapper, real-data invocation)
- No new test — this task's deliverable is a real artifact, not a pinned assertion (per spec:
  "a real-data run against the live artifacts is a separate manual step; it is not asserted to
  produce a specific outcome in a unit test").

**Interfaces:**
- Consumes: `ascent.analyst.proof_audit.run.run` (Task 7)
- Produces: `outputs/analyst/proof_audit_<today>.json` (real scorecard) and a printed
  human-readable table for the session record.

- [ ] **Step 1: Write the script**

Create `scripts/run_proof_audit.py`:

```python
#!/usr/bin/env python
"""Run the proof audit against real repo data and print a human-readable summary.

    .venv/bin/python scripts/run_proof_audit.py
"""
from __future__ import annotations

from ascent.data.store.parquet import load_parquet
from ascent.features import build_features
from ascent.analyst.proof_audit.run import run


def main() -> int:
    prices = load_parquet("prices_live")
    features = build_features(prices)
    rows = run(features, prices)

    print(f"{'component':30s} {'kind':14s} {'method':16s} {'metric':>10s} {'p':>8s} {'n':>5s}  verdict")
    for r in sorted(rows, key=lambda r: (r.kind, r.component)):
        metric = f"{r.metric:.4f}" if r.metric is not None else "n/a"
        p = f"{r.p_value:.4f}" if r.p_value is not None else "n/a"
        print(f"{r.component:30s} {r.kind:14s} {r.method:16s} {metric:>10s} {p:>8s} {r.sample_size:5d}  {r.verdict}")

    n_keep = sum(1 for r in rows if r.verdict == "KEEP")
    n_cut = sum(1 for r in rows if r.verdict == "CUT")
    n_insufficient = sum(1 for r in rows if r.verdict == "INSUFFICIENT_DATA")
    print(f"\nKEEP={n_keep} CUT={n_cut} INSUFFICIENT_DATA={n_insufficient} (total {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it against real data**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/run_proof_audit.py').read())"
.venv/bin/python scripts/run_proof_audit.py
```

Expected: prints one row per component in `components.py` (23 rows), a summary line, and writes
`outputs/analyst/proof_audit_<today>.json`. **Record the actual KEEP/CUT counts in the commit
message** — this is the real output sub-project 2 will consume, not a hypothetical.

**Stop and read the printed table before continuing.** If every row is `INSUFFICIENT_DATA`, the
`prices_live` cache or `features` builder likely isn't producing what Task 4-6 expect — do not
proceed to sub-project 2 on an all-INSUFFICIENT_DATA scorecard; report the discrepancy instead.

- [ ] **Step 3: Run the full test suite once more and check doc guard**

```bash
.venv/bin/python -m pytest tests/analyst/proof_audit/ -v
.venv/bin/python scripts/verify_docs.py --quiet
```

Expected: 28 tests PASS, 0 verify_docs failures (this plan added no CLAUDE.md claims to check).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_proof_audit.py outputs/analyst/
git commit -m "feat(proof-audit): real-data run and human-readable summary script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Done criteria

```bash
.venv/bin/python -m pytest tests/analyst/proof_audit/ -v
.venv/bin/python scripts/run_proof_audit.py
.venv/bin/python scripts/verify_docs.py --quiet
```

- 28 tests pass across Tasks 1-7.
- `outputs/analyst/proof_audit_<date>.json` exists with 23 rows (one per `Component` in
  `components.py`), each with a non-null `verdict`.
- `verify_docs.py` still reports 0 failures.
- `.venv/bin/python -c "import ascent.analyst.proof_audit.run"` succeeds.

## Explicitly out of scope

- Sub-project 2 (target architecture design) — consumes this scorecard, doesn't start until it exists.
- Deleting any code based on the scorecard's CUT verdicts — that's sub-project 3.
- Resuming `com.ascentcapital.eod`/`heartbeat` — stays paused through sub-project 4.
- Building dedicated synthetic counterfactual tracks for `regime_overlay`/`hedge_overlay` (Task 6
  approximates both onto the earned-authority pair) — flagged for sub-project 2, not solved here.
- Retraining the `ml` sleeve per fold, or re-running `llm_fundamental`/`narrative` historically.
