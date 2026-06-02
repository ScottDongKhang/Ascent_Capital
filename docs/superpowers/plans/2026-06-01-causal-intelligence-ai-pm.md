# Causal Intelligence for the AI PM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the AI PM a structural causal model of the economy and each holding so it reasons about *why* trades work — earning authority faster through sharper, regime-compatible, independently-timed bets.

**Architecture:** Four phases. Phase A builds the `ascent/causal/` module and foundational types. Phase B wires Gate 1 (regime-causal compatibility) and Gate 2 (priced-in filter) into AI PM Phase 1. Phase C adds the falsification tracker and Gate 4 (intra-horizon early exit). Phase D injects causal context into the devil's advocate. All existing 675 tests must remain green throughout. All schema additions to existing dataclasses are backward-compatible (`field(default_factory=list)`).

**Tech Stack:** Python 3.12, `causal-learn` (PC algorithm — installed), `anthropic` via `ascent.llm.client`, `pandas`/`numpy`, `pytest`/`unittest.mock`

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `ascent/config/types.py` | Modify | Add `CausalMechanism` dataclass |
| `agents/ai_pm_agent.py` | Modify | Add `causal_mechanisms` to `AIPreThesis`; `get_causal_graph` tool; gate 1+2 pre-thesis; Phase 2 track record injection |
| `ascent/causal/__init__.py` | Create | Module init |
| `ascent/causal/velocity.py` | Create | `mechanism_velocity_score()` — pure Python, no imports |
| `ascent/causal/causal_discovery.py` | Create | PC algorithm on FRED + sector ETF returns → `data_cache/macro_causal_dag.json` |
| `ascent/causal/dag_builder.py` | Create | Haiku per-symbol causal graph builder; cache by `(symbol, quarter_end)` to `data_cache/causal_graphs/` |
| `ascent/causal/compatibility.py` | Create | `regime_compatible(mechanism_type, regime)` — static dict, no LLM |
| `ascent/causal/tracker.py` | Create | Write `logs/causal_predictions.jsonl`; `check_outcomes()`; `check_early_exits()` → early-cut symbols |
| `ascent/monitoring/weekend_runner.py` | Modify | Add `causal_macro_dag` + `causal_graph_builder` jobs |
| `run_all_agents.py` | Modify | Non-rebalance path: call `check_early_exits()`; Phase 2: pass `causal_track_record` |
| `debate/agents.py` | Modify | Append `causal_mechanisms` context to devil's advocate system prompt |
| `tests/test_causal_velocity.py` | Create | Velocity math, [0,1] clamp, catalyst_imminent boundary |
| `tests/test_causal_discovery.py` | Create | PC on synthetic data; JSON schema; no duplicate edges |
| `tests/test_dag_builder.py` | Create | Mock Haiku; cache hit skips call; quarterly cache key; schema valid |
| `tests/test_causal_compatibility.py` | Create | All regime × mechanism-type combinations |
| `tests/test_causal_tracker.py` | Create | Write/read predictions; early_exit flag; outcome classification |
| `tests/test_ai_pm_prethesis_causal.py` | Create | `causal_mechanisms` field on `AIPreThesis`; gate 1 filtering; priced_in exclusion |

---

## PHASE A — Foundation

---

## Task 1: `CausalMechanism` datatype in `types.py`

**Files:**
- Modify: `ascent/config/types.py`
- Test: `tests/test_causal_velocity.py` (test for the datatype lives here — velocity tests come later in the same file)

### Background
`CausalMechanism` is the shared language between dag_builder, compatibility, tracker, AI PM, and debate. It lives in `types.py` (alongside `AgentOutput`) so all modules can import it without circular dependencies.

- [ ] **Step 1: Write the failing test**

Create `tests/test_causal_velocity.py`:

```python
# tests/test_causal_velocity.py
import pytest


def test_causal_mechanism_dataclass_fields():
    """CausalMechanism must have all spec-required fields with correct types."""
    from ascent.config.types import CausalMechanism
    m = CausalMechanism(
        symbol="WDC",
        mechanism="NAND oversupply correction → margin expansion → EPS rerating",
        intervention="IF NAND spot +15% from trough THEN WDC gross margin > 40%",
        falsification_condition="IF WDC Q3 gross margin < 38%, thesis broken",
        horizon_days=63,
        timing="catalyst_imminent",
        velocity=0.72,
        mechanism_type="supply_demand_inflection",
        regime_compatible=True,
    )
    assert m.symbol == "WDC"
    assert m.timing in ("priced_in", "not_yet_priced", "catalyst_imminent")
    assert 0.0 <= m.velocity <= 1.0
    assert isinstance(m.horizon_days, int)
    assert isinstance(m.regime_compatible, bool)


def test_causal_mechanism_timing_values():
    """timing must be one of three permitted values."""
    from ascent.config.types import CausalMechanism
    for t in ("priced_in", "not_yet_priced", "catalyst_imminent"):
        m = CausalMechanism(
            symbol="X", mechanism="m", intervention="i",
            falsification_condition="f", horizon_days=21,
            timing=t, velocity=0.5, mechanism_type="momentum_catalyst",
            regime_compatible=True,
        )
        assert m.timing == t
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_causal_velocity.py -v 2>&1 | tail -10
```

Expected: 2 FAILs — `ImportError: cannot import name 'CausalMechanism'`

- [ ] **Step 3: Add `CausalMechanism` to `types.py`**

Append after line 64 (after the `AgentOutput` class, before the file ends) in `ascent/config/types.py`:

```python
@dataclass
class CausalMechanism:
    """
    A single causal mechanism for one holding, built by dag_builder.py (Haiku).
    Stored in AIPreThesis.causal_mechanisms and logged to causal_predictions.jsonl.
    """
    symbol: str
    mechanism: str              # "X causes Y via Z"
    intervention: str           # "IF [trigger] THEN [expected outcome]"
    falsification_condition: str  # "IF [observable] < [threshold], thesis broken"
    horizon_days: int
    timing: str                 # "priced_in" | "not_yet_priced" | "catalyst_imminent"
    velocity: float             # 0.0–1.0, Python-computed at build time
    mechanism_type: str         # "momentum_catalyst" | "quality_defensive" | "macro_hedge" |
                                # "mean_reversion" | "valuation" | "supply_demand_inflection"
    regime_compatible: bool     # gate 1 result, set by compatibility.py
```

Also add `List` to the `typing` import at the top of `types.py`:

```python
from typing import Dict, List, Optional, Any
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_causal_velocity.py -v 2>&1 | tail -10
```

Expected: 2 PASSes

- [ ] **Step 5: Run full suite to check no regressions**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: 675 passed, 1 skipped, 0 failures

- [ ] **Step 6: Commit**

```bash
git add ascent/config/types.py tests/test_causal_velocity.py
git commit -m "feat: add CausalMechanism dataclass to types.py"
```

---

## Task 2: `ascent/causal/velocity.py`

**Files:**
- Create: `ascent/causal/__init__.py`
- Create: `ascent/causal/velocity.py`
- Modify: `tests/test_causal_velocity.py`

### Background

Velocity measures how fast a causal trigger is progressing toward its threshold — pure math, no imports beyond stdlib. A velocity > 0.80 is considered `catalyst_imminent` quality even if the graph was built with `not_yet_priced`.

- [ ] **Step 1: Add failing velocity tests to the existing file**

Append to `tests/test_causal_velocity.py`:

```python
def test_velocity_mid_progress():
    from ascent.causal.velocity import mechanism_velocity_score
    # 12% of 20% needed = 0.60
    assert abs(mechanism_velocity_score(
        current_value=0.12, baseline_value=0.0, threshold_value=0.20
    ) - 0.60) < 1e-9


def test_velocity_clamped_below_zero():
    from ascent.causal.velocity import mechanism_velocity_score
    # current below baseline → clamp to 0.0
    assert mechanism_velocity_score(
        current_value=-5.0, baseline_value=0.0, threshold_value=10.0
    ) == 0.0


def test_velocity_clamped_above_one():
    from ascent.causal.velocity import mechanism_velocity_score
    # current beyond threshold → clamp to 1.0
    assert mechanism_velocity_score(
        current_value=25.0, baseline_value=0.0, threshold_value=20.0
    ) == 1.0


def test_velocity_zero_range_returns_zero():
    from ascent.causal.velocity import mechanism_velocity_score
    # baseline == threshold → undefined, return 0.0
    assert mechanism_velocity_score(
        current_value=5.0, baseline_value=5.0, threshold_value=5.0
    ) == 0.0


def test_velocity_at_exactly_threshold():
    from ascent.causal.velocity import mechanism_velocity_score
    assert mechanism_velocity_score(
        current_value=20.0, baseline_value=0.0, threshold_value=20.0
    ) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_causal_velocity.py -k "velocity" -v 2>&1 | tail -10
```

Expected: 5 FAILs — `ModuleNotFoundError: No module named 'ascent.causal'`

- [ ] **Step 3: Create `ascent/causal/__init__.py`**

```python
"""ascent/causal — causal intelligence for the AI PM."""
```

- [ ] **Step 4: Create `ascent/causal/velocity.py`**

```python
"""ascent/causal/velocity.py
Pure-Python mechanism velocity score.
No external imports — safe to call from any context.
"""


def mechanism_velocity_score(
    current_value: float,
    baseline_value: float,
    threshold_value: float,
) -> float:
    """
    Returns a velocity score in [0.0, 1.0] measuring progress from baseline
    toward threshold.

    velocity = (current - baseline) / (threshold - baseline)

    Returns 0.0 when the range is zero (baseline == threshold).
    Clamped to [0.0, 1.0].
    """
    denom = threshold_value - baseline_value
    if abs(denom) < 1e-10:
        return 0.0
    v = (current_value - baseline_value) / denom
    return max(0.0, min(1.0, v))
```

- [ ] **Step 5: Run the velocity tests**

```bash
.venv/bin/python -m pytest tests/test_causal_velocity.py -v 2>&1 | tail -12
```

Expected: 7 PASSes (2 from Task 1 + 5 new)

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 677 passed, 1 skipped

- [ ] **Step 7: Commit**

```bash
git add ascent/causal/__init__.py ascent/causal/velocity.py tests/test_causal_velocity.py
git commit -m "feat: add ascent/causal module + mechanism_velocity_score()"
```

---

## Task 3: `ascent/causal/causal_discovery.py` — macro DAG

**Files:**
- Create: `ascent/causal/causal_discovery.py`
- Create: `tests/test_causal_discovery.py`

### Background

Runs the PC constraint-based algorithm on 2-year weekly returns of: FRED macro series (fed_funds_rate, hy_spread, vix, unemployment) + 5 sector ETF returns (XLF, XLK, XLV, XLE, XLP). Writes `data_cache/macro_causal_dag.json`.

The PC algorithm from `causal-learn` returns a `CausalGraph` object. We read its adjacency matrix: `cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1` means edge `i → j`. We annotate strength using Pearson correlation magnitude on the weekly returns series.

- [ ] **Step 1: Write failing tests**

Create `tests/test_causal_discovery.py`:

```python
# tests/test_causal_discovery.py
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_synthetic_data():
    """5 nodes, 100 observations — clear A→B and C→D causal links via construction."""
    rng = np.random.default_rng(42)
    n = 100
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.2, n)   # A causes B
    c = rng.normal(0, 1, n)
    d = 0.7 * c + rng.normal(0, 0.3, n)   # C causes D
    e = rng.normal(0, 1, n)
    return np.column_stack([a, b, c, d, e])


def test_run_pc_returns_dag_schema(tmp_path):
    """run_pc must return a dict with 'nodes', 'edges', 'active_transmission_chains'."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    assert "nodes" in result
    assert "edges" in result
    assert "active_transmission_chains" in result
    assert result["nodes"] == node_names
    assert isinstance(result["edges"], list)
    assert isinstance(result["active_transmission_chains"], list)


def test_run_pc_edges_have_required_fields(tmp_path):
    """Each edge must have 'from', 'to', 'strength', 'direction' fields."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    for edge in result["edges"]:
        assert "from" in edge, f"Edge missing 'from': {edge}"
        assert "to" in edge,   f"Edge missing 'to': {edge}"
        assert edge["strength"] in ("strong", "moderate", "weak")
        assert edge["direction"] in ("positive", "negative")


def test_run_pc_no_self_loops(tmp_path):
    """PC algorithm must not produce self-loop edges."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    for edge in result["edges"]:
        assert edge["from"] != edge["to"], f"Self-loop detected: {edge}"


def test_run_pc_no_duplicate_edges(tmp_path):
    """Each directed edge should appear at most once."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    pairs = [(e["from"], e["to"]) for e in result["edges"]]
    assert len(pairs) == len(set(pairs)), "Duplicate edges in DAG output"


def test_discover_macro_dag_writes_json(tmp_path):
    """discover_macro_dag must write a valid JSON file to the given path."""
    import pandas as pd
    from ascent.causal.causal_discovery import discover_macro_dag

    # Build synthetic macro + sector data matching expected column names
    dates = pd.date_range("2024-01-05", periods=100, freq="W-FRI")
    macro_df = pd.DataFrame({
        "fed_rate": 5.25 + 0.01 * np.random.randn(100),
        "hy_spread": 3.5 + 0.1 * np.random.randn(100),
        "vix": 15 + np.random.randn(100),
        "unemployment": 4.0 + 0.01 * np.random.randn(100),
    }, index=dates)
    sector_df = pd.DataFrame({
        "XLF": 0.001 * np.random.randn(100),
        "XLK": 0.001 * np.random.randn(100),
        "XLV": 0.001 * np.random.randn(100),
        "XLE": 0.001 * np.random.randn(100),
        "XLP": 0.001 * np.random.randn(100),
    }, index=dates)
    out_path = tmp_path / "macro_causal_dag.json"

    discover_macro_dag(
        macro_df=macro_df,
        sector_df=sector_df,
        regime="calm_bull",
        output_path=out_path,
    )

    assert out_path.exists()
    dag = json.loads(out_path.read_text())
    assert "as_of" in dag
    assert "regime" in dag
    assert dag["regime"] == "calm_bull"
    assert len(dag["nodes"]) >= 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_causal_discovery.py -v 2>&1 | tail -10
```

Expected: 5 FAILs — `ModuleNotFoundError: No module named 'ascent.causal.causal_discovery'`

- [ ] **Step 3: Create `ascent/causal/causal_discovery.py`**

```python
"""ascent/causal/causal_discovery.py

Runs the PC constraint-based causal discovery algorithm on FRED macro
data + sector ETF weekly returns to produce the macro causal DAG.

Output written to data_cache/macro_causal_dag.json.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MACRO_DAG_PATH = Path("data_cache/macro_causal_dag.json")

MACRO_SERIES = ["fed_rate", "hy_spread", "vix", "unemployment"]
SECTOR_ETFS  = ["XLF", "XLK", "XLV", "XLE", "XLP"]


def run_pc(
    data: np.ndarray,
    node_names: List[str],
    alpha: float = 0.05,
) -> dict:
    """
    Run the PC algorithm on a T×N data matrix.
    Returns a dict with nodes, edges (directed), and active_transmission_chains.
    """
    from causallearn.search.ConstraintBased.PC import pc as run_pc_alg

    cg = run_pc_alg(data, alpha=alpha, indep_test="fisherz", show_progress=False)
    adj = cg.G.graph  # N×N: adj[i,j]==-1 and adj[j,i]==1 means i→j

    # Compute pairwise correlations for strength annotation
    corr = np.corrcoef(data.T)

    n = len(node_names)
    edges = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Directed edge i→j: adj[i,j]==-1 and adj[j,i]==1
            if adj[i, j] == -1 and adj[j, i] == 1:
                key = (node_names[i], node_names[j])
                if key in seen:
                    continue
                seen.add(key)
                r = corr[i, j]
                strength = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
                direction = "positive" if r > 0 else "negative"
                edges.append({
                    "from": node_names[i],
                    "to": node_names[j],
                    "strength": strength,
                    "direction": direction,
                })

    chains = _find_transmission_chains(edges, node_names)
    return {
        "nodes": node_names,
        "edges": edges,
        "active_transmission_chains": chains,
    }


def _find_transmission_chains(edges: list, node_names: list) -> List[str]:
    """
    Find paths of length 2+ through the DAG and return them as strings.
    Limited to paths starting from macro nodes (fed_rate, hy_spread, vix).
    """
    adj_map: dict = {}
    for e in edges:
        adj_map.setdefault(e["from"], []).append(e["to"])

    chains = []
    source_nodes = [n for n in node_names if n in MACRO_SERIES[:3]]
    for src in source_nodes:
        for mid in adj_map.get(src, []):
            for dst in adj_map.get(mid, []):
                if dst != src:
                    chains.append(f"{src} → {mid} → {dst}")
    return chains[:10]  # cap at 10 to keep JSON small


def _load_macro_data(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Convert macro_df to weekly frequency using last observation per week."""
    numeric = macro_df.select_dtypes(include=[np.number])
    weekly = numeric.resample("W-FRI").last().dropna(how="all")
    return weekly


def _load_sector_data(sector_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly returns from sector ETF price data."""
    prices = sector_df.resample("W-FRI").last()
    returns = prices.pct_change().dropna(how="all")
    return returns


def discover_macro_dag(
    macro_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    regime: str = "calm_bull",
    output_path: Optional[Path] = None,
) -> dict:
    """
    Build the macro causal DAG from FRED macro data + sector ETF returns.

    Args:
        macro_df: DataFrame indexed by date with columns matching MACRO_SERIES names
        sector_df: DataFrame indexed by date with columns for SECTOR_ETFS prices
        regime: current regime label for metadata
        output_path: where to write JSON (default: MACRO_DAG_PATH)

    Returns:
        The DAG dict (also written to output_path).
    """
    if output_path is None:
        output_path = MACRO_DAG_PATH

    macro_weekly = _load_macro_data(macro_df)
    sector_returns = _load_sector_data(sector_df)

    # Align on common dates, keep last 2 years (~104 weekly observations)
    combined = macro_weekly.join(sector_returns, how="inner").dropna()
    combined = combined.iloc[-104:]

    if len(combined) < 30:
        log.warning("[CausalDiscovery] Insufficient data (%d rows) for PC algorithm", len(combined))
        return {}

    node_names = list(combined.columns)
    data = combined.values.astype(float)

    log.info("[CausalDiscovery] Running PC on %d nodes, %d observations", len(node_names), len(data))
    dag = run_pc(data, node_names, alpha=0.05)
    dag["as_of"] = str(date.today())
    dag["regime"] = regime

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dag, indent=2))
    log.info(
        "[CausalDiscovery] DAG written: %d nodes, %d edges, %d chains",
        len(dag["nodes"]), len(dag["edges"]), len(dag["active_transmission_chains"]),
    )
    return dag


def run_discovery(regime: str = "calm_bull") -> dict:
    """
    Entry point for the weekend runner.
    Loads data from standard parquet caches and writes macro_causal_dag.json.
    """
    macro_path = Path("data_cache/macro_live.parquet")
    prices_path = Path("data_cache/prices_live.parquet")

    if not macro_path.exists():
        log.warning("[CausalDiscovery] macro_live.parquet not found — skipping")
        return {}

    # Load FRED macro: pivot from long to wide format
    macro_raw = pd.read_parquet(macro_path)
    macro_pivot = (
        macro_raw[macro_raw["name"].isin(MACRO_SERIES)]
        .pivot_table(index="date", columns="name", values="value", aggfunc="last")
    )
    macro_pivot.index = pd.to_datetime(macro_pivot.index)

    # Load sector ETF prices
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        if "symbol" in prices.columns:
            sector_prices = (
                prices[prices["symbol"].isin(SECTOR_ETFS)]
                .pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
            )
            sector_prices.index = pd.to_datetime(sector_prices.index)
        else:
            sector_prices = prices[[c for c in SECTOR_ETFS if c in prices.columns]]
    else:
        log.warning("[CausalDiscovery] prices_live.parquet not found — using macro only")
        # Use macro columns as both macro and sector proxy
        sector_prices = macro_pivot.copy().rename(
            columns={c: f"ETF_{c}" for c in macro_pivot.columns}
        )

    return discover_macro_dag(macro_pivot, sector_prices, regime=regime)
```

- [ ] **Step 4: Run the discovery tests**

```bash
.venv/bin/python -m pytest tests/test_causal_discovery.py -v 2>&1 | tail -12
```

Expected: 5 PASSes

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 682 passed, 1 skipped

- [ ] **Step 6: Commit**

```bash
git add ascent/causal/causal_discovery.py tests/test_causal_discovery.py
git commit -m "feat: add causal_discovery.py — PC algorithm macro DAG builder"
```

---

## Task 4: `ascent/causal/dag_builder.py` — per-symbol Haiku graph builder

**Files:**
- Create: `ascent/causal/dag_builder.py`
- Create: `tests/test_dag_builder.py`

### Background

For each portfolio holding, Haiku builds a JSON causal graph cached by `(symbol, quarter_end)`. The cache lives at `data_cache/causal_graphs/{symbol}_{quarter_end}.json`. A cache hit skips the LLM call entirely. Quarter end is derived from the most recent earnings date in `data_cache/earnings.parquet` (or today's quarter end as fallback).

The Haiku prompt receives: latest fundamental ratios (from `data_cache/fundamentals.parquet`), the most recent earnings transcript summary (from `data_cache/altdata_transcripts.parquet` if available), and the most recent 10-K summary (from `data_cache/altdata_sec.parquet` if available). It returns a list of 1-3 causal mechanisms.

- [ ] **Step 1: Write failing tests**

Create `tests/test_dag_builder.py`:

```python
# tests/test_dag_builder.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


_MOCK_HAIKU_RESPONSE = json.dumps({
    "mechanisms": [
        {
            "mechanism": "NAND oversupply correction → gross margin expansion → EPS rerating",
            "intervention": "IF NAND spot price +15% from trough THEN WDC gross margin > 40%",
            "falsification_condition": "IF WDC Q3 gross margin < 38%, thesis broken",
            "horizon_days": 63,
            "timing": "catalyst_imminent",
            "mechanism_type": "supply_demand_inflection",
        }
    ]
})


def test_build_graph_returns_schema(tmp_path):
    """build_graph must return a dict with symbol, quarter_end, built_at, mechanisms."""
    from ascent.causal.dag_builder import build_graph

    with patch("ascent.causal.dag_builder.generate_structured", return_value=_MOCK_HAIKU_RESPONSE):
        result = build_graph(
            symbol="WDC",
            quarter_end="2026-03-31",
            fundamental_text="Q0: gross_profitability=0.32, accruals=-0.01",
            transcript_summary="Management discussed NAND market recovery.",
            sec_summary="10-K highlights supply reduction and margin guidance.",
            cache_dir=tmp_path,
        )

    assert result["symbol"] == "WDC"
    assert result["quarter_end"] == "2026-03-31"
    assert "built_at" in result
    assert isinstance(result["mechanisms"], list)
    assert len(result["mechanisms"]) == 1


def test_build_graph_cache_hit_skips_llm(tmp_path):
    """Second call with same (symbol, quarter_end) must skip LLM and return cached data."""
    from ascent.causal.dag_builder import build_graph

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        return _MOCK_HAIKU_RESPONSE

    with patch("ascent.causal.dag_builder.generate_structured", side_effect=mock_generate):
        build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)
        build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    assert call_count[0] == 1, "LLM should only be called once per (symbol, quarter_end)"


def test_build_graph_cache_file_path(tmp_path):
    """Cache file must be at {cache_dir}/{symbol}_{quarter_end}.json."""
    from ascent.causal.dag_builder import build_graph

    with patch("ascent.causal.dag_builder.generate_structured", return_value=_MOCK_HAIKU_RESPONSE):
        build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    expected = tmp_path / "WDC_2026-03-31.json"
    assert expected.exists(), f"Cache file not found at {expected}"


def test_build_graph_mechanism_schema(tmp_path):
    """Each mechanism in the result must have all required fields."""
    from ascent.causal.dag_builder import build_graph

    with patch("ascent.causal.dag_builder.generate_structured", return_value=_MOCK_HAIKU_RESPONSE):
        result = build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    m = result["mechanisms"][0]
    for field in ("mechanism", "intervention", "falsification_condition",
                  "horizon_days", "timing", "mechanism_type"):
        assert field in m, f"Mechanism missing required field: {field}"
    assert m["timing"] in ("priced_in", "not_yet_priced", "catalyst_imminent")
    assert m["mechanism_type"] in (
        "momentum_catalyst", "quality_defensive", "macro_hedge",
        "mean_reversion", "valuation", "supply_demand_inflection",
    )


def test_load_or_build_respects_cache_hit(tmp_path):
    """load_or_build returns cached data without LLM call if cache file exists."""
    from ascent.causal.dag_builder import load_or_build

    # Pre-write a cache file
    cached = {
        "symbol": "AAPL", "quarter_end": "2026-03-31",
        "built_at": "2026-05-01",
        "mechanisms": [{"mechanism": "cached", "intervention": "i",
                        "falsification_condition": "f", "horizon_days": 21,
                        "timing": "not_yet_priced", "mechanism_type": "momentum_catalyst"}]
    }
    (tmp_path / "AAPL_2026-03-31.json").write_text(json.dumps(cached))

    call_count = [0]
    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        return _MOCK_HAIKU_RESPONSE

    with patch("ascent.causal.dag_builder.generate_structured", side_effect=mock_generate):
        result = load_or_build("AAPL", "2026-03-31", cache_dir=tmp_path)

    assert call_count[0] == 0, "LLM must not be called when cache exists"
    assert result["mechanisms"][0]["mechanism"] == "cached"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dag_builder.py -v 2>&1 | tail -10
```

Expected: 5 FAILs — `ModuleNotFoundError`

- [ ] **Step 3: Create `ascent/causal/dag_builder.py`**

```python
"""ascent/causal/dag_builder.py

Per-symbol causal graph builder. Haiku reads fundamental + transcript +
SEC summary and returns 1-3 causal mechanisms per holding.
Cache: data_cache/causal_graphs/{symbol}_{quarter_end}.json
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data_cache/causal_graphs")

_MECHANISM_TYPES = (
    "momentum_catalyst", "quality_defensive", "macro_hedge",
    "mean_reversion", "valuation", "supply_demand_inflection",
)

_SYSTEM_PROMPT = (
    "You are a financial analyst building a causal model for a portfolio holding. "
    "Identify 1-3 causal mechanisms that explain the current investment thesis. "
    "Each mechanism must be falsifiable: state a specific observable condition "
    "that would break the thesis. Base your analysis only on the data provided — "
    "do not use training-data knowledge about the company beyond what is given. "
    "Respond with valid JSON matching the provided schema exactly. No other text."
)

_DAG_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanisms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mechanism": {
                        "type": "string",
                        "description": "One sentence: 'X causes Y via Z'",
                    },
                    "intervention": {
                        "type": "string",
                        "description": "IF [observable trigger] THEN [expected outcome]",
                    },
                    "falsification_condition": {
                        "type": "string",
                        "description": "IF [observable] < [threshold], thesis broken",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "Trading days until falsification check (21, 42, or 63)",
                    },
                    "timing": {
                        "type": "string",
                        "enum": ["priced_in", "not_yet_priced", "catalyst_imminent"],
                    },
                    "mechanism_type": {
                        "type": "string",
                        "enum": list(_MECHANISM_TYPES),
                    },
                },
                "required": [
                    "mechanism", "intervention", "falsification_condition",
                    "horizon_days", "timing", "mechanism_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mechanisms"],
    "additionalProperties": False,
}


def build_graph(
    symbol: str,
    quarter_end: str,
    fundamental_text: str,
    transcript_summary: str,
    sec_summary: str,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Build and cache a causal graph for a single symbol.

    Args:
        symbol: ticker (e.g. "WDC")
        quarter_end: ISO date string for the quarter (e.g. "2026-03-31")
        fundamental_text: formatted fundamental ratios (Q-3 to Q0)
        transcript_summary: short earnings call summary
        sec_summary: short 10-K summary
        cache_dir: directory for JSON cache files (default: DEFAULT_CACHE_DIR)

    Returns:
        Dict with {symbol, quarter_end, built_at, mechanisms[]}
    """
    from ascent.llm.client import generate_structured, HAIKU_MODEL

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{symbol}_{quarter_end}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            log.debug("[DagBuilder] Cache hit: %s %s", symbol, quarter_end)
            return cached
        except Exception:
            pass

    user_prompt = f"""Symbol: {symbol} | Quarter end: {quarter_end}

Fundamental ratios (Q-3 = three quarters ago, Q0 = most recent):
{fundamental_text}

Earnings call summary:
{transcript_summary or 'Not available'}

10-K / SEC filing summary:
{sec_summary or 'Not available'}

Build 1-3 causal mechanisms that explain this company's current investment dynamics.
For each: state a mechanism (X causes Y via Z), an intervention condition, a falsification condition,
a horizon, whether the mechanism is already priced in, and the mechanism type."""

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.3,
            use_cache=True,
            json_schema=_DAG_SCHEMA,
        )
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        mechanisms = parsed.get("mechanisms", [])
    except Exception as exc:
        log.warning("[DagBuilder] Haiku call failed for %s: %s", symbol, exc)
        mechanisms = []

    result = {
        "symbol": symbol,
        "quarter_end": quarter_end,
        "built_at": str(date.today()),
        "mechanisms": mechanisms,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    log.info("[DagBuilder] Built graph for %s: %d mechanisms", symbol, len(mechanisms))
    return result


def load_or_build(
    symbol: str,
    quarter_end: str,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Load cached graph if available; otherwise return empty graph.
    Does NOT trigger a Haiku call — use build_graph() for that.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = Path(cache_dir) / f"{symbol}_{quarter_end}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return {"symbol": symbol, "quarter_end": quarter_end, "built_at": None, "mechanisms": []}


def get_quarter_end(symbol: str) -> str:
    """
    Derive the most recent quarter_end date for a symbol from earnings.parquet.
    Falls back to current calendar quarter end if no earnings data.
    """
    try:
        import pandas as pd
        ep = Path("data_cache/earnings.parquet")
        if ep.exists():
            earnings = pd.read_parquet(ep)
            sym_rows = earnings[earnings["symbol"] == symbol].sort_values("date", ascending=False)
            if not sym_rows.empty:
                last_date = pd.to_datetime(sym_rows.iloc[0]["date"])
                # Round down to quarter end: Mar 31, Jun 30, Sep 30, Dec 31
                month = ((last_date.month - 1) // 3 * 3) + 3
                import calendar
                day = calendar.monthrange(last_date.year, month)[1]
                return f"{last_date.year}-{month:02d}-{day:02d}"
    except Exception:
        pass

    # Fallback: current calendar quarter end
    today = date.today()
    month = ((today.month - 1) // 3 * 3) + 3
    import calendar
    day = calendar.monthrange(today.year, month)[1]
    return f"{today.year}-{month:02d}-{day:02d}"


def build_portfolio_graphs(
    symbols: list,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Build causal graphs for all holdings in the current portfolio.
    Called by weekend_runner. Returns {symbol: graph_dict}.
    """
    from ascent.data.store.parquet_store import ParquetStore

    results = {}
    for symbol in symbols:
        quarter_end = get_quarter_end(symbol)
        if cache_dir is None:
            check_path = DEFAULT_CACHE_DIR / f"{symbol}_{quarter_end}.json"
        else:
            check_path = Path(cache_dir) / f"{symbol}_{quarter_end}.json"

        if check_path.exists():
            log.debug("[DagBuilder] Already have graph for %s %s", symbol, quarter_end)
            results[symbol] = load_or_build(symbol, quarter_end, cache_dir)
            continue

        # Gather context from data caches
        fundamental_text = _get_fundamental_text(symbol)
        transcript_summary = _get_transcript_summary(symbol)
        sec_summary = _get_sec_summary(symbol)

        results[symbol] = build_graph(
            symbol, quarter_end,
            fundamental_text, transcript_summary, sec_summary,
            cache_dir=cache_dir,
        )

    return results


def _get_fundamental_text(symbol: str) -> str:
    try:
        import pandas as pd
        fp = Path("data_cache/fundamentals.parquet")
        if not fp.exists():
            return "No fundamental data available"
        df = pd.read_parquet(fp)
        rows = df[df["symbol"] == symbol].sort_values("date", ascending=False).head(4)
        if rows.empty:
            return "No fundamental data available"
        lines = []
        for i, (_, row) in enumerate(rows.iterrows()):
            cols = [c for c in ["gross_profitability", "accruals_ratio", "asset_growth"]
                    if c in row.index and pd.notna(row[c])]
            vals = ", ".join(f"{c}={row[c]:.3f}" for c in cols)
            lines.append(f"Q-{i}: {vals}")
        return "\n".join(lines) if lines else "No data"
    except Exception:
        return "Fundamental data load error"


def _get_transcript_summary(symbol: str) -> str:
    try:
        import pandas as pd
        tp = Path("data_cache/altdata_transcripts.parquet")
        if not tp.exists():
            return ""
        df = pd.read_parquet(tp)
        rows = df[df["symbol"] == symbol].sort_values("date", ascending=False)
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("summary", ""))[:500]
    except Exception:
        return ""


def _get_sec_summary(symbol: str) -> str:
    try:
        import pandas as pd
        sp = Path("data_cache/altdata_sec.parquet")
        if not sp.exists():
            return ""
        df = pd.read_parquet(sp)
        rows = df[df["symbol"] == symbol].sort_values("date", ascending=False)
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("summary", ""))[:500]
    except Exception:
        return ""
```

- [ ] **Step 4: Run the dag_builder tests**

```bash
.venv/bin/python -m pytest tests/test_dag_builder.py -v 2>&1 | tail -12
```

Expected: 5 PASSes

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 687 passed, 1 skipped

- [ ] **Step 6: Commit**

```bash
git add ascent/causal/dag_builder.py tests/test_dag_builder.py
git commit -m "feat: add dag_builder.py — Haiku per-symbol causal graph builder"
```

---

## PHASE B — Gates 1 and 2

---

## Task 5: `ascent/causal/compatibility.py` — Gate 1 (regime-causal)

**Files:**
- Create: `ascent/causal/compatibility.py`
- Create: `tests/test_causal_compatibility.py`

### Background

Gate 1 is a static dict lookup — no LLM. `calm_bull` allows momentum, supply/demand, and quality mechanisms. Anti-momentum types (`valuation`, `mean_reversion`) are blocked unless `crowding == OVERCROWDED` (handled by the caller). `crisis` only allows `macro_hedge`. The gate returns `True/False` — callers decide what to do with incompatible mechanisms.

- [ ] **Step 1: Write failing tests**

Create `tests/test_causal_compatibility.py`:

```python
# tests/test_causal_compatibility.py
import pytest


def test_momentum_catalyst_compatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("momentum_catalyst", "calm_bull") is True


def test_valuation_incompatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("valuation", "calm_bull") is False


def test_mean_reversion_incompatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("mean_reversion", "calm_bull") is False


def test_supply_demand_compatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("supply_demand_inflection", "calm_bull") is True


def test_quality_defensive_compatible_stressed():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("quality_defensive", "stressed") is True


def test_momentum_catalyst_incompatible_crisis():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("momentum_catalyst", "crisis") is False


def test_macro_hedge_compatible_crisis():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("macro_hedge", "crisis") is True


def test_macro_hedge_compatible_all_regimes():
    from ascent.causal.compatibility import regime_compatible
    for regime in ("calm_bull", "stressed", "crisis", "neutral", "uncertain"):
        assert regime_compatible("macro_hedge", regime) is True, \
            f"macro_hedge should be compatible with {regime}"


def test_unknown_regime_defaults_to_quality_defensive_only():
    from ascent.causal.compatibility import regime_compatible
    # Unknown regimes fall back to conservative (only quality_defensive + macro_hedge)
    assert regime_compatible("quality_defensive", "euphoric") is True
    assert regime_compatible("momentum_catalyst", "euphoric") is False


def test_filter_mechanisms_returns_compatible_only():
    from ascent.causal.compatibility import filter_mechanisms

    mechanisms = [
        {"mechanism_type": "momentum_catalyst", "timing": "catalyst_imminent"},
        {"mechanism_type": "valuation",          "timing": "not_yet_priced"},
        {"mechanism_type": "supply_demand_inflection", "timing": "not_yet_priced"},
        {"mechanism_type": "mean_reversion",     "timing": "priced_in"},
    ]
    compatible = filter_mechanisms(mechanisms, regime="calm_bull")
    types = [m["mechanism_type"] for m in compatible]
    assert "valuation" not in types
    assert "mean_reversion" not in types
    assert "momentum_catalyst" in types
    assert "supply_demand_inflection" in types
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_causal_compatibility.py -v 2>&1 | tail -12
```

Expected: 10 FAILs

- [ ] **Step 3: Create `ascent/causal/compatibility.py`**

```python
"""ascent/causal/compatibility.py

Gate 1: regime-causal mechanism compatibility.
Static dict lookup — no LLM, no external calls.
"""
from typing import List

# Mechanism types allowed per regime.
# macro_hedge is allowed in every regime — always preserve it.
_REGIME_ALLOWED: dict = {
    "calm_bull": {
        "momentum_catalyst",
        "supply_demand_inflection",
        "quality_defensive",
        "macro_hedge",
    },
    "stressed": {
        "quality_defensive",
        "macro_hedge",
        "mean_reversion",       # short-term reversion can work in stressed markets
    },
    "crisis": {
        "macro_hedge",
    },
    "neutral": {
        "momentum_catalyst",
        "quality_defensive",
        "supply_demand_inflection",
        "macro_hedge",
        "mean_reversion",
    },
    "uncertain": {
        "quality_defensive",
        "momentum_catalyst",
        "macro_hedge",
    },
}

# Conservative fallback for unknown regimes
_FALLBACK_ALLOWED = {"quality_defensive", "macro_hedge"}


def regime_compatible(mechanism_type: str, regime: str) -> bool:
    """
    Return True if a mechanism type is compatible with the current regime.

    Args:
        mechanism_type: one of the six mechanism types from dag_builder.py
        regime: current regime label (e.g. "calm_bull", "stressed", "crisis")

    Returns:
        True if the mechanism is allowed in this regime, False otherwise.
    """
    allowed = _REGIME_ALLOWED.get(regime, _FALLBACK_ALLOWED)
    return mechanism_type in allowed


def filter_mechanisms(mechanisms: List[dict], regime: str) -> List[dict]:
    """
    Filter a list of mechanism dicts to those compatible with the current regime.
    Each dict must have a 'mechanism_type' key.

    Returns:
        Filtered list (preserves order, no mutation of originals).
    """
    return [m for m in mechanisms if regime_compatible(m.get("mechanism_type", ""), regime)]
```

- [ ] **Step 4: Run the compatibility tests**

```bash
.venv/bin/python -m pytest tests/test_causal_compatibility.py -v 2>&1 | tail -12
```

Expected: 10 PASSes

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 697 passed, 1 skipped

- [ ] **Step 6: Commit**

```bash
git add ascent/causal/compatibility.py tests/test_causal_compatibility.py
git commit -m "feat: add compatibility.py — Gate 1 regime-causal mechanism filter"
```

---

## Task 6: AI PM Phase 1 integration — gates 1, 2, velocity, `get_causal_graph` tool

**Files:**
- Modify: `agents/ai_pm_agent.py`
- Create: `tests/test_ai_pm_prethesis_causal.py`

### Background

Three changes to `run_ai_pm_prethesis()`:

1. **`get_causal_graph` tool** added to `PRE_THESIS_TOOLS` — AI PM can call it for any symbol to read the cached graph. Returns the JSON graph as a formatted string.

2. **Context injection** — before the Phase 1 tool loop starts, build a ranked list of `(symbol, velocity, timing, mechanism_summary)` tuples for current holdings and inject into the system prompt. Gate 1 filtering (compatibility) and Gate 2 filtering (priced_in exclusion) happen here in Python.

3. **`AIPreThesis.causal_mechanisms` population** — after `propose_prethesis` is called, assemble `CausalMechanism` objects from the cached graphs for each symbol in `high_conviction_names`. Only mechanisms that survived Gate 1 + Gate 2 are included. Store on the returned `AIPreThesis`.

The `causal_mechanisms` field is added to `AIPreThesis` in `agents/ai_pm_agent.py` (not `types.py`, since `AIPreThesis` lives there).

- [ ] **Step 1: Write failing tests**

Create `tests/test_ai_pm_prethesis_causal.py`:

```python
# tests/test_ai_pm_prethesis_causal.py
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_aiprethesis_has_causal_mechanisms_field():
    """AIPreThesis must have a causal_mechanisms field (list, default empty)."""
    from agents.ai_pm_agent import AIPreThesis
    pt = AIPreThesis(
        macro_view="rates falling",
        regime_interpretation="calm_bull",
        high_conviction_names=[{"symbol": "WDC", "thesis": "NAND recovery"}],
        names_to_avoid=[],
        sector_tilts=[],
    )
    assert hasattr(pt, "causal_mechanisms"), "AIPreThesis missing causal_mechanisms field"
    assert isinstance(pt.causal_mechanisms, list)


def test_get_causal_graph_tool_in_pre_thesis_tools():
    """get_causal_graph must be registered in PRE_THESIS_TOOLS."""
    from agents.ai_pm_agent import PRE_THESIS_TOOLS
    names = {t["name"] for t in PRE_THESIS_TOOLS}
    assert "get_causal_graph" in names, \
        "get_causal_graph tool must be in PRE_THESIS_TOOLS"


def test_assemble_causal_mechanisms_from_cache(tmp_path):
    """
    _assemble_causal_mechanisms must load graphs from cache, apply gate 1+2 filtering,
    and return CausalMechanism objects for regime-compatible, non-priced-in mechanisms.
    """
    from ascent.config.types import CausalMechanism
    import agents.ai_pm_agent as mod

    # Write a fake cache file for WDC
    graph = {
        "symbol": "WDC", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [
            {
                "mechanism": "NAND recovery → margins", "intervention": "IF nand +15%",
                "falsification_condition": "IF margin < 38%",
                "horizon_days": 63, "timing": "catalyst_imminent",
                "mechanism_type": "supply_demand_inflection",
            },
            {
                "mechanism": "DCF compression", "intervention": "IF rates fall",
                "falsification_condition": "IF rate > 5%",
                "horizon_days": 42, "timing": "not_yet_priced",
                "mechanism_type": "valuation",   # blocked in calm_bull
            },
            {
                "mechanism": "Already rallied", "intervention": "IF eps beats",
                "falsification_condition": "IF price flat",
                "horizon_days": 21, "timing": "priced_in",  # blocked by gate 2
                "mechanism_type": "momentum_catalyst",
            },
        ]
    }
    (tmp_path / "WDC_2026-03-31.json").write_text(json.dumps(graph))

    with patch("ascent.causal.dag_builder.DEFAULT_CACHE_DIR", tmp_path):
        with patch("ascent.causal.dag_builder.get_quarter_end", return_value="2026-03-31"):
            result = mod._assemble_causal_mechanisms(
                high_conviction_symbols=["WDC"],
                regime="calm_bull",
                cache_dir=tmp_path,
            )

    assert len(result) == 1, f"Expected 1 (valuation+priced_in blocked), got {len(result)}"
    assert isinstance(result[0], CausalMechanism)
    assert result[0].symbol == "WDC"
    assert result[0].timing == "catalyst_imminent"
    assert result[0].regime_compatible is True


def test_build_velocity_context_returns_sorted_list(tmp_path):
    """_build_velocity_context must return candidates sorted by timing priority desc."""
    import agents.ai_pm_agent as mod

    graph_a = {
        "symbol": "AAPL", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [{"mechanism": "m", "intervention": "i", "falsification_condition": "f",
                        "horizon_days": 21, "timing": "not_yet_priced",
                        "mechanism_type": "momentum_catalyst"}]
    }
    graph_b = {
        "symbol": "WDC", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [{"mechanism": "m2", "intervention": "i2", "falsification_condition": "f2",
                        "horizon_days": 63, "timing": "catalyst_imminent",
                        "mechanism_type": "supply_demand_inflection"}]
    }
    (tmp_path / "AAPL_2026-03-31.json").write_text(json.dumps(graph_a))
    (tmp_path / "WDC_2026-03-31.json").write_text(json.dumps(graph_b))

    with patch("ascent.causal.dag_builder.DEFAULT_CACHE_DIR", tmp_path):
        with patch("ascent.causal.dag_builder.get_quarter_end", return_value="2026-03-31"):
            lines = mod._build_velocity_context(
                symbols=["AAPL", "WDC"],
                regime="calm_bull",
                cache_dir=tmp_path,
            )

    # WDC (catalyst_imminent) should appear before AAPL (not_yet_priced)
    text = "\n".join(lines)
    assert "WDC" in text
    wdc_pos = text.index("WDC")
    aapl_pos = text.index("AAPL")
    assert wdc_pos < aapl_pos, "catalyst_imminent must rank higher than not_yet_priced"
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_prethesis_causal.py -v 2>&1 | tail -12
```

Expected: 4 FAILs

- [ ] **Step 3: Add `causal_mechanisms` field to `AIPreThesis`**

In `agents/ai_pm_agent.py`, find the `AIPreThesis` dataclass (line ~51) and add the field:

```python
# Add this import at the top if not already there
from typing import List, Dict, Optional, Any
# Also import CausalMechanism
from ascent.config.types import AgentOutput, CausalMechanism
```

Update `AIPreThesis`:

```python
@dataclass
class AIPreThesis:
    """Output of Phase 1 — original AI PM thesis formed before seeing quant output."""
    macro_view: str
    regime_interpretation: str
    high_conviction_names: List[Dict]
    names_to_avoid: List[Dict]
    sector_tilts: List[Dict]
    regime_assessment: Dict = field(default_factory=dict)
    sleeve_weight_prior: Dict = field(default_factory=dict)
    market_character: str = ""
    raw: Dict = field(default_factory=dict)
    causal_mechanisms: List["CausalMechanism"] = field(default_factory=list)  # Phase B addition
```

- [ ] **Step 4: Add `get_causal_graph` tool to `AI_PM_TOOLS` and `PRE_THESIS_TOOLS`**

In `agents/ai_pm_agent.py`, find `AI_PM_TOOLS` (the list starting around line 71). Add this entry after `get_crowding_signal`:

```python
    {
        "name": "get_causal_graph",
        "description": (
            "Look up the cached causal graph for a portfolio holding. "
            "The graph contains 1-3 causal mechanisms explaining why the stock "
            "should move, with timing (priced_in / not_yet_priced / catalyst_imminent) "
            "and falsification conditions. Use before making a high-conviction call "
            "to understand the causal thesis, not just correlation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
```

Then add `"get_causal_graph"` to the set in `PRE_THESIS_TOOLS`:

```python
PRE_THESIS_TOOLS = [
    t for t in AI_PM_TOOLS
    if t["name"] in {
        "get_rebalance_brief", "get_regime_state", "get_macro_data",
        "get_regime_memory", "get_live_news", "get_analyst_estimates",
        "get_sec_signal", "get_transcript_signal", "get_earnings_signal",
        "get_narrative_shift", "get_scenario_plan", "get_weekend_research",
        "get_crowding_signal", "get_attribution_history", "get_calibration_report",
        "get_causal_graph",   # ← add this
    }
] + [_PROPOSE_PRETHESIS_TOOL]
```

- [ ] **Step 5: Add the tool executor for `get_causal_graph`**

Find `_make_prethesis_executor` (line ~1307) and add the handler inside the executor function dict:

```python
def _tool_get_causal_graph(inputs: dict) -> str:
    """Return the cached causal graph for a symbol, or a 'not available' message."""
    from ascent.causal.dag_builder import load_or_build, get_quarter_end
    symbol = inputs.get("symbol", "").upper()
    if not symbol:
        return "Error: symbol required"
    quarter_end = get_quarter_end(symbol)
    graph = load_or_build(symbol, quarter_end)
    if not graph.get("mechanisms"):
        return f"No causal graph available for {symbol}. Build one by running the weekend pipeline."
    lines = [f"Causal graph for {symbol} (quarter_end={quarter_end}):"]
    for i, m in enumerate(graph["mechanisms"], 1):
        lines.append(
            f"\n[Mechanism {i}] {m.get('mechanism', 'N/A')}\n"
            f"  Timing: {m.get('timing', 'N/A')}\n"
            f"  Intervention: {m.get('intervention', 'N/A')}\n"
            f"  Falsification: {m.get('falsification_condition', 'N/A')}\n"
            f"  Horizon: {m.get('horizon_days', 'N/A')} trading days"
        )
    return "\n".join(lines)
```

In the executor dict inside `_make_prethesis_executor`, add:

```python
"get_causal_graph": lambda i: _tool_get_causal_graph(i),
```

- [ ] **Step 6: Add `_assemble_causal_mechanisms` and `_build_velocity_context` helper functions**

Add these two functions before `run_ai_pm_prethesis()`:

```python
def _assemble_causal_mechanisms(
    high_conviction_symbols: list,
    regime: str,
    cache_dir=None,
) -> list:
    """
    After propose_prethesis, assemble CausalMechanism objects for all
    high-conviction symbols. Applies Gate 1 (compatibility) + Gate 2 (priced_in).

    Returns list[CausalMechanism] — only regime-compatible, not-yet-priced or
    catalyst_imminent mechanisms included.
    """
    from ascent.causal.dag_builder import load_or_build, get_quarter_end
    from ascent.causal.compatibility import regime_compatible
    from ascent.causal.velocity import mechanism_velocity_score
    from ascent.config.types import CausalMechanism

    results = []
    for symbol in high_conviction_symbols:
        quarter_end = get_quarter_end(symbol)
        graph = load_or_build(symbol, quarter_end, cache_dir)
        for m in graph.get("mechanisms", []):
            # Gate 1 — regime compatibility
            mtype = m.get("mechanism_type", "")
            if not regime_compatible(mtype, regime):
                continue
            # Gate 2 — exclude priced_in mechanisms from thesis
            if m.get("timing") == "priced_in":
                continue
            results.append(CausalMechanism(
                symbol=symbol,
                mechanism=m.get("mechanism", ""),
                intervention=m.get("intervention", ""),
                falsification_condition=m.get("falsification_condition", ""),
                horizon_days=int(m.get("horizon_days", 63)),
                timing=m.get("timing", "not_yet_priced"),
                velocity=0.0,  # velocity snapshot at build time (not recomputed here)
                mechanism_type=mtype,
                regime_compatible=True,
            ))
    return results


_TIMING_PRIORITY = {"catalyst_imminent": 2, "not_yet_priced": 1, "priced_in": 0}


def _build_velocity_context(
    symbols: list,
    regime: str,
    cache_dir=None,
) -> list:
    """
    Build a ranked list of causal context lines for injection into Phase 1 prompt.
    Returns list of strings, sorted by timing priority (catalyst_imminent first).
    """
    from ascent.causal.dag_builder import load_or_build, get_quarter_end
    from ascent.causal.compatibility import regime_compatible

    candidates = []
    for symbol in symbols:
        quarter_end = get_quarter_end(symbol)
        graph = load_or_build(symbol, quarter_end, cache_dir)
        for m in graph.get("mechanisms", []):
            mtype = m.get("mechanism_type", "")
            timing = m.get("timing", "not_yet_priced")
            if not regime_compatible(mtype, regime):
                continue
            if timing == "priced_in":
                continue
            priority = _TIMING_PRIORITY.get(timing, 0)
            candidates.append((priority, symbol, m.get("mechanism", ""), timing))

    candidates.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for _, symbol, mechanism, timing in candidates:
        lines.append(f"  {symbol} [{timing}]: {mechanism}")
    return lines
```

- [ ] **Step 7: Inject causal context + populate causal_mechanisms in `run_ai_pm_prethesis()`**

Find `run_ai_pm_prethesis()` (line ~1398). After the existing portfolio symbols are known (after loading `merged_weights.json` or wherever symbols are gathered), add context injection.

The key change is: (a) build velocity context for current portfolio, (b) append it to the user prompt, (c) after prethesis is stored, populate `causal_mechanisms`.

Find the section in `run_ai_pm_prethesis()` where the tool loop call is made (around line 1411) and update it to inject context:

```python
    # Build causal context for current holdings
    _current_regime = "calm_bull"  # will be overridden by get_regime_state tool call
    try:
        from pathlib import Path as _Path
        import json as _json
        rfile = _Path("dashboard/regime_signal.json")
        if rfile.exists():
            _current_regime = _json.loads(rfile.read_text()).get("regime", "calm_bull")
    except Exception:
        pass

    from ascent.causal.dag_builder import load_or_build, get_quarter_end, DEFAULT_CACHE_DIR
    try:
        _portfolio_symbols = [
            s for s in __import__("json").loads(
                __import__("pathlib").Path("data_cache/merged_weights.json").read_text()
            ).keys()
        ] if __import__("pathlib").Path("data_cache/merged_weights.json").exists() else []
    except Exception:
        _portfolio_symbols = []

    _causal_context_lines = _build_velocity_context(_portfolio_symbols, _current_regime)
    _causal_context = ""
    if _causal_context_lines:
        _causal_context = (
            "\n\n══ CAUSAL INTELLIGENCE (filtered: regime-compatible, catalyst not yet priced) ══\n"
            "These are the top causal mechanisms for current holdings, ranked by timing priority.\n"
            "Use them to AMPLIFY where mechanism + quant agree. Call get_causal_graph(symbol) "
            "for full falsification conditions before concentrating.\n"
            + "\n".join(_causal_context_lines)
        )
```

Then append `_causal_context` to the user_prompt that is passed to `tool_completion`. Find the line with `user_prompt=` in `run_ai_pm_prethesis()` and append it:

```python
        user_prompt=(
            "You are building Ascent Capital's pre-rebalance investment thesis. "
            "Read macro, SEC filings, earnings calls, narratives, and crowding signals. "
            "Form your original thesis (8-15 names with written reasons) BEFORE seeing quant output. "
            "When ready, call propose_prethesis to seal your thesis."
            + _causal_context
        ),
```

After `result_store` is populated (after the tool loop), populate `causal_mechanisms`:

```python
    # ... (existing code that builds AIPreThesis from result_store)
    prethesis = AIPreThesis(
        ...
    )
    # Populate causal_mechanisms after gate 1+2 filtering
    try:
        prethesis.causal_mechanisms = _assemble_causal_mechanisms(
            high_conviction_symbols=prethesis.conviction_symbols,
            regime=_current_regime,
        )
        log.info("[AIPMAgent] Pre-thesis: %d causal mechanisms assembled", len(prethesis.causal_mechanisms))
    except Exception as exc:
        log.warning("[AIPMAgent] Causal mechanism assembly failed: %s", exc)
    return prethesis
```

- [ ] **Step 8: Run the prethesis causal tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_prethesis_causal.py -v 2>&1 | tail -12
```

Expected: 4 PASSes

- [ ] **Step 9: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 701+ passed, 1 skipped, 0 failures

- [ ] **Step 10: Commit**

```bash
git add agents/ai_pm_agent.py tests/test_ai_pm_prethesis_causal.py
git commit -m "feat: wire Gate 1+2 + causal context into AI PM Phase 1 prethesis"
```

---

## Task 7: Weekend runner — add causal pipeline jobs

**Files:**
- Modify: `ascent/monitoring/weekend_runner.py`

### Background

Two new jobs added after the existing AI PM research job:
1. `causal_macro_dag` — runs `causal_discovery.run_discovery()`, writes `macro_causal_dag.json`. ~30s, no LLM.
2. `causal_graph_builder` — runs `dag_builder.build_portfolio_graphs()` for all current holdings. Skips cache-hit symbols. ~$0.015/run.

Both run once per weekend. Neither blocks existing jobs on failure.

- [ ] **Step 1: No separate test — the existing weekend_runner tests cover `_run_job` mechanics. Just add the jobs and verify the suite stays green.**

Open `ascent/monitoring/weekend_runner.py`. After job 7 (`ai_pm_research`, line ~380), add:

```python
    # 7b. Causal macro DAG — once per weekend (~30s, no LLM)
    if _run_job("causal_macro_dag", _job_causal_macro_dag, once_per_weekend=True):
        completed.append("causal_macro_dag")

    # 7c. Causal graph builder — once per weekend (~$0.015)
    if _run_job(
        "causal_graph_builder",
        lambda: _job_causal_graph_builder(portfolio_symbols),
        once_per_weekend=True,
    ):
        completed.append("causal_graph_builder")
```

Also update the total job count in the print statement from 11 to 13:

```python
    print(f"# Weekend run complete: {len(completed)}/{13} jobs succeeded")
```

Add the two job functions (before `run_weekend`):

```python
def _job_causal_macro_dag() -> None:
    """Run causal discovery on FRED + sector ETFs → macro_causal_dag.json."""
    from ascent.causal.causal_discovery import run_discovery
    try:
        from pathlib import Path as _Path
        import json as _json
        rfile = _Path("dashboard/regime_signal.json")
        regime = "calm_bull"
        if rfile.exists():
            regime = _json.loads(rfile.read_text()).get("regime", "calm_bull")
    except Exception:
        regime = "calm_bull"

    dag = run_discovery(regime=regime)
    n_edges = len(dag.get("edges", []))
    print(f"  [CausalDiscovery] DAG built: {n_edges} edges, regime={regime}")


def _job_causal_graph_builder(portfolio_symbols: list) -> None:
    """Build/refresh Haiku causal graphs for all current holdings."""
    from ascent.causal.dag_builder import build_portfolio_graphs
    if not portfolio_symbols:
        print("  [DagBuilder] No portfolio symbols — skipping")
        return
    results = build_portfolio_graphs(portfolio_symbols)
    print(f"  [DagBuilder] Processed {len(results)} symbols")
```

- [ ] **Step 2: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: same pass count, 0 failures

- [ ] **Step 3: Commit**

```bash
git add ascent/monitoring/weekend_runner.py
git commit -m "feat: add causal_macro_dag + causal_graph_builder jobs to weekend runner"
```

---

## PHASE C — Tracker and Gate 4

---

## Task 8: `ascent/causal/tracker.py` — predictions log + early exit

**Files:**
- Create: `ascent/causal/tracker.py`
- Create: `tests/test_causal_tracker.py`

### Background

The tracker writes one prediction record per causal mechanism to `logs/causal_predictions.jsonl` on rebalance day. Weekly, `check_outcomes()` reads pending predictions, checks if the horizon has passed, and marks `outcome` as `confirmed` or `falsified`. Daily (non-rebalance), `check_early_exits()` flags mechanisms that have broken early based on price movement since rebalance.

Early exit heuristic (pure Python, no LLM):
- `catalyst_imminent`: if price return from rebalance_date to today < -8%, set `early_exit=True`
- `not_yet_priced`: if >70% of horizon has elapsed AND price return < -5%, set `early_exit=True`
- `outcome` field: `confirmed` if price return > +5% at horizon, `falsified` if < -5%, else `neutral`

`check_early_exits()` returns a list of symbols that have active early-exit flags.

- [ ] **Step 1: Write failing tests**

Create `tests/test_causal_tracker.py`:

```python
# tests/test_causal_tracker.py
import json
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch


def _write_prediction(log_path: Path, symbol: str, timing: str,
                      rebalance_date: str, horizon_days: int,
                      velocity: float = 0.5) -> None:
    record = {
        "symbol": symbol,
        "mechanism": f"Test mechanism for {symbol}",
        "intervention": "IF x THEN y",
        "falsification_condition": "IF price < -8% thesis broken",
        "horizon_days": horizon_days,
        "rebalance_date": rebalance_date,
        "timing": timing,
        "velocity": velocity,
        "regime_compatible": True,
        "outcome": "pending",
        "early_exit": False,
        "checked_date": None,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def test_write_predictions_creates_jsonl(tmp_path):
    """write_predictions must create the log file with correct records."""
    from ascent.config.types import CausalMechanism
    from ascent.causal.tracker import write_predictions

    log_path = tmp_path / "causal_predictions.jsonl"
    mechanisms = [
        CausalMechanism(
            symbol="WDC", mechanism="NAND recovery", intervention="IF nand +15%",
            falsification_condition="IF margin < 38%", horizon_days=63,
            timing="catalyst_imminent", velocity=0.72,
            mechanism_type="supply_demand_inflection", regime_compatible=True,
        )
    ]
    write_predictions(mechanisms, rebalance_date="2026-06-15", log_path=log_path)

    assert log_path.exists()
    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert len(records) == 1
    r = records[0]
    assert r["symbol"] == "WDC"
    assert r["timing"] == "catalyst_imminent"
    assert r["outcome"] == "pending"
    assert r["early_exit"] is False
    assert r["rebalance_date"] == "2026-06-15"


def test_check_early_exits_flags_catalyst_imminent_on_large_drawdown(tmp_path):
    """catalyst_imminent with price -10% must be flagged early_exit=True."""
    from ascent.causal.tracker import check_early_exits

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=5)).isoformat()
    _write_prediction(log_path, "WDC", "catalyst_imminent", rebalance_date, horizon_days=63)

    # Mock price return as -10% (below -8% threshold for catalyst_imminent)
    with patch("ascent.causal.tracker._get_price_return", return_value=-0.10):
        early_exits = check_early_exits(log_path=log_path)

    assert "WDC" in early_exits, "WDC should be flagged for early exit"


def test_check_early_exits_no_flag_for_small_drawdown(tmp_path):
    """catalyst_imminent with price -3% must NOT be flagged."""
    from ascent.causal.tracker import check_early_exits

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=5)).isoformat()
    _write_prediction(log_path, "AAPL", "catalyst_imminent", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=-0.03):
        early_exits = check_early_exits(log_path=log_path)

    assert "AAPL" not in early_exits, "AAPL should NOT be flagged at -3% drawdown"


def test_check_outcomes_marks_confirmed_after_horizon(tmp_path):
    """Prediction past horizon with +7% return should be marked 'confirmed'."""
    from ascent.causal.tracker import check_outcomes

    log_path = tmp_path / "causal_predictions.jsonl"
    # rebalance_date far enough in the past that horizon has passed
    rebalance_date = (date.today() - timedelta(days=70)).isoformat()
    _write_prediction(log_path, "WDC", "catalyst_imminent", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=0.07):
        check_outcomes(log_path=log_path)

    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert records[0]["outcome"] == "confirmed"
    assert records[0]["checked_date"] is not None


def test_check_outcomes_marks_falsified_after_horizon(tmp_path):
    """Prediction past horizon with -7% return should be marked 'falsified'."""
    from ascent.causal.tracker import check_outcomes

    log_path = tmp_path / "causal_predictions.jsonl"
    rebalance_date = (date.today() - timedelta(days=70)).isoformat()
    _write_prediction(log_path, "AMD", "not_yet_priced", rebalance_date, horizon_days=63)

    with patch("ascent.causal.tracker._get_price_return", return_value=-0.07):
        check_outcomes(log_path=log_path)

    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert records[0]["outcome"] == "falsified"


def test_get_track_record_counts_outcomes(tmp_path):
    """get_track_record must return counts of total/confirmed/falsified and accuracy_pct."""
    from ascent.causal.tracker import write_predictions, get_track_record
    from ascent.config.types import CausalMechanism

    log_path = tmp_path / "causal_predictions.jsonl"
    # Write one confirmed + one falsified directly
    _write_prediction(log_path, "A", "catalyst_imminent", "2026-01-01", 63)
    _write_prediction(log_path, "B", "not_yet_priced",   "2026-01-01", 63)

    # Manually update outcomes in the file
    records = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    records[0]["outcome"] = "confirmed"
    records[1]["outcome"] = "falsified"
    log_path.write_text("\n".join(json.dumps(r) for r in records))

    tr = get_track_record(log_path=log_path)
    assert tr["total"] == 2
    assert tr["confirmed"] == 1
    assert tr["falsified"] == 1
    assert tr["accuracy_pct"] == 50.0
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/python -m pytest tests/test_causal_tracker.py -v 2>&1 | tail -12
```

Expected: 7 FAILs — `ModuleNotFoundError`

- [ ] **Step 3: Create `ascent/causal/tracker.py`**

```python
"""ascent/causal/tracker.py

Causal predictions log writer and outcome tracker.

write_predictions() — called on rebalance day, writes one record per mechanism.
check_outcomes()    — called weekly, marks outcome for past-horizon predictions.
check_early_exits() — called daily (non-rebalance), returns symbols to cut early.
get_track_record()  — called at Phase 2 start, returns accuracy stats.
"""
from __future__ import annotations

import json
import logging
import tempfile
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs/causal_predictions.jsonl")

# Thresholds for early exit and outcome classification
_EARLY_EXIT_CATALYST_THRESHOLD = -0.08   # -8% for catalyst_imminent
_EARLY_EXIT_NOTYET_THRESHOLD   = -0.05   # -5% + >70% horizon elapsed
_OUTCOME_CONFIRMED_THRESHOLD   = 0.05    # +5% → confirmed
_OUTCOME_FALSIFIED_THRESHOLD   = -0.05   # -5% → falsified


def write_predictions(
    mechanisms: list,
    rebalance_date: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Append one prediction record per CausalMechanism to the log file.
    Safe to call multiple times — appends only (does not overwrite).

    Args:
        mechanisms: list of CausalMechanism objects
        rebalance_date: ISO date string for this rebalance
        log_path: path to jsonl log (default: DEFAULT_LOG_PATH)
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        for m in mechanisms:
            record = {
                "symbol": m.symbol,
                "mechanism": m.mechanism,
                "intervention": m.intervention,
                "falsification_condition": m.falsification_condition,
                "horizon_days": m.horizon_days,
                "rebalance_date": rebalance_date,
                "timing": m.timing,
                "velocity": m.velocity,
                "regime_compatible": m.regime_compatible,
                "outcome": "pending",
                "early_exit": False,
                "checked_date": None,
            }
            f.write(json.dumps(record) + "\n")

    log.info("[CausalTracker] Wrote %d predictions for rebalance %s", len(mechanisms), rebalance_date)


def _read_records(log_path: Path) -> list:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text().strip().split("\n"):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _write_records(records: list, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=log_path.parent, suffix=".tmp"))
    tmp.write_text("\n".join(json.dumps(r) for r in records))
    os.replace(tmp, log_path)


def _get_price_return(symbol: str, from_date: str) -> float:
    """
    Compute price return for symbol from from_date to today.
    Returns 0.0 if data unavailable.
    """
    try:
        import pandas as pd
        pp = Path("data_cache/prices_live.parquet")
        if not pp.exists():
            return 0.0
        prices = pd.read_parquet(pp)
        if "symbol" in prices.columns:
            sym_prices = prices[prices["symbol"] == symbol].copy()
            sym_prices["date"] = pd.to_datetime(sym_prices["date"])
            sym_prices = sym_prices.sort_values("date")
        else:
            sym_prices = prices[[c for c in prices.columns if symbol in str(c)]].copy()
            if sym_prices.empty:
                return 0.0

        from_dt = pd.to_datetime(from_date)
        after = sym_prices[sym_prices["date"] >= from_dt]
        if len(after) < 2:
            return 0.0

        close_col = "close" if "close" in after.columns else after.columns[-1]
        start_price = after.iloc[0][close_col]
        end_price   = after.iloc[-1][close_col]
        if start_price == 0:
            return 0.0
        return float((end_price - start_price) / start_price)
    except Exception as exc:
        log.debug("[CausalTracker] Price fetch failed for %s: %s", symbol, exc)
        return 0.0


def check_early_exits(log_path: Optional[Path] = None) -> List[str]:
    """
    Check all pending predictions for early exit conditions.
    Updates early_exit flag in the log file.

    Returns:
        List of symbols with active early_exit flags.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    records = _read_records(Path(log_path))
    today = date.today()
    early_exit_symbols = []
    changed = False

    for r in records:
        if r.get("outcome") != "pending":
            continue
        if r.get("early_exit"):
            early_exit_symbols.append(r["symbol"])
            continue

        symbol = r["symbol"]
        timing = r.get("timing", "not_yet_priced")
        rebalance_date = r.get("rebalance_date", str(today))
        horizon_days = int(r.get("horizon_days", 63))

        price_return = _get_price_return(symbol, rebalance_date)

        should_exit = False
        if timing == "catalyst_imminent":
            should_exit = price_return < _EARLY_EXIT_CATALYST_THRESHOLD
        elif timing == "not_yet_priced":
            rebalance_dt = date.fromisoformat(rebalance_date)
            elapsed_days = (today - rebalance_dt).days
            elapsed_fraction = elapsed_days / max(horizon_days, 1)
            should_exit = (elapsed_fraction > 0.70 and price_return < _EARLY_EXIT_NOTYET_THRESHOLD)

        if should_exit:
            r["early_exit"] = True
            r["checked_date"] = str(today)
            changed = True
            early_exit_symbols.append(symbol)
            log.info(
                "[CausalTracker] Early exit flagged: %s (timing=%s, return=%.1f%%)",
                symbol, timing, price_return * 100,
            )

    if changed:
        _write_records(records, Path(log_path))

    return list(set(early_exit_symbols))


def check_outcomes(log_path: Optional[Path] = None) -> None:
    """
    For all past-horizon pending predictions, classify outcome as
    'confirmed', 'falsified', or 'neutral' based on realized price return.
    Updates the log file in place.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    records = _read_records(Path(log_path))
    today = date.today()
    changed = False

    for r in records:
        if r.get("outcome") != "pending":
            continue

        rebalance_date = r.get("rebalance_date", str(today))
        horizon_days = int(r.get("horizon_days", 63))
        rebalance_dt = date.fromisoformat(rebalance_date)
        # Horizon elapsed with 5-day buffer
        if (today - rebalance_dt).days < horizon_days + 5:
            continue

        symbol = r["symbol"]
        price_return = _get_price_return(symbol, rebalance_date)
        if price_return >= _OUTCOME_CONFIRMED_THRESHOLD:
            r["outcome"] = "confirmed"
        elif price_return <= _OUTCOME_FALSIFIED_THRESHOLD:
            r["outcome"] = "falsified"
        else:
            r["outcome"] = "neutral"
        r["checked_date"] = str(today)
        changed = True
        log.info(
            "[CausalTracker] Outcome for %s: %s (return=%.1f%%)",
            symbol, r["outcome"], price_return * 100,
        )

    if changed:
        _write_records(records, Path(log_path))


def get_track_record(log_path: Optional[Path] = None) -> dict:
    """
    Return accuracy statistics for all resolved predictions.

    Returns:
        {total, confirmed, falsified, neutral, pending, accuracy_pct}
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    records = _read_records(Path(log_path))
    counts = {"confirmed": 0, "falsified": 0, "neutral": 0, "pending": 0}
    for r in records:
        outcome = r.get("outcome", "pending")
        counts[outcome] = counts.get(outcome, 0) + 1

    resolved = counts["confirmed"] + counts["falsified"] + counts["neutral"]
    accuracy_pct = (
        round(counts["confirmed"] / (counts["confirmed"] + counts["falsified"]) * 100, 1)
        if (counts["confirmed"] + counts["falsified"]) > 0 else 0.0
    )
    return {
        "total": resolved,
        "confirmed": counts["confirmed"],
        "falsified": counts["falsified"],
        "neutral": counts["neutral"],
        "pending": counts["pending"],
        "accuracy_pct": accuracy_pct,
    }
```

- [ ] **Step 4: Run tracker tests**

```bash
.venv/bin/python -m pytest tests/test_causal_tracker.py -v 2>&1 | tail -12
```

Expected: 7 PASSes

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 708+ passed, 1 skipped, 0 failures

- [ ] **Step 6: Commit**

```bash
git add ascent/causal/tracker.py tests/test_causal_tracker.py
git commit -m "feat: add causal tracker — predictions log, early exit, outcome classification"
```

---

## Task 9: `run_all_agents.py` — Gate 4 early exit + Phase 2 track record

**Files:**
- Modify: `run_all_agents.py`

### Background

Two additions to the daily runner:

1. **Non-rebalance daily path**: after agents run, call `check_early_exits()`. For each returned symbol, log `causal_mechanism_broken` in `ai_pm_shadow_returns.jsonl` and zero out the AI PM shadow weight for that symbol.

2. **Phase 2 (rebalance path)**: before calling `run_ai_pm()`, call `get_track_record()` and pass the result as `causal_track_record` kwarg. Phase 2 synthesis prompt then references this.

The `run_ai_pm()` function needs a new optional `causal_track_record` parameter.

- [ ] **Step 1: Add `causal_track_record` param to `run_ai_pm()`**

In `agents/ai_pm_agent.py`, find `run_ai_pm()` signature (line ~1474):

```python
def run_ai_pm(
    quant_outputs: Optional[List] = None,
    prethesis: Optional[AIPreThesis] = None,
    causal_track_record: Optional[dict] = None,   # ← add this
) -> AIPMResult:
```

Find where the Phase 2 system prompt is built (the large string starting `"You are the portfolio manager..."`) and append causal track record context after the existing calibration track record section:

```python
    # Inject causal track record into Phase 2 prompt
    _causal_track_context = ""
    if causal_track_record and causal_track_record.get("total", 0) >= 3:
        acc = causal_track_record.get("accuracy_pct", 0)
        total = causal_track_record.get("total", 0)
        _causal_track_context = (
            f"\n\n══ CAUSAL THESIS TRACK RECORD ══\n"
            f"Your past causal mechanisms: {total} resolved, "
            f"{causal_track_record.get('confirmed', 0)} confirmed, "
            f"{causal_track_record.get('falsified', 0)} falsified, "
            f"accuracy={acc:.1f}%.\n"
            f"{'High accuracy — trust your causal mechanisms.' if acc >= 60 else 'Below-target accuracy — only concentrate when mechanism velocity > 0.70 AND timing=catalyst_imminent.'}"
        )
```

Append `_causal_track_context` to the user_prompt in `run_ai_pm()`.

- [ ] **Step 2: Add Gate 4 early exit to non-rebalance path in `run_all_agents.py`**

Find the non-rebalance branch in `run_all_agents.py`. After `run_ai_pm_agent()` or equivalent forward PnL section, add:

```python
    # Gate 4 — causal early exit check
    try:
        from ascent.causal.tracker import check_early_exits
        early_exit_symbols = check_early_exits()
        if early_exit_symbols:
            log.info("[Causal] Early exit flagged for: %s", early_exit_symbols)
            # Zero out AI PM shadow weight for broken mechanisms
            import json as _json
            from pathlib import Path as _Path
            shadow_path = _Path("data_cache/ai_pm_shadow_returns.jsonl")
            with open(shadow_path, "a") as f:
                for sym in early_exit_symbols:
                    f.write(_json.dumps({
                        "date": str(date.today()),
                        "symbol": sym,
                        "ai_pm_shadow_weight": 0.0,
                        "reason": "causal_mechanism_broken",
                    }) + "\n")
    except Exception as exc:
        log.warning("[Causal] Gate 4 early exit check failed: %s", exc)
```

- [ ] **Step 3: Pass `causal_track_record` on rebalance day**

Find where `run_ai_pm()` is called in the rebalance branch of `run_all_agents.py`. Add:

```python
    # Load causal track record for Phase 2 context
    _causal_track_record = None
    try:
        from ascent.causal.tracker import get_track_record
        _causal_track_record = get_track_record()
    except Exception:
        pass

    ai_pm_result = run_ai_pm(
        quant_outputs=agent_outputs,
        prethesis=prethesis,
        causal_track_record=_causal_track_record,   # ← pass it
    )
```

- [ ] **Step 4: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: same pass count, 0 failures

- [ ] **Step 5: Commit**

```bash
git add run_all_agents.py agents/ai_pm_agent.py
git commit -m "feat: Gate 4 early exit + causal track record in Phase 2"
```

---

## PHASE D — Debate Integration

---

## Task 10: Devil's advocate causal context injection

**Files:**
- Modify: `debate/agents.py`
- Modify: `tests/test_debate_agents.py`

### Background

The devil's advocate is the natural home for causal mechanism attacks: "Samsung announced NAND capacity additions — your NAND recovery mechanism is broken before it runs." The debate runner calls `run_devils_advocate()` with the existing `portfolio_context` dict. We add `causal_mechanisms` to that context and append a formatted summary to the devil's advocate system prompt.

No new LLM call — prompt append only.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_debate_agents.py`:

```python
def test_devils_advocate_system_prompt_contains_causal_attack_instruction():
    """Devil's advocate source must reference causal mechanisms attack capability."""
    import inspect
    import debate.agents as mod
    src = inspect.getsource(mod.run_devils_advocate)
    assert "causal" in src.lower() or "mechanism" in src.lower(), \
        "Devil's advocate must reference causal mechanism attack in its prompt/code"
```

- [ ] **Step 2: Run failing test**

```bash
.venv/bin/python -m pytest tests/test_debate_agents.py::test_devils_advocate_system_prompt_contains_causal_attack_instruction -v 2>&1 | tail -8
```

Expected: 1 FAIL

- [ ] **Step 3: Find `run_devils_advocate` and add causal context**

In `debate/agents.py`, find `run_devils_advocate` (the function that builds `_da_system_prompt`). Add causal_mechanisms formatting before the system prompt is built:

Find the part that builds `_da_system_prompt` and add after the existing track_record text:

```python
    # Format causal mechanisms for devil's advocate context
    _causal_context = ""
    causal_mechanisms = portfolio_context.get("causal_mechanisms", [])
    if causal_mechanisms:
        lines = ["══ AI PM CAUSAL MECHANISMS (attack these if the thesis is broken) ══"]
        for m in causal_mechanisms:
            sym = m.get("symbol", "?") if isinstance(m, dict) else getattr(m, "symbol", "?")
            mech = m.get("mechanism", "") if isinstance(m, dict) else getattr(m, "mechanism", "")
            falsif = m.get("falsification_condition", "") if isinstance(m, dict) else getattr(m, "falsification_condition", "")
            timing = m.get("timing", "") if isinstance(m, dict) else getattr(m, "timing", "")
            lines.append(f"  {sym} [{timing}]: {mech}")
            if falsif:
                lines.append(f"    Falsification: {falsif}")
        _causal_context = "\n" + "\n".join(lines)
```

Then append `_causal_context` to `_da_system_prompt`:

```python
    _da_system_prompt = (
        "You are the Devil's Advocate at Ascent Capital. Your job is to "
        "find the SINGLE most dangerous assumption in the current portfolio construction. "
        "What could go catastrophically wrong that the quant signals would NOT catch? "
        "You have been given Monte Carlo scenario analysis showing worst-case portfolio "
        "impacts. Use these numbers to make a specific, quantified argument. "
        "You also have historical accuracy data — use it to understand when your "
        "past warnings were prescient vs. over-cautious. "
        "Use the available tools to look up sector concentration, VaR, and momentum data "
        "to make quantitative arguments. "
        "CAUSAL MECHANISM ATTACK: If the AI PM's causal mechanisms are listed above, "
        "look for evidence that the causal mechanism has already failed or will fail — "
        "supply additions that break a supply-shortage thesis, earnings misses that "
        "break a margin-recovery thesis, or regime shifts that invalidate the mechanism type. "
        "Think about: earnings surprises, geopolitical events, liquidity gaps, "
        f"correlation breakdowns. Be specific. Keep under 150 words."
        f"{track_record}"
        f"{_EVIDENCE_RULE}"
        f"{_causal_context}"
    )
```

- [ ] **Step 4: Run the debate tests**

```bash
.venv/bin/python -m pytest tests/test_debate_agents.py -v 2>&1 | tail -12
```

Expected: all PASSes including the new one

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add debate/agents.py tests/test_debate_agents.py
git commit -m "feat: inject causal mechanism attack context into devil's advocate"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| `CausalMechanism` dataclass | Task 1 |
| `AIPreThesis.causal_mechanisms` field | Task 6 |
| `ascent/causal/velocity.py` — `mechanism_velocity_score()` | Task 2 |
| `ascent/causal/causal_discovery.py` — PC algorithm + `run_discovery()` | Task 3 |
| `ascent/causal/dag_builder.py` — Haiku per-symbol cache | Task 4 |
| `ascent/causal/compatibility.py` — Gate 1 static dict | Task 5 |
| Weekend runner: causal_macro_dag + causal_graph_builder jobs | Task 7 |
| Gate 1 in Phase 1 pre-thesis | Task 6 |
| Gate 2 (priced_in filter) in Phase 1 | Task 6 |
| `get_causal_graph` tool in PRE_THESIS_TOOLS | Task 6 |
| Velocity-ranked context injected into Phase 1 | Task 6 |
| `ascent/causal/tracker.py` + `causal_predictions.jsonl` | Task 8 |
| `check_early_exits()` daily non-rebalance | Task 9 |
| `check_outcomes()` weekly | Task 8 |
| Phase 2 `causal_track_record` injection | Task 9 |
| Gate 4 early exit → zero shadow weight + log | Task 9 |
| Devil's advocate causal mechanism attack context | Task 10 |
| `data_cache/causal_graphs/` directory | Tasks 4, 7 |
| `logs/causal_predictions.jsonl` | Task 8 |
| `data_cache/macro_causal_dag.json` | Task 3 |
| All 675 existing tests green | Every task (step 5/run full suite) |

### Placeholder scan

No TBD, TODO, or "implement later" entries. All code blocks are complete and runnable.

### Type consistency

- `CausalMechanism` defined in Task 1 (`types.py`) with field `mechanism_type`. All tasks that create `CausalMechanism` objects include `mechanism_type`. ✓
- `AIPreThesis.causal_mechanisms: List["CausalMechanism"]` — forward ref string avoids circular import since `CausalMechanism` is in `types.py` and `AIPreThesis` is in `agents/ai_pm_agent.py`. ✓
- `write_predictions()` takes `list` of `CausalMechanism` objects and reads `.symbol`, `.mechanism`, etc. These fields are defined in Task 1. ✓
- `_assemble_causal_mechanisms()` returns `List[CausalMechanism]` — matches the `AIPreThesis.causal_mechanisms` type. ✓
- `get_track_record()` returns `dict` with keys `total, confirmed, falsified, neutral, pending, accuracy_pct` — matches the keys referenced in `run_ai_pm()` Task 9 injection. ✓
- `_get_price_return()` is a module-level function used in both `check_early_exits()` and `check_outcomes()` — patched correctly in tests. ✓
- `build_graph()` and `load_or_build()` both live in `dag_builder.py` and use `cache_dir` parameter consistently. ✓
- `run_pc()` returns dict with `nodes, edges, active_transmission_chains` — used by `discover_macro_dag()` which adds `as_of, regime`. Test in Task 3 validates both. ✓
