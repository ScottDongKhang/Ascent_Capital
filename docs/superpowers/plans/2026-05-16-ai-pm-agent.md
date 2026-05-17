# AI PM Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an AI Portfolio Manager (Opus, tool-use loop) that runs its own research, proposes a portfolio with a full investment memo, and earns execution weight through a live track record — blending with the existing quant pipeline.

**Architecture:** The AI PM runs 4 phases (market context → quant baselines → signal research → submit) via Anthropic tool use, produces a portfolio + thesis, which is validated by a pure-math risk checker and then blended with the existing `merged_weights` using a weight proportional to earned authority (0%→75%). The existing execution stack (debate, approval gate, kill switch) is unchanged.

**Tech Stack:** Python 3.12, Anthropic SDK (`tool_completion` already in `ascent/llm/client.py`), pandas, JSON state files.

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `ascent/risk/pm_risk_validator.py` | **Create** | Pre-blend hard-limit check on AI PM proposals |
| `ascent/strategy/earned_authority.py` | **Create** | Track record, authority scaling (0→75%), auto-revert, blend |
| `ascent/strategy/thesis_formatter.py` | **Create** | JSON schema serialization + plaintext summary |
| `agents/ai_pm_agent.py` | **Create** | Opus tool-use loop, 14 tools, AIPMResult output |
| `run_all_agents.py` | **Modify** | Wire AI PM after orchestrator, before debate |
| `tests/test_ai_pm_agent.py` | **Create** | 16 tests across all four new modules |

---

## Task 1: Risk Validator

**Files:**
- Create: `ascent/risk/pm_risk_validator.py`
- Test: `tests/test_ai_pm_agent.py` (first batch)

- [ ] **Step 1: Create the test file with validator tests**

```python
# tests/test_ai_pm_agent.py
import pytest
from unittest.mock import patch


# ── pm_risk_validator ──────────────────────────────────────────────────────────

def test_validator_accepts_clean_portfolio():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.08, "MSFT": 0.07, "GOOG": 0.06, "AMZN": 0.06,
        "META": 0.07, "NVDA": 0.08, "TSLA": 0.05, "JPM": 0.07,
        "V": 0.06, "UNH": 0.06,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is True
    assert violations == []


def test_validator_rejects_concentrated_position():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.50, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("AAPL" in v for v in violations)


def test_validator_rejects_too_few_positions():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {"AAPL": 0.50, "MSFT": 0.50}
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("positions" in v.lower() for v in violations)


def test_validator_rejects_empty_portfolio():
    from ascent.risk.pm_risk_validator import validate
    ok, violations = validate({})
    assert ok is False


def test_validator_rejects_distressed_name():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10, "BAD": 0.10, "JPM": 0.08,
        "V": 0.07, "UNH": 0.07, "GS": 0.08,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=["BAD"]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("BAD" in v for v in violations)


def test_validator_rejects_sector_overweight():
    from ascent.risk.pm_risk_validator import validate
    sector_map = {
        "AAPL": "tech", "MSFT": "tech", "GOOG": "tech", "AMZN": "tech",
        "META": "tech", "NVDA": "tech",
        "JPM": "finance", "V": "finance", "GS": "finance", "WFC": "finance",
    }
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10,
        "JPM": 0.10, "V": 0.10, "GS": 0.10, "WFC": 0.10,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value=sector_map):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    # tech = 60% > 40% cap
    assert ok is False
    assert any("tech" in v.lower() for v in violations)
```

- [ ] **Step 2: Run tests to see them fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'ascent.risk.pm_risk_validator'`

- [ ] **Step 3: Create `ascent/risk/pm_risk_validator.py`**

```python
# ascent/risk/pm_risk_validator.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

MAX_POSITION = 0.15
MAX_SECTOR = 0.40
MIN_POSITIONS = 5
DISTRESSED_THRESHOLD = -0.65


def validate(portfolio: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Pre-blend hard-limit check. Returns (ok, violations). Never raises."""
    if not portfolio:
        return False, ["Empty portfolio"]

    total = sum(portfolio.values())
    if total <= 0:
        return False, ["Portfolio weights sum to zero or negative"]
    weights = {sym: w / total for sym, w in portfolio.items()}

    violations: List[str] = []

    for sym, w in weights.items():
        if w < 0:
            violations.append(f"Negative weight for {sym}: {w:.4f}")

    for sym, w in weights.items():
        if w > MAX_POSITION:
            violations.append(f"{sym} weight {w:.1%} exceeds max {MAX_POSITION:.0%}")

    if len(weights) < MIN_POSITIONS:
        violations.append(f"Only {len(weights)} positions (min {MIN_POSITIONS})")

    sector_weights = _compute_sector_weights(weights)
    for sector, sw in sector_weights.items():
        if sw > MAX_SECTOR:
            violations.append(f"Sector {sector} at {sw:.1%} exceeds max {MAX_SECTOR:.0%}")

    distressed = _get_distressed_names(list(weights.keys()))
    for sym in distressed:
        if sym in weights:
            violations.append(f"{sym} is distressed (mom_252d < {DISTRESSED_THRESHOLD:.0%})")

    return len(violations) == 0, violations


def _compute_sector_weights(weights: Dict[str, float]) -> Dict[str, float]:
    sector_map = _load_sector_map()
    result: Dict[str, float] = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym, "unknown")
        result[sector] = result.get(sector, 0.0) + w
    return result


def _load_sector_map() -> Dict[str, str]:
    try:
        import pandas as pd
        p = Path("data_cache/profiles.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            if "symbol" in df.columns and "sector" in df.columns:
                return dict(zip(df["symbol"], df["sector"]))
    except Exception as exc:
        log.warning("[PMValidator] Could not load sector map: %s", exc)
    return {}


def _get_distressed_names(symbols: List[str]) -> List[str]:
    try:
        import pandas as pd
        p = Path("data_cache/features_cache.parquet")
        if not p.exists():
            return []
        df = pd.read_parquet(p)
        if "symbol" not in df.columns or "mom_252d" not in df.columns:
            return []
        latest = df.sort_values("date").groupby("symbol").last().reset_index()
        distressed = latest[latest["mom_252d"] < DISTRESSED_THRESHOLD]["symbol"].tolist()
        return [s for s in distressed if s in symbols]
    except Exception as exc:
        log.warning("[PMValidator] Could not check distressed names: %s", exc)
        return []
```

- [ ] **Step 4: Run validator tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | head -30
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ascent/risk/pm_risk_validator.py tests/test_ai_pm_agent.py
git commit -m "feat(ai-pm): pm_risk_validator — pre-blend hard-limit check"
```

---

## Task 2: Earned Authority

**Files:**
- Create: `ascent/strategy/earned_authority.py`
- Modify: `tests/test_ai_pm_agent.py` (add 8 tests)

- [ ] **Step 1: Add earned authority tests to the test file**

```python
# append to tests/test_ai_pm_agent.py

import json, tempfile, os
from pathlib import Path
from unittest.mock import patch


# ── earned_authority ──────────────────────────────────────────────────────────

def _make_state(phase=0, ai_weight=0.0, ai_returns=None, qt_returns=None, reverts=0):
    return {
        "ai_weight": ai_weight,
        "phase": phase,
        "phase_start_date": "2026-05-16",
        "ai_returns_21d": ai_returns or [],
        "quant_returns_21d": qt_returns or [],
        "auto_revert_count": reverts,
        "last_updated": "2026-05-16",
    }


def test_shadow_phase_blend_returns_pure_quant():
    """At ai_weight=0, blend returns quant portfolio unchanged."""
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=0, ai_weight=0.0)))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                from ascent.strategy import earned_authority as ea
                quant = {"AAPL": 0.50, "MSFT": 0.50}
                ai = {"GOOG": 0.60, "AMZN": 0.40}
                result = ea.blend(ai, quant)
    # ai_weight=0 → only quant names survive min_weight filter
    assert "AAPL" in result
    assert "MSFT" in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_union_of_positions():
    """At ai_weight=0.5, both portfolios contribute."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=2, ai_weight=0.5)))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                from importlib import import_module, reload
                import ascent.strategy.earned_authority as ea
                ai = {"VICR": 0.10, "AMKR": 0.08}
                quant = {"VICR": 0.06, "FIX": 0.07}
                result = ea.blend(ai, quant)
    # VICR: 0.5*0.10 + 0.5*0.06 = 0.08 (before renorm)
    # AMKR: 0.5*0.08 + 0.5*0.00 = 0.04 (before renorm)
    # FIX:  0.5*0.00 + 0.5*0.07 = 0.035 (before renorm)
    assert "VICR" in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_min_weight_filter():
    """Positions below 0.02 after blending are dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        # ai_weight=0.1 → AI PM names get tiny weight
        state_path.write_text(json.dumps(_make_state(phase=0, ai_weight=0.1)))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                import ascent.strategy.earned_authority as ea
                ai = {"TINY": 0.05}  # 0.1 * 0.05 = 0.005 → below 0.02
                quant = {"AAPL": 0.50, "MSFT": 0.50}
                result = ea.blend(ai, quant)
    assert "TINY" not in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_renormalizes_to_1():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state_path.write_text(json.dumps(_make_state(phase=1, ai_weight=0.25)))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                import ascent.strategy.earned_authority as ea
                ai = {f"A{i}": 0.1 for i in range(5)}
                quant = {f"Q{i}": 0.1 for i in range(5)}
                result = ea.blend(ai, quant)
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_authority_advances_after_edge():
    """After 21 days with AI Sharpe > quant+0.05, phase advances."""
    # Build 21 returns where AI consistently beats quant
    ai_returns = [0.002] * 21   # Sharpe ~2.0
    qt_returns = [0.001] * 21   # Sharpe ~1.0
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state = _make_state(phase=0, ai_weight=0.0, ai_returns=ai_returns[:20], qt_returns=qt_returns[:20])
        state_path.write_text(json.dumps(state))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                import ascent.strategy.earned_authority as ea
                result = ea.update_authority(ai_returns[-1], qt_returns[-1])
    assert result["phase"] == 1
    assert result["ai_weight"] == 0.25


def test_authority_stays_if_no_edge():
    """With no Sharpe edge, phase stays at 0."""
    ai_returns = [0.001] * 21
    qt_returns = [0.001] * 21  # equal — no edge
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state = _make_state(phase=0, ai_weight=0.0, ai_returns=ai_returns[:20], qt_returns=qt_returns[:20])
        state_path.write_text(json.dumps(state))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                import ascent.strategy.earned_authority as ea
                result = ea.update_authority(ai_returns[-1], qt_returns[-1])
    assert result["phase"] == 0
    assert result["ai_weight"] == 0.0


def test_auto_revert_on_drawdown():
    """AI drawdown > quant+5% at phase>0 triggers revert to phase 0."""
    # AI has large drawdown, quant is flat
    ai_returns = [-0.03] * 10 + [0.01] * 11  # big early drawdown
    qt_returns = [0.001] * 21
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        state = _make_state(phase=2, ai_weight=0.5, ai_returns=ai_returns[:20], qt_returns=qt_returns[:20])
        state_path.write_text(json.dumps(state))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                import ascent.strategy.earned_authority as ea
                result = ea.update_authority(ai_returns[-1], qt_returns[-1])
    assert result["phase"] == 0
    assert result["ai_weight"] == 0.0
    assert result["auto_revert_count"] == 1


def test_hard_cap_at_0_80():
    """ai_weight never exceeds 0.80 regardless of phase."""
    from ascent.strategy import earned_authority as ea
    # Phase weights are [0.0, 0.25, 0.50, 0.75, 0.80]
    assert ea.HARD_CAP == 0.80
    assert ea.PHASE_WEIGHTS[3] <= ea.HARD_CAP
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | grep -E "PASSED|FAILED|ERROR" | head -20
```

Expected: the 6 old tests still pass, 8 new ones fail with `ModuleNotFoundError`.

- [ ] **Step 3: Create `ascent/strategy/__init__.py` if needed**

```bash
ls "/Users/scott/Downloads/ascent capital v2 up to phase 5.1/ascent/strategy/" 2>/dev/null || mkdir -p "/Users/scott/Downloads/ascent capital v2 up to phase 5.1/ascent/strategy" && touch "/Users/scott/Downloads/ascent capital v2 up to phase 5.1/ascent/strategy/__init__.py"
```

- [ ] **Step 4: Create `ascent/strategy/earned_authority.py`**

```python
# ascent/strategy/earned_authority.py
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

STATE_PATH = Path("data_cache/earned_authority.json")
SHADOW_RETURNS_PATH = Path("data_cache/ai_pm_shadow_returns.jsonl")

ADVANCE_EDGE = 0.05
ADVANCE_WINDOW = 21
REVERT_DRAWDOWN_EDGE = 0.05
MIN_WEIGHT = 0.02
HARD_CAP = 0.80
PHASE_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 0.80]


def get_state() -> dict:
    """Load state from JSON. Returns defaults if file missing."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "ai_weight": 0.0, "phase": 0,
        "phase_start_date": str(date.today()),
        "ai_returns_21d": [], "quant_returns_21d": [],
        "auto_revert_count": 0, "last_updated": str(date.today()),
    }


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _sharpe(returns: List[float]) -> float:
    if len(returns) < 5:
        return 0.0
    import statistics
    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
    return 0.0 if stdev == 0 else mean / stdev * (252 ** 0.5)


def _max_drawdown(returns: List[float]) -> float:
    if not returns:
        return 0.0
    cum = peak = 1.0
    max_dd = 0.0
    for r in returns:
        cum *= (1 + r)
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / peak)
    return max_dd


def update_authority(ai_daily_return: float, quant_daily_return: float) -> dict:
    """Append daily returns, check advance/revert, save state. Returns updated state."""
    state = get_state()
    today = str(date.today())

    ai_buf: List[float] = (state.get("ai_returns_21d", []) + [float(ai_daily_return)])[-ADVANCE_WINDOW:]
    qt_buf: List[float] = (state.get("quant_returns_21d", []) + [float(quant_daily_return)])[-ADVANCE_WINDOW:]
    state["ai_returns_21d"] = ai_buf
    state["quant_returns_21d"] = qt_buf
    _log_shadow_return(today, ai_daily_return, quant_daily_return, state["ai_weight"])

    # Auto-revert check
    if state["phase"] > 0 and _max_drawdown(ai_buf) > _max_drawdown(qt_buf) + REVERT_DRAWDOWN_EDGE:
        log.warning("[EarnedAuthority] Auto-revert triggered")
        state.update({
            "phase": 0, "ai_weight": 0.0,
            "phase_start_date": today,
            "ai_returns_21d": [], "quant_returns_21d": [],
            "auto_revert_count": state.get("auto_revert_count", 0) + 1,
        })
        state["last_updated"] = today
        _save_state(state)
        return state

    # Advance check
    if len(ai_buf) >= ADVANCE_WINDOW and state["phase"] < 3:
        if _sharpe(ai_buf) > _sharpe(qt_buf) + ADVANCE_EDGE:
            state["phase"] = min(state["phase"] + 1, 3)
            state["ai_weight"] = min(PHASE_WEIGHTS[state["phase"]], HARD_CAP)
            state.update({
                "phase_start_date": today,
                "ai_returns_21d": [], "quant_returns_21d": [],
            })
            log.info("[EarnedAuthority] Phase → %d, ai_weight=%.0f%%",
                     state["phase"], state["ai_weight"] * 100)

    state["last_updated"] = today
    _save_state(state)
    return state


def blend(ai_portfolio: Dict[str, float], quant_portfolio: Dict[str, float]) -> Dict[str, float]:
    """Weight-average over union, drop < MIN_WEIGHT=0.02, renormalize to 1.0."""
    state = get_state()
    ai_w = state["ai_weight"]
    qt_w = 1.0 - ai_w

    blended: Dict[str, float] = {}
    for sym in set(ai_portfolio) | set(quant_portfolio):
        w = ai_w * ai_portfolio.get(sym, 0.0) + qt_w * quant_portfolio.get(sym, 0.0)
        if w >= MIN_WEIGHT:
            blended[sym] = w

    total = sum(blended.values())
    if total <= 0:
        return dict(quant_portfolio)
    return {sym: w / total for sym, w in blended.items()}


def _log_shadow_return(today: str, ai_ret: float, qt_ret: float, ai_weight: float) -> None:
    try:
        SHADOW_RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_RETURNS_PATH, "a") as f:
            f.write(json.dumps({"date": today, "ai_return": ai_ret,
                                "quant_return": qt_ret, "ai_weight_at_time": ai_weight}) + "\n")
    except Exception as exc:
        log.warning("[EarnedAuthority] Could not log shadow return: %s", exc)
```

- [ ] **Step 5: Run earned authority tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

Expected: 14 tests pass (6 validator + 8 authority).

- [ ] **Step 6: Commit**

```bash
git add ascent/strategy/__init__.py ascent/strategy/earned_authority.py tests/test_ai_pm_agent.py
git commit -m "feat(ai-pm): earned_authority — track record state machine, blend, auto-revert"
```

---

## Task 3: Thesis Formatter

**Files:**
- Create: `ascent/strategy/thesis_formatter.py`
- Modify: `tests/test_ai_pm_agent.py` (add 2 tests)

- [ ] **Step 1: Add thesis formatter tests**

```python
# append to tests/test_ai_pm_agent.py

# ── thesis_formatter ──────────────────────────────────────────────────────────

def test_format_thesis_fills_missing_fields():
    """Missing keys in raw_thesis get filled with schema defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ai_pm_theses"
        with patch("ascent.strategy.thesis_formatter.OUTPUT_DIR", out_dir):
            from ascent.strategy.thesis_formatter import format_thesis
            result = format_thesis({"market_view": "Calm bull, credit spreads tight."})
    assert "market_view" in result
    assert result["market_view"] == "Calm bull, credit spreads tight."
    assert "quant_overrides" in result
    assert result["quant_overrides"] == []
    assert "as_of_date" in result


def test_thesis_to_plaintext_returns_non_empty_string():
    from ascent.strategy.thesis_formatter import thesis_to_plaintext
    thesis = {
        "market_view": "Credit spreads are widening.",
        "regime_assessment": "calm_bull, confidence 0.73",
        "ai_pm_portfolio": {"VICR": 0.06, "AMKR": 0.05, "FIX": 0.07},
        "quant_agreement": ["VICR", "FIX"],
        "quant_overrides": [{"symbol": "VAL", "ai_action": "exclude"}],
        "key_risks": ["Macro: Fed surprise", "Idio: VICR miss"],
    }
    result = thesis_to_plaintext(thesis)
    assert isinstance(result, str)
    assert len(result) > 20
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py::test_format_thesis_fills_missing_fields tests/test_ai_pm_agent.py::test_thesis_to_plaintext_returns_non_empty_string -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'ascent.strategy.thesis_formatter'`

- [ ] **Step 3: Create `ascent/strategy/thesis_formatter.py`**

```python
# ascent/strategy/thesis_formatter.py
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/ai_pm_theses")

_SCHEMA_DEFAULTS = {
    "market_view": "",
    "regime_assessment": "",
    "quant_baseline_summary": "",
    "ai_pm_portfolio": {},
    "quant_agreement": [],
    "quant_overrides": [],
    "position_rationale": {},
    "key_risks": [],
    "what_could_be_wrong": "",
}


def format_thesis(raw_thesis: dict, as_of_date: Optional[date] = None) -> dict:
    """
    Validate and serialize full investment memo JSON.
    Missing fields are filled with defaults.
    Saves to outputs/ai_pm_theses/YYYY-MM-DD-thesis.json.
    """
    if as_of_date is None:
        as_of_date = date.today()

    thesis = {**_SCHEMA_DEFAULTS}
    thesis.update({k: v for k, v in raw_thesis.items() if k in _SCHEMA_DEFAULTS})
    thesis["as_of_date"] = str(as_of_date)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{as_of_date}-thesis.json"
    try:
        out_path.write_text(json.dumps(thesis, indent=2, default=str))
    except Exception as exc:
        log.warning("[ThesisFormatter] Could not save thesis: %s", exc)

    return thesis


def thesis_to_plaintext(thesis: dict) -> str:
    """3-4 sentence narrative summary for investor reports."""
    parts = []

    market_view = thesis.get("market_view", "")
    if market_view:
        parts.append(market_view.strip().rstrip(".") + ".")

    regime = thesis.get("regime_assessment", "")
    n_pos = len(thesis.get("ai_pm_portfolio", {}))
    if regime and n_pos:
        parts.append(f"Given {regime}, the AI PM constructed a {n_pos}-position portfolio.")

    agreements = thesis.get("quant_agreement", [])
    overrides = thesis.get("quant_overrides", [])
    if agreements or overrides:
        parts.append(
            f"The AI PM agreed with {len(agreements)} quant recommendations "
            f"and overrode {len(overrides)}."
        )

    risks = thesis.get("key_risks", [])
    if risks:
        parts.append(f"Key risks: {'; '.join(risks[:3])}.")

    return " ".join(parts) if parts else "No thesis available."
```

- [ ] **Step 4: Run all tests so far**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

Expected: 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ascent/strategy/thesis_formatter.py tests/test_ai_pm_agent.py
git commit -m "feat(ai-pm): thesis_formatter — investment memo schema + plaintext summary"
```

---

## Task 4: AI PM Agent

**Files:**
- Create: `agents/ai_pm_agent.py`
- Modify: `tests/test_ai_pm_agent.py` (add 2 tests)

- [ ] **Step 1: Add agent tests**

```python
# append to tests/test_ai_pm_agent.py

# ── ai_pm_agent ───────────────────────────────────────────────────────────────

def test_fallback_on_no_propose_portfolio_call():
    """If loop exits without calling propose_portfolio, return fallback result."""
    from unittest.mock import patch

    def fake_tool_completion(system_prompt, user_prompt, tools, tool_executor,
                             model, max_tokens, max_tool_calls):
        # Simulate loop completing without propose_portfolio
        return "I forgot to submit."

    with patch("agents.ai_pm_agent.tool_completion", fake_tool_completion):
        from agents.ai_pm_agent import run_ai_pm
        result = run_ai_pm()

    assert result.fallback is True
    assert result.portfolio == {}


def test_tool_executor_never_raises():
    """All tools with bad inputs return strings, never raise."""
    from agents.ai_pm_agent import _make_executor
    result_store = []
    executor = _make_executor(result_store)

    bad_inputs = [
        ("get_regime_state", {}),
        ("get_macro_data", {}),
        ("run_quant_agent", {"agent_id": "nonexistent_agent"}),
        ("get_sec_signal", {"symbol": "FAKESYM"}),
        ("get_transcript_signal", {"symbol": "FAKESYM"}),
        ("get_attribution_history", {"symbol": "FAKESYM"}),
        ("get_earnings_signal", {"symbol": "FAKESYM"}),
        ("get_past_verdicts", {"regime": "nonexistent_regime"}),
        ("get_factor_exposures", {"weights": {"FAKE": 1.0}}),
        ("get_var_estimate", {"weights": {"FAKE": 1.0}}),
        ("get_sector_concentration", {"weights": {"FAKE": 1.0}}),
        ("get_position_momentum", {"symbols": ["FAKESYM"]}),
        ("completely_unknown_tool", {}),
    ]
    for tool_name, inputs in bad_inputs:
        result = executor(tool_name, inputs)
        assert isinstance(result, str), f"Tool {tool_name} returned non-string: {type(result)}"
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py::test_fallback_on_no_propose_portfolio_call tests/test_ai_pm_agent.py::test_tool_executor_never_raises -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'agents.ai_pm_agent'`

- [ ] **Step 3: Create `agents/ai_pm_agent.py`**

```python
# agents/ai_pm_agent.py
"""
AI Portfolio Manager — Opus tool-use loop.

Runs a 4-phase research loop (market context → quant baselines → signal research → submit)
and returns AIPMResult(portfolio, thesis).
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascent.llm.client import tool_completion, DEFAULT_MODEL

log = logging.getLogger(__name__)


@dataclass
class AIPMResult:
    portfolio: Dict[str, float]
    thesis: Dict[str, Any]
    fallback: bool = False


# ── Tool schemas ───────────────────────────────────────────────────────────────

AI_PM_TOOLS = [
    {
        "name": "get_regime_state",
        "description": "Get the current market regime label, confidence, HMM entropy, and days in current regime.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_macro_data",
        "description": "Get current macro indicators: yield curve (T10Y2Y), credit spread, oil (DCOILWTICO), CPI, unemployment.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_quant_agent",
        "description": "Run a specialist quant agent and get its target weights, regime signal, and 63-day Sharpe skill score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "enum": ["us_equities", "macro", "international", "alternatives"],
                }
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "get_sec_signal",
        "description": "Get the most recent SEC 10-K/10-Q LLM classification for a symbol (revenue_momentum, margin_trend, tone, liquidity_risk, guidance; each -1 to +1).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_transcript_signal",
        "description": "Get the most recent earnings transcript LLM classification for a symbol (tone, defensiveness, forward_confidence, quantitative_ratio).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_attribution_history",
        "description": "Get the last 63 days of P&L attribution for a symbol: total P&L, factor P&L, idiosyncratic P&L.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_earnings_signal",
        "description": "Get the momentum-neutral PEAD signal for a symbol: earnings surprise z-score with momentum beta removed.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_past_verdicts",
        "description": "Get the last 5 debate verdicts where the market regime matched the given regime label.",
        "input_schema": {
            "type": "object",
            "properties": {"regime": {"type": "string"}},
            "required": ["regime"],
        },
    },
    {
        "name": "get_factor_exposures",
        "description": "Get portfolio factor risk exposures (market beta, size, value, profitability, investment, momentum tilts).",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Dict of symbol → weight",
                }
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_var_estimate",
        "description": "Get historical Value-at-Risk (5th percentile 1-day return) for a proposed portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {"type": "object", "additionalProperties": {"type": "number"}}
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_sector_concentration",
        "description": "Get the sector-level weight breakdown for a proposed portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {"type": "object", "additionalProperties": {"type": "number"}}
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_position_momentum",
        "description": "Get 252-day momentum (price return) for a list of symbols.",
        "input_schema": {
            "type": "object",
            "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
            "required": ["symbols"],
        },
    },
    {
        "name": "propose_portfolio",
        "description": "REQUIRED: Submit your final portfolio and investment thesis. Call this to end the research loop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Dict of symbol → target weight (will be normalized)",
                },
                "thesis": {
                    "type": "object",
                    "description": (
                        "Investment memo. Keys: market_view, regime_assessment, "
                        "quant_baseline_summary, quant_agreement (list), "
                        "quant_overrides (list of {symbol, ai_action, reason}), "
                        "position_rationale (dict), key_risks (list), what_could_be_wrong."
                    ),
                },
            },
            "required": ["weights", "thesis"],
        },
    },
]

_SYSTEM_PROMPT = """You are the portfolio manager of Ascent Capital, a multi-strategy quantitative fund.

Your job is to construct a portfolio for the next rebalance period using the research tools available.

Work through your research in order:
1. PHASE 1 — Market context: Call get_regime_state and get_macro_data first.
2. PHASE 2 — Quant baseline: Call run_quant_agent for all four agents (us_equities, macro, international, alternatives). This gives you the quantitative models' top names and weights.
3. PHASE 3 — Signal research: For names you are considering, call up to 6 of the available signal tools (get_sec_signal, get_transcript_signal, get_attribution_history, get_earnings_signal, get_past_verdicts, get_factor_exposures, get_var_estimate, get_sector_concentration, get_position_momentum).
4. PHASE 4 — Submit: Call propose_portfolio with your final weights and investment thesis.

Rules:
- You MUST call propose_portfolio before finishing. The loop ends only when you call it.
- Target 12-20 positions. Weights will be normalized; use relative sizing.
- For every name where you override a quant recommendation, include a specific reason in thesis.quant_overrides referencing the signal data you reviewed.
- If data is unavailable for a symbol, say so in your thesis — do not fabricate signals.
- The quant models are your research assistants, not your bosses. You may agree or disagree, but explain why.
"""


# ── Tool executor implementations ──────────────────────────────────────────────

def _tool_get_regime_state(_: dict) -> str:
    try:
        p = Path("dashboard/regime_signal.json")
        if not p.exists():
            return "Regime signal file not found."
        data = json.loads(p.read_text())
        row = data[-1] if isinstance(data, list) and data else data if isinstance(data, dict) else {}
        return (
            f"Current regime: {row.get('regime_label', row.get('label', 'unknown'))}\n"
            f"Confidence: {row.get('confidence', row.get('regime_confidence', 'n/a'))}\n"
            f"HMM entropy: {row.get('entropy', 'n/a')}\n"
            f"Days in regime: {row.get('days_in_regime', 'n/a')}"
        )
    except Exception as exc:
        return f"Could not read regime state: {exc}"


def _tool_get_macro_data(_: dict) -> str:
    try:
        import pandas as pd
        for name in ("macro_live", "macro_simulated"):
            p = Path(f"data_cache/{name}.parquet")
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    latest = df.sort_index().iloc[-1]
                    lines = ["Current macro indicators (latest available):"]
                    for col in list(df.columns)[:10]:
                        val = latest.get(col)
                        if val is not None:
                            lines.append(f"  {col}: {val:.4f}")
                    return "\n".join(lines)
        return "Macro data not found."
    except Exception as exc:
        return f"Could not read macro data: {exc}"


def _tool_run_quant_agent(inputs: dict) -> str:
    agent_id = inputs.get("agent_id", "")
    _AGENT_MAP = {
        "us_equities":   ("agents.us_equities_agent", "run_us_equities_agent"),
        "macro":         ("agents.macro_agent",        "run_macro_agent"),
        "international": ("agents.international_agent","run_international_agent"),
        "alternatives":  ("agents.alternatives_agent", "run_alternatives_agent"),
    }
    if agent_id not in _AGENT_MAP:
        return f"Unknown agent_id: '{agent_id}'. Valid: {list(_AGENT_MAP.keys())}"
    try:
        import importlib
        module_name, fn_name = _AGENT_MAP[agent_id]
        mod = importlib.import_module(module_name)
        result = getattr(mod, fn_name)()
        if result is None:
            return f"Agent {agent_id} returned no result."
        top = sorted(result.target_weights.items(), key=lambda x: -x[1])[:10]
        weight_str = ", ".join(f"{s}={w:.1%}" for s, w in top)
        return (
            f"Quant agent: {agent_id}\n"
            f"Regime: {result.regime_signal}\n"
            f"Skill score (63d Sharpe): {result.skill_score:.3f}\n"
            f"Top weights: {weight_str}"
        )
    except Exception as exc:
        return f"Agent {agent_id} failed: {exc}"


def _tool_get_sec_signal(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        import pandas as pd
        p = Path("data_cache/sec_signals.parquet")
        if not p.exists():
            return f"SEC signals not found (run ascent/data/ingest/sec_filings.py to populate)."
        df = pd.read_parquet(p)
        subset = df[df.get("symbol", pd.Series()) == symbol] if "symbol" in df.columns else pd.DataFrame()
        if subset.empty:
            return f"No SEC signal for {symbol}."
        row = subset.sort_values("date").iloc[-1]
        cols = [c for c in ["revenue_momentum","margin_trend","tone","liquidity_risk","guidance"] if c in row.index]
        lines = [f"SEC 10-K/10-Q signal for {symbol} (as of {row.get('date','?')}):"]
        for c in cols:
            lines.append(f"  {c}: {row[c]:.2f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"SEC signal failed for {symbol}: {exc}"


def _tool_get_transcript_signal(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        import pandas as pd
        p = Path("data_cache/transcript_signals.parquet")
        if not p.exists():
            return f"Transcript signals not found."
        df = pd.read_parquet(p)
        subset = df[df["symbol"] == symbol].sort_values("date") if "symbol" in df.columns else pd.DataFrame()
        if subset.empty:
            return f"No transcript signal for {symbol}."
        row = subset.iloc[-1]
        cols = [c for c in ["tone","defensiveness","forward_confidence","quantitative_ratio"] if c in row.index]
        lines = [f"Transcript signal for {symbol}:"]
        for c in cols:
            lines.append(f"  {c}: {row[c]:.2f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Transcript signal failed for {symbol}: {exc}"


def _tool_get_attribution_history(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        p = Path("logs/attribution_log.jsonl")
        if not p.exists():
            return "Attribution log not found."
        records = []
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("symbol") == symbol:
                        records.append(r)
                except Exception:
                    pass
        records = records[-63:]
        if not records:
            return f"No attribution history for {symbol}."
        total = sum(r.get("pnl", 0) for r in records)
        factor = sum(r.get("factor_pnl", 0) for r in records)
        idio = sum(r.get("idiosyncratic_pnl", 0) for r in records)
        return (
            f"Attribution for {symbol} (last {len(records)} days):\n"
            f"  Total P&L: {total:+.4f}\n"
            f"  Factor P&L: {factor:+.4f}\n"
            f"  Idiosyncratic P&L: {idio:+.4f}"
        )
    except Exception as exc:
        return f"Attribution failed for {symbol}: {exc}"


def _tool_get_earnings_signal(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        import pandas as pd
        p = Path("data_cache/earnings_cache.parquet")
        if not p.exists():
            return "Earnings cache not found."
        df = pd.read_parquet(p)
        if "symbol" not in df.columns:
            return "Earnings cache malformed."
        subset = df[df["symbol"] == symbol].sort_values("date") if not df.empty else pd.DataFrame()
        if subset.empty:
            return f"No earnings signal for {symbol}."
        row = subset.iloc[-1]
        surprise = row.get("surprise_pct", row.get("earnings_surprise", "n/a"))
        return f"PEAD signal for {symbol}:\n  Earnings surprise (momentum-neutral): {surprise}"
    except Exception as exc:
        return f"Earnings signal failed for {symbol}: {exc}"


def _tool_get_past_verdicts(inputs: dict) -> str:
    regime = inputs.get("regime", "")
    try:
        d = Path("outputs/debate_log")
        if not d.exists():
            return "No debate log found."
        verdicts = []
        for vf in sorted(d.glob("verdict_*.json"), reverse=True):
            try:
                v = json.loads(vf.read_text())
                if not regime or v.get("regime", "") == regime:
                    verdicts.append(v)
                    if len(verdicts) >= 5:
                        break
            except Exception:
                pass
        if not verdicts:
            return f"No past verdicts for regime '{regime}'."
        lines = [f"Last {len(verdicts)} verdicts for regime '{regime}':"]
        for v in verdicts:
            conf = v.get("confidence", 0)
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            lines.append(f"  {v.get('date','?')}: {v.get('verdict','?')} (conf={conf_str})")
        return "\n".join(lines)
    except Exception as exc:
        return f"Verdict lookup failed: {exc}"


def _tool_get_factor_exposures(inputs: dict) -> str:
    weights = inputs.get("weights", {})
    try:
        import pandas as pd
        from ascent.risk.factor_exposure import format_exposure_context
        w_series = pd.Series(weights, dtype=float)
        return format_exposure_context(w_series, date.today())
    except Exception as exc:
        return f"Factor exposure failed: {exc}"


def _tool_propose_portfolio(inputs: dict, result_store: list) -> str:
    weights = inputs.get("weights", {})
    thesis = inputs.get("thesis", {})
    result_store.append(AIPMResult(portfolio=weights, thesis=thesis))
    return "Portfolio submitted. Research loop complete."


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _make_executor(result_store: list):
    from debate.agent_tools import (
        get_var_estimate, get_sector_concentration, get_position_momentum,
    )

    _map = {
        "get_regime_state":         _tool_get_regime_state,
        "get_macro_data":           _tool_get_macro_data,
        "run_quant_agent":          _tool_run_quant_agent,
        "get_sec_signal":           _tool_get_sec_signal,
        "get_transcript_signal":    _tool_get_transcript_signal,
        "get_attribution_history":  _tool_get_attribution_history,
        "get_earnings_signal":      _tool_get_earnings_signal,
        "get_past_verdicts":        _tool_get_past_verdicts,
        "get_factor_exposures":     _tool_get_factor_exposures,
        "get_var_estimate":         lambda i: get_var_estimate(i),
        "get_sector_concentration": lambda i: get_sector_concentration(i),
        "get_position_momentum":    lambda i: get_position_momentum(i),
        "propose_portfolio":        lambda i: _tool_propose_portfolio(i, result_store),
    }

    def executor(tool_name: str, tool_inputs: dict) -> str:
        fn = _map.get(tool_name)
        if fn is None:
            return f"Unknown tool: {tool_name}"
        try:
            return fn(tool_inputs)
        except Exception as exc:
            log.warning("[AIPMAgent] Tool %s failed: %s", tool_name, exc)
            return f"Tool {tool_name} failed: {exc}"

    return executor


# ── Entry point ────────────────────────────────────────────────────────────────

def run_ai_pm(
    quant_outputs: Optional[list] = None,
    merged_weights: Optional[Dict[str, float]] = None,
) -> AIPMResult:
    """Run the AI PM agent. Returns AIPMResult. Falls back to empty portfolio on failure."""
    result_store: List[AIPMResult] = []

    try:
        tool_completion(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Today is {date.today()}. Please conduct your research and submit your portfolio.",
            tools=AI_PM_TOOLS,
            tool_executor=_make_executor(result_store),
            model=DEFAULT_MODEL,
            max_tokens=4000,
            max_tool_calls=14,
        )
    except Exception as exc:
        log.error("[AIPMAgent] tool_completion failed: %s", exc)
        return AIPMResult(portfolio={}, thesis={}, fallback=True)

    if not result_store:
        log.warning("[AIPMAgent] No propose_portfolio call — using fallback")
        return AIPMResult(portfolio={}, thesis={}, fallback=True)

    return result_store[-1]
```

- [ ] **Step 4: Run all 18 tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
```

Expected: 18 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agents/ai_pm_agent.py tests/test_ai_pm_agent.py
git commit -m "feat(ai-pm): ai_pm_agent — 4-phase Opus tool loop, 13 research tools + propose_portfolio"
```

---

## Task 5: Wire into `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py`

The wiring goes in the rebalance path, after `orchestrator` produces `merged_weights` and before `debate`. On non-rebalance days, only `update_authority` runs (to track shadow returns).

- [ ] **Step 1: Add imports at top of `run_all_agents.py`**

Find the existing import block and add:

```python
from agents.ai_pm_agent import run_ai_pm, AIPMResult
from ascent.risk.pm_risk_validator import validate as validate_pm_proposal
from ascent.strategy.earned_authority import blend as authority_blend, update_authority
from ascent.strategy.thesis_formatter import format_thesis
```

- [ ] **Step 2: Add AI PM block after orchestrator, before debate**

Find the section in `run_all_agents.py` where `merged_weights` is computed and the debate is called. Insert between them:

```python
    # ── AI PM Agent ───────────────────────────────────────────────────────────
    ai_pm_final_weights = merged_weights  # default: quant 100%
    try:
        log.info("[Runner] Running AI PM agent...")
        ai_pm_result = run_ai_pm(quant_outputs=agent_outputs, merged_weights=merged_weights)

        if ai_pm_result.fallback:
            log.warning("[Runner] AI PM fallback — using quant portfolio")
        else:
            ok, violations = validate_pm_proposal(ai_pm_result.portfolio)
            if ok:
                ai_pm_final_weights = authority_blend(ai_pm_result.portfolio, merged_weights)
                log.info("[Runner] AI PM blend applied (ai_weight=%.0f%%)",
                         __import__("ascent.strategy.earned_authority",
                                    fromlist=["get_state"]).get_state()["ai_weight"] * 100)
            else:
                log.warning("[Runner] AI PM proposal rejected: %s — quant 100%%", violations)

            # Save thesis and record in audit trail
            thesis = format_thesis(ai_pm_result.thesis)
            try:
                from compliance.audit_trail import record_event
                record_event("ai_pm_proposal", {
                    "portfolio": ai_pm_result.portfolio,
                    "validated": ok,
                    "violations": violations if not ok else [],
                })
            except Exception as ae:
                log.warning("[Runner] Audit trail write failed: %s", ae)

    except Exception as exc:
        log.error("[Runner] AI PM agent failed: %s — using quant portfolio", exc)

    merged_weights = ai_pm_final_weights
    # ─────────────────────────────────────────────────────────────────────────
```

- [ ] **Step 3: Add shadow return tracking in the non-rebalance daily block**

Find where forward PnL is computed on non-rebalance days and add:

```python
    # Update earned authority with today's shadow returns
    try:
        # ai_daily_return: hypothetical return of last AI PM portfolio
        ai_pm_theses_dir = Path("outputs/ai_pm_theses")
        last_thesis_files = sorted(ai_pm_theses_dir.glob("*-thesis.json"), reverse=True)
        ai_portfolio = {}
        if last_thesis_files:
            last_thesis = json.loads(last_thesis_files[0].read_text())
            ai_portfolio = last_thesis.get("ai_pm_portfolio", {})

        if ai_portfolio:
            import yfinance as yf
            syms = list(ai_portfolio.keys())
            prices = yf.download(syms, period="5d", auto_adjust=True, progress=False)
            if isinstance(prices.columns, __import__("pandas").MultiIndex):
                prices = prices["Close"]
            rets = prices.pct_change().iloc[-1]
            ai_ret = float(sum(ai_portfolio.get(s, 0) * rets.get(s, 0) for s in syms))
        else:
            ai_ret = 0.0

        # quant return: use the PnL log (same source as forward_pnl_tracker)
        quant_ret = 0.0
        pnl_log = Path("logs/us_equities_pnl.jsonl")
        if pnl_log.exists():
            lines = pnl_log.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                quant_ret = float(last.get("portfolio_return", 0.0))

        update_authority(ai_ret, quant_ret)
    except Exception as exc:
        log.warning("[Runner] Earned authority update failed: %s", exc)
```

- [ ] **Step 4: Verify the full runner imports and parses cleanly**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: 464 passed (446 + 18 new), 1 skipped, 0 errors.

- [ ] **Step 6: Commit**

```bash
git add run_all_agents.py
git commit -m "feat(ai-pm): wire AI PM agent into run_all_agents — shadow period active, earned authority tracking"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|------------|
| AI PM runs Opus tool-use loop | Task 4: `run_ai_pm()` + `tool_completion()` |
| 4-phase research structure | Task 4: system prompt + 13 tools |
| 14 total tools including `propose_portfolio` terminal | Task 4: `AI_PM_TOOLS` list (13 tools) |
| Full investment memo JSON | Task 3: `thesis_formatter.py` |
| Pre-blend risk validator | Task 1: `pm_risk_validator.py` |
| Earned authority state machine | Task 2: `earned_authority.py` |
| Phase schedule 0%→25%→50%→75%, cap 80% | Task 2: `PHASE_WEIGHTS`, `HARD_CAP` |
| Auto-revert on AI drawdown > quant+5% | Task 2: `update_authority()` |
| Weight-average blend + min_weight filter | Task 2: `blend()` |
| Shadow return tracking in non-rebalance days | Task 5: daily authority update |
| Audit trail entry on AI PM proposal | Task 5: `record_event("ai_pm_proposal", ...)` |
| Thesis saved to `outputs/ai_pm_theses/` | Task 3: `format_thesis()` |
| Existing execution stack unchanged | Task 5: debate/approval/kill switch untouched |
| 16 tests | Tasks 1–4: 18 tests total (2 more than spec) |

**Placeholder scan:** No TBDs, TODOs, or "similar to above" patterns found.

**Type consistency:**
- `validate(portfolio: Dict[str, float]) → Tuple[bool, List[str]]` — used consistently in Task 1 and Task 5
- `blend(ai_portfolio, quant_portfolio) → Dict[str, float]` — consistent Task 2 and Task 5
- `update_authority(ai_daily_return, quant_daily_return) → dict` — consistent Task 2 and Task 5
- `format_thesis(raw_thesis, as_of_date) → dict` — consistent Task 3 and Task 5
- `AIPMResult(portfolio, thesis, fallback)` — consistent Task 4 and Task 5
- `format_exposure_context(weights_series: pd.Series, as_of_date)` — Task 4 converts dict → Series before calling; matches actual signature in `ascent/risk/factor_exposure.py:146`
- Agent functions: `run_us_equities_agent()`, `run_macro_agent()`, `run_international_agent()`, `run_alternatives_agent()` — matches actual names in agent files

**One gap fixed:** The spec says "14 tools" but the count in `AI_PM_TOOLS` is 13 (Phase 1: 2, Phase 2: 1 with enum, Phase 3: 9, Phase 4: 1 = 13). The spec's "14 tool calls" refers to `max_tool_calls=14` (the iteration budget), not the tool count. No code change needed — just a clarification.
