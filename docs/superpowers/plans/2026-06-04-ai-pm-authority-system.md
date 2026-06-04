# AI PM Progressive Authority System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the AI PM real capital allocation starting at 5% today, with a 5-level career ladder (Analyst→CEO), daily Python learning loop, four-track counterfactual (A★/A/B/D), level-specific guardrails, anti-hallucination schema enforcement, sector thesis requirement, and GitHub Pages dashboard integration — all within a $5/year API budget.

**Architecture:** `earned_authority.py` (rewritten as 5-level state machine) drives promotion/demotion using Sortino ratio on Track D vs Track A★ daily returns. A new `ai_pm_guardrails.py` enforces per-level constraints before blending. A new `ai_pm_counterfactual.py` snapshots quant and AI PM weights at each rebalance and scores all four tracks daily. A new `ai_pm_perf_feedback.py` computes the learning brief in pure Python (zero LLM cost) every day after `_log_holdings()`. `run_all_agents.py` gets a new Haiku daily-view block (non-rebalance days) and updated quant-snapshot + decision-log writes on rebalance days. `agents/ai_pm_agent.py` gains a required sector thesis schema field, Sharpe-optimization objective header, and Phase 1→2 context-strip enforcement. The GitHub Pages dashboard gets four new sections wired to the new log files.

**Tech Stack:** Python 3.12, existing `.venv`, `statistics` stdlib (Sortino), `pandas` (price lookups), `anthropic` SDK (Haiku daily view), Chart.js (already in dashboard).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Rewrite | `ascent/strategy/earned_authority.py` | 5-level state machine, Sortino promotion/demotion, cooldown, stuck alert |
| Create | `ascent/strategy/ai_pm_guardrails.py` | Per-level weight/type/correlation/tracking-error guardrails |
| Create | `ascent/monitoring/ai_pm_counterfactual.py` | Snapshot A★/A/D, score daily A★/A/B/C/D returns |
| Create | `ascent/strategy/ai_pm_perf_feedback.py` | Daily Python feedback brief, outcome scoring, gate checks |
| Modify | `run_all_agents.py` | Haiku daily view, quant snapshots, feedback trigger, decision log |
| Modify | `agents/ai_pm_agent.py` | Sector thesis schema, Sharpe objective, Phase 1→2 strip, Opus trigger |
| Modify | `scripts/generate_performance_page.py` | New loaders + 4 HTML section builders |
| Bootstrap | `data_cache/earned_authority.json` | Manually set level=1, ai_weight=0.05 |
| New tests | `tests/test_ai_pm_authority.py` | earned_authority + guardrails |
| New tests | `tests/test_ai_pm_counterfactual.py` | counterfactual engine |
| New tests | `tests/test_ai_pm_perf_feedback.py` | feedback computation |

---

## Task 1: Rewrite earned_authority.py — 5-level state machine

**Files:**
- Modify: `ascent/strategy/earned_authority.py`
- Test: `tests/test_ai_pm_authority.py`

- [ ] **Step 1.1: Write failing tests first**

```python
# tests/test_ai_pm_authority.py
import json, tempfile, pytest
from pathlib import Path
from unittest.mock import patch

LEVEL_WEIGHTS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]
LEVEL_TITLES  = ["Shadow", "Analyst", "Associate", "Manager", "Director", "CEO"]

def _write_state(tmp, state):
    p = Path(tmp) / "earned_authority.json"
    p.write_text(json.dumps(state))
    return p

def _default_state(level=1, days=0):
    return {
        "level": level, "title": LEVEL_TITLES[level],
        "ai_weight": LEVEL_WEIGHTS[level],
        "level_start_date": "2026-06-04",
        "days_at_level": days,
        "days_stuck": days,
        "in_cooldown": False,
        "cooldown_until": None,
        "auto_revert_count": 0,
        "last_updated": "2026-06-03",  # yesterday so today processes
        "track_d_returns": [],
        "track_astar_returns": [],
        "disable_sleeve_priors": False,
        # legacy compat
        "phase": level, "ai_returns_21d": [], "quant_returns_21d": [],
    }

def test_get_state_defaults_on_missing_file():
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "earned_authority.json"
        with patch.object(ea, "STATE_PATH", p):
            s = ea.get_state()
    assert s["level"] == 0
    assert s["ai_weight"] == 0.0

def test_blend_at_level1_uses_5pct():
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        sp = _write_state(tmp, _default_state(level=1))
        with patch.object(ea, "STATE_PATH", sp):
            result = ea.blend({"STRL": 1.0}, {"AAPL": 0.5, "MSFT": 0.5})
    # 5% AI + 95% quant → STRL should appear at ~5% weight
    assert "STRL" in result
    assert abs(result["STRL"] - 0.05) < 0.01

def test_sortino_positive_only_penalizes_downside():
    import ascent.strategy.earned_authority as ea
    # All positive returns → high Sortino
    pos_returns = [0.01] * 21
    # Mixed with negatives → lower Sortino
    mix_returns = [0.01, -0.02, 0.01, -0.02] * 5 + [0.01]
    assert ea._sortino(pos_returns) > ea._sortino(mix_returns)

def test_cooldown_blocks_promotion():
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        state = _default_state(level=1)
        state["in_cooldown"] = True
        state["cooldown_until"] = "2099-01-01"
        sp = _write_state(tmp, state)
        shadow = Path(tmp) / "shadow.jsonl"
        with patch.object(ea, "STATE_PATH", sp):
            with patch.object(ea, "SHADOW_RETURNS_PATH", shadow):
                # Even with great returns, should not promote during cooldown
                for _ in range(25):
                    ea.update_authority(0.05, -0.01)
                s = ea.get_state()
    assert s["level"] == 1  # no promotion

def test_catastrophic_demotion_reverts_to_shadow():
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        state = _default_state(level=3)
        sp = _write_state(tmp, state)
        shadow = Path(tmp) / "shadow.jsonl"
        with patch.object(ea, "STATE_PATH", sp):
            with patch.object(ea, "SHADOW_RETURNS_PATH", shadow):
                # Track D 10pp worse than Track A★ in one day
                ea.update_authority(track_d_return=-0.12, track_astar_return=-0.02)
                s = ea.get_state()
    assert s["level"] == 0

def test_stuck_alert_fires_at_63_days():
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        state = _default_state(level=1)
        state["days_stuck"] = 63
        sp = _write_state(tmp, state)
        with patch.object(ea, "STATE_PATH", sp):
            s = ea.get_state()
            assert ea.is_stuck(s) is True
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_ai_pm_authority.py -v 2>&1 | head -30
```
Expected: all fail with ImportError or AttributeError.

- [ ] **Step 1.3: Rewrite earned_authority.py**

```python
# ascent/strategy/earned_authority.py
from __future__ import annotations
import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data_cache" / "earned_authority.json"
SHADOW_RETURNS_PATH = Path(__file__).resolve().parent.parent.parent / "data_cache" / "ai_pm_shadow_returns.jsonl"

LEVEL_WEIGHTS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]
LEVEL_TITLES  = ["Shadow", "Analyst", "Associate", "Manager", "Director", "CEO"]
HARD_CAP      = 0.80
MIN_WEIGHT    = 0.02
_TRADING_DAYS_PER_YEAR = 252

# Promotion config per transition (from_level → to_level)
PROMOTION_CONFIG = {
    (1, 2): {"window": 21, "sortino_edge": 0.20, "hit_rate": 0.52, "profit_factor": 1.2, "min_decisions": 5,  "primary_window": 10},
    (2, 3): {"window": 21, "sortino_edge": 0.30, "hit_rate": 0.55, "profit_factor": 1.3, "min_decisions": 8,  "primary_window": 10},
    (3, 4): {"window": 42, "sortino_edge": 0.40, "hit_rate": 0.55, "profit_factor": 1.3, "min_decisions": 10, "primary_window": 63},
    (4, 5): {"window": 63, "sortino_edge": 0.50, "hit_rate": 0.58, "profit_factor": 1.4, "min_decisions": 15, "primary_window": 63},
}


def _sortino(returns: List[float]) -> float:
    """Sortino ratio: mean / downside_deviation * sqrt(252). Returns 0 if insufficient data."""
    if len(returns) < 5:
        return 0.0
    mean = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return mean * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100  # cap at high value
    downside_dev = math.sqrt(sum(r ** 2 for r in neg) / len(returns))
    return 0.0 if downside_dev == 0 else mean / downside_dev * math.sqrt(_TRADING_DAYS_PER_YEAR)


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


def is_stuck(state: dict) -> bool:
    return state.get("days_stuck", 0) >= 63


def get_state() -> dict:
    if STATE_PATH.exists():
        try:
            s = json.loads(STATE_PATH.read_text())
            # migrate old phase-based state
            if "phase" in s and "level" not in s:
                s["level"] = s["phase"]
                s["title"] = LEVEL_TITLES[min(s["level"], 5)]
                s["ai_weight"] = LEVEL_WEIGHTS[min(s["level"], 5)]
                s.setdefault("track_d_returns", [])
                s.setdefault("track_astar_returns", [])
                s.setdefault("days_at_level", 0)
                s.setdefault("days_stuck", 0)
                s.setdefault("in_cooldown", False)
                s.setdefault("cooldown_until", None)
                s.setdefault("disable_sleeve_priors", False)
            return s
        except Exception as exc:
            log.warning("[EarnedAuthority] Corrupt state, resetting: %s", exc)
    return {
        "level": 0, "title": "Shadow", "ai_weight": 0.0,
        "level_start_date": str(date.today()),
        "days_at_level": 0, "days_stuck": 0,
        "in_cooldown": False, "cooldown_until": None,
        "auto_revert_count": 0, "last_updated": str(date.today()),
        "track_d_returns": [], "track_astar_returns": [],
        "disable_sleeve_priors": False,
        # legacy compat
        "phase": 0, "ai_returns_21d": [], "quant_returns_21d": [],
    }


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def update_authority(
    track_d_return: float,
    track_astar_return: float,
    n_decisions_evaluated: int = 0,
    hit_rate: Optional[float] = None,
    profit_factor: Optional[float] = None,
    fade_rate: Optional[float] = None,
    regime_gate_pass: bool = True,
) -> dict:
    """Append daily Track D / Track A★ returns, check demotion then promotion. Returns state."""
    state = get_state()
    today = str(date.today())

    if state.get("last_updated") == today:
        log.debug("[EarnedAuthority] Already updated today, skipping")
        return state

    level = state.get("level", 0)

    # Update buffers (keep last 63 days for Level 3+ windows)
    d_buf  = (state.get("track_d_returns", [])     + [float(track_d_return)])[-63:]
    as_buf = (state.get("track_astar_returns", []) + [float(track_astar_return)])[-63:]
    state["track_d_returns"]     = d_buf
    state["track_astar_returns"] = as_buf

    # Legacy buffer update for backward compat
    state["ai_returns_21d"]    = d_buf[-21:]
    state["quant_returns_21d"] = as_buf[-21:]

    _log_shadow_return(today, track_d_return, track_astar_return, state["ai_weight"])

    # ── Cooldown check ──
    cooldown_until = state.get("cooldown_until")
    if cooldown_until and today <= cooldown_until:
        state["days_at_level"] = state.get("days_at_level", 0) + 1
        state["days_stuck"]    = state.get("days_stuck", 0) + 1
        state["in_cooldown"]   = True
        state["last_updated"]  = today
        _save_state(state)
        return state
    elif cooldown_until and today > cooldown_until:
        state["in_cooldown"]   = False
        state["cooldown_until"] = None
        log.info("[EarnedAuthority] Cooldown expired")

    # ── Demotion checks (Track D vs Track A★) ──
    if level > 0 and len(d_buf) >= 10:
        daily_diff = track_d_return - track_astar_return

        # Catastrophic: single day 10pp worse
        if daily_diff <= -0.10:
            log.warning("[EarnedAuthority] CATASTROPHIC demotion: Track D %.2f%% vs A★ %.2f%%",
                        track_d_return * 100, track_astar_return * 100)
            _apply_demotion(state, today, target_level=0, cooldown_days=5)
            _save_state(state)
            return state

        # Hard: single day 5pp worse
        if daily_diff <= -0.05:
            log.warning("[EarnedAuthority] HARD demotion: -%.2fpp vs A★", abs(daily_diff) * 100)
            _apply_demotion(state, today, target_level=max(0, level - 1), cooldown_days=5)
            _save_state(state)
            return state

        # Soft: drawdown gap over rolling window
        dd_d  = _max_drawdown(d_buf[-21:])
        dd_as = _max_drawdown(as_buf[-21:])
        if dd_d > dd_as + 0.03:
            log.warning("[EarnedAuthority] SOFT demotion: DD gap %.2fpp", (dd_d - dd_as) * 100)
            _apply_demotion(state, today, target_level=max(0, level - 1), cooldown_days=5)
            _save_state(state)
            return state

    # ── Promotion check ──
    cfg = PROMOTION_CONFIG.get((level, level + 1))
    if cfg and not state.get("in_cooldown"):
        win = cfg["window"]
        if len(d_buf) >= win:
            sortino_d  = _sortino(d_buf[-win:])
            sortino_as = _sortino(as_buf[-win:])
            edge       = sortino_d - sortino_as

            gates = {
                "sortino_edge":    edge > cfg["sortino_edge"],
                "hit_rate":        (hit_rate or 0) >= cfg["hit_rate"],
                "profit_factor":   (profit_factor or 0) > cfg["profit_factor"],
                "min_decisions":   n_decisions_evaluated >= cfg["min_decisions"],
                "fade_rate":       (fade_rate or 0) <= 0.30,
                "regime_gate":     regime_gate_pass,
            }
            if all(gates.values()):
                new_level = level + 1
                log.info("[EarnedAuthority] PROMOTED to Level %d (%s, %.0f%%)",
                         new_level, LEVEL_TITLES[new_level], LEVEL_WEIGHTS[new_level] * 100)
                state.update({
                    "level": new_level,
                    "title": LEVEL_TITLES[new_level],
                    "ai_weight": min(LEVEL_WEIGHTS[new_level], HARD_CAP),
                    "phase": new_level,  # legacy compat
                    "level_start_date": today,
                    "days_at_level": 0,
                    "days_stuck": 0,
                    "track_d_returns": [],
                    "track_astar_returns": [],
                    "ai_returns_21d": [],
                    "quant_returns_21d": [],
                })

    # Increment counters
    state["days_at_level"] = state.get("days_at_level", 0) + 1
    state["days_stuck"]    = state.get("days_stuck", 0) + 1
    if is_stuck(state):
        log.warning("[AIPMAuthority] WARNING: AI PM at Level %d for 63+ days without promoting", level)

    state["last_updated"] = today
    _save_state(state)
    return state


def _apply_demotion(state: dict, today: str, target_level: int, cooldown_days: int) -> None:
    from datetime import timedelta
    state.update({
        "level": target_level,
        "title": LEVEL_TITLES[target_level],
        "ai_weight": LEVEL_WEIGHTS[target_level],
        "phase": target_level,
        "level_start_date": today,
        "days_at_level": 0,
        "days_stuck": 0,
        "in_cooldown": True,
        "cooldown_until": str((datetime.strptime(today, "%Y-%m-%d") + timedelta(days=cooldown_days)).date()),
        "auto_revert_count": state.get("auto_revert_count", 0) + 1,
        "track_d_returns": [],
        "track_astar_returns": [],
        "ai_returns_21d": [],
        "quant_returns_21d": [],
    })


def blend(ai_portfolio: Dict[str, float], quant_portfolio: Dict[str, float]) -> Dict[str, float]:
    """Weight-average AI PM + quant portfolios using current level's ai_weight."""
    state = get_state()
    ai_w  = state["ai_weight"]
    qt_w  = 1.0 - ai_w

    blended: Dict[str, float] = {}
    for sym in set(ai_portfolio) | set(quant_portfolio):
        w = ai_w * ai_portfolio.get(sym, 0.0) + qt_w * quant_portfolio.get(sym, 0.0)
        if w >= MIN_WEIGHT:
            blended[sym] = w

    total = sum(blended.values())
    if total <= 0:
        return dict(quant_portfolio)
    return {sym: w / total for sym, w in blended.items()}


def _log_shadow_return(today: str, d_ret: float, as_ret: float, ai_weight: float) -> None:
    try:
        SHADOW_RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_RETURNS_PATH, "a") as f:
            f.write(json.dumps({
                "date": today, "ai_return": d_ret, "quant_return": as_ret,
                "ai_weight_at_time": ai_weight,
            }) + "\n")
    except Exception as exc:
        log.warning("[EarnedAuthority] Could not log shadow return: %s", exc)
```

- [ ] **Step 1.4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_authority.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 1.5: Verify existing tests still pass**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v -k "blend or phase or revert" 2>&1 | tail -15
```
Expected: blend/phase tests pass (backward compat preserved).

- [ ] **Step 1.6: Commit**

```bash
git add ascent/strategy/earned_authority.py tests/test_ai_pm_authority.py
git commit -m "feat: rewrite earned_authority.py — 5-level career ladder with Sortino promotion/demotion"
```

---

## Task 2: Create ai_pm_guardrails.py

**Files:**
- Create: `ascent/strategy/ai_pm_guardrails.py`
- Test: `tests/test_ai_pm_authority.py` (extend)

- [ ] **Step 2.1: Add guardrail tests**

Append to `tests/test_ai_pm_authority.py`:

```python
# ── guardrails ────────────────────────────────────────────────────────────────

def test_level1_blocks_reduce():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    quant = {"AAPL": 0.10, "MSFT": 0.10}
    ai_pm = {"AAPL": 0.07, "MSFT": 0.10}  # AAPL is a REDUCE
    alpha_scores = {"AAPL": 0.8, "MSFT": 0.9}  # both top-50%
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    # AAPL should be clamped back to quant weight (REDUCE blocked at Level 1)
    assert result["AAPL"] >= quant["AAPL"] - 0.001
    assert any("REDUCE" in v or "reduce" in v.lower() for v in violations)

def test_level1_blocks_amplify_bottom_50pct_alpha():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    quant = {"AAPL": 0.07, "WEAK": 0.05}
    ai_pm = {"AAPL": 0.09, "WEAK": 0.09}  # amplifying WEAK
    # WEAK ranks below 50th percentile
    alpha_scores = {"AAPL": 0.80, "WEAK": 0.20}
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    # WEAK amplification should be blocked
    assert abs(result.get("WEAK", 0.05) - 0.05) < 0.001
    assert any("WEAK" in v for v in violations)

def test_level1_max_weight_change_capped_at_2pp():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    quant = {"AAPL": 0.07}
    ai_pm = {"AAPL": 0.15}  # +8pp, above 2pp cap
    alpha_scores = {"AAPL": 0.90}
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    assert result["AAPL"] <= 0.07 + 0.02 + 1e-6

def test_max_overrides_enforced():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    # Level 1 allows max 2 overrides
    quant = {f"S{i}": 0.07 for i in range(5)}
    ai_pm = {f"S{i}": 0.09 for i in range(5)}  # 5 amplifications
    alpha_scores = {f"S{i}": 0.90 for i in range(5)}
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    n_changed = sum(1 for sym in quant if abs(result.get(sym, quant[sym]) - quant[sym]) > 0.001)
    assert n_changed <= 2
```

- [ ] **Step 2.2: Run to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_authority.py::test_level1_blocks_reduce -v
```
Expected: ImportError (module not created yet).

- [ ] **Step 2.3: Create ai_pm_guardrails.py**

```python
# ascent/strategy/ai_pm_guardrails.py
from __future__ import annotations
import logging
import math
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# Per-level guardrail config
_LEVEL_CONFIG = {
    1: {"max_change": 0.02, "max_new": 0, "allowed_types": {"AMPLIFY"},          "max_overrides": 2,  "max_te": 0.003},
    2: {"max_change": 0.04, "max_new": 1, "allowed_types": {"AMPLIFY", "HOLD"},   "max_overrides": 3,  "max_te": 0.005},
    3: {"max_change": 0.06, "max_new": 2, "allowed_types": {"AMPLIFY", "HOLD", "REDUCE"}, "max_overrides": 4, "max_te": 0.008},
    4: {"max_change": 0.08, "max_new": 3, "allowed_types": {"AMPLIFY", "HOLD", "REDUCE", "NEW"}, "max_overrides": 5, "max_te": 0.012},
    5: {"max_change": 0.10, "max_new": 5, "allowed_types": {"AMPLIFY", "HOLD", "REDUCE", "NEW"}, "max_overrides": 999, "max_te": 1.0},
}
_CORR_BLOCK_THRESHOLD = 0.65
_MEDIAN_ALPHA_PERCENTILE = 0.50  # Level 1: only top 50% of alpha scores


def apply_guardrails(
    ai_pm_portfolio: Dict[str, float],
    quant_portfolio: Dict[str, float],
    quant_alpha_scores: Dict[str, float],
    level: int,
    price_returns_63d: Dict[str, List[float]] | None = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Apply level-specific guardrails to ai_pm_portfolio.
    Returns (filtered_portfolio, violations_log).
    Violations are logged but the call never raises.
    """
    if level not in _LEVEL_CONFIG:
        return dict(quant_portfolio), [f"Unknown level {level}, falling back to quant"]

    cfg        = _LEVEL_CONFIG[level]
    violations = []
    result     = dict(quant_portfolio)  # start from quant, apply AI PM changes

    # Classify each AI PM proposed change
    overrides_applied = 0
    quant_syms  = set(quant_portfolio.keys())
    alpha_vals  = sorted(quant_alpha_scores.values())
    median_alpha = alpha_vals[len(alpha_vals) // 2] if alpha_vals else 0.0

    for sym, ai_w in ai_pm_portfolio.items():
        quant_w = quant_portfolio.get(sym, 0.0)
        delta   = ai_w - quant_w

        # Classify override type
        if sym not in quant_syms:
            ov_type = "NEW"
        elif delta > 0.001:
            ov_type = "AMPLIFY"
        elif delta < -0.001:
            ov_type = "REDUCE"
        else:
            ov_type = "HOLD"  # negligible change

        if ov_type == "HOLD":
            continue  # no override needed

        # Check allowed types
        if ov_type not in cfg["allowed_types"]:
            violations.append(f"{sym}: {ov_type} not allowed at Level {level}")
            continue

        # Level 1: amplification quality — only top 50% alpha names
        if level <= 2 and ov_type == "AMPLIFY":
            score = quant_alpha_scores.get(sym, 0.0)
            if score < median_alpha:
                violations.append(f"{sym}: AMPLIFY blocked — alpha score {score:.3f} below median {median_alpha:.3f}")
                continue

        # Max weight change cap
        capped_delta = max(-cfg["max_change"], min(cfg["max_change"], delta))
        if abs(capped_delta - delta) > 0.001:
            violations.append(f"{sym}: weight change capped {delta:+.3f} → {capped_delta:+.3f} (Level {level} max ±{cfg['max_change']:.0%})")
            delta = capped_delta

        # New symbol cap
        if ov_type == "NEW":
            new_syms_so_far = sum(1 for s in result if s not in quant_syms)
            if new_syms_so_far >= cfg["max_new"]:
                violations.append(f"{sym}: new symbol blocked (max {cfg['max_new']} new at Level {level})")
                continue

        # Override correlation check (if price history provided)
        if price_returns_63d and overrides_applied > 0:
            existing_overrides = [s for s in result if abs(result[s] - quant_portfolio.get(s, 0.0)) > 0.001]
            for prev_sym in existing_overrides[:3]:
                corr = _rolling_corr(price_returns_63d.get(sym, []), price_returns_63d.get(prev_sym, []))
                if corr > _CORR_BLOCK_THRESHOLD:
                    violations.append(f"{sym}: corr {corr:.2f} with {prev_sym} > {_CORR_BLOCK_THRESHOLD} — override blocked")
                    continue

        # Max overrides cap
        if overrides_applied >= cfg["max_overrides"]:
            violations.append(f"{sym}: max overrides ({cfg['max_overrides']}) reached at Level {level}")
            continue

        result[sym] = quant_w + delta
        overrides_applied += 1

    # Apply tracking error cap
    result, te_violations = _apply_tracking_error_cap(result, quant_portfolio, cfg["max_te"])
    violations.extend(te_violations)

    return result, violations


def _rolling_corr(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b), 63)
    if n < 10:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da  = math.sqrt(sum((x - ma) ** 2 for x in a))
    db  = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db > 0 else 0.0


def _apply_tracking_error_cap(
    blended: Dict[str, float],
    quant: Dict[str, float],
    max_te: float,
) -> Tuple[Dict[str, float], List[str]]:
    """Diagonal covariance proxy: TE = sqrt(sum((w_blend - w_quant)^2 * var_i))
    Using var_i = 0.0004 (≈ 2% daily vol per name) as proxy."""
    VAR_PROXY = 0.0004
    diffs = {sym: blended.get(sym, 0.0) - quant.get(sym, 0.0) for sym in set(blended) | set(quant)}
    te = math.sqrt(sum(d ** 2 * VAR_PROXY for d in diffs.values()))

    if te <= max_te:
        return blended, []

    # Scale back all AI PM changes proportionally
    scale = max_te / te
    result = {}
    for sym in set(blended) | set(quant):
        q_w = quant.get(sym, 0.0)
        b_w = blended.get(sym, 0.0)
        result[sym] = q_w + (b_w - q_w) * scale

    # Renormalize
    total = sum(v for v in result.values() if v > 0.02)
    if total > 0:
        result = {s: v / total for s, v in result.items() if v > 0.02}

    return result, [f"Tracking error {te:.4f} > {max_te:.4f} cap — AI PM changes scaled by {scale:.2f}"]


def check_conviction_inflation(proposals: Dict[str, str]) -> Dict[str, str]:
    """Downgrade 'high' conviction to 'medium' if >40% are high."""
    high_count = sum(1 for v in proposals.values() if v == "high")
    if len(proposals) == 0 or high_count / len(proposals) <= 0.40:
        return proposals
    # Keep top 40%, downgrade rest
    threshold = max(1, int(len(proposals) * 0.40))
    high_syms = [s for s, v in proposals.items() if v == "high"][:threshold]
    return {s: v if (v != "high" or s in high_syms) else "medium" for s, v in proposals.items()}
```

- [ ] **Step 2.4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_authority.py -v
```
Expected: all tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add ascent/strategy/ai_pm_guardrails.py tests/test_ai_pm_authority.py
git commit -m "feat: add ai_pm_guardrails.py — level-specific weight/type/correlation/TE guardrails"
```

---

## Task 3: Create ai_pm_counterfactual.py

**Files:**
- Create: `ascent/monitoring/ai_pm_counterfactual.py`
- Test: `tests/test_ai_pm_counterfactual.py`

- [ ] **Step 3.1: Write failing tests**

```python
# tests/test_ai_pm_counterfactual.py
import json, tempfile, pytest
from pathlib import Path
from datetime import date
from unittest.mock import patch

def test_snapshot_quant_star_is_idempotent():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_quant_star
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_star.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_STAR_LOG", log_path):
            snapshot_quant_star(date(2026, 6, 4), {"AAPL": 0.5, "MSFT": 0.5})
            snapshot_quant_star(date(2026, 6, 4), {"AAPL": 0.6, "MSFT": 0.4})  # re-run
            lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1  # idempotent — second write skipped
    assert json.loads(lines[0])["weights"]["AAPL"] == 0.5  # first write preserved

def test_snapshot_ai_pm_normalizes_weights():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_ai.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 4), {"AAPL": 0.6, "MSFT": 0.6})  # sums to 1.2
            entry = json.loads(log_path.read_text().strip())
    total = sum(entry["weights"].values())
    assert abs(total - 1.0) < 0.001

def test_score_daily_appends_all_tracks():
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 5),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.012,
                spy_return=0.008,
                prices={"AAPL": {"prev": 100.0, "curr": 101.5}, "MSFT": {"prev": 200.0, "curr": 201.0}},
            )
        entry = json.loads(log_path.read_text().strip())
    assert "track_astar_return" in entry
    assert "track_a_return" in entry
    assert "track_b_return" in entry
    assert "track_c_return" in entry
    assert "track_d_return" in entry
    assert abs(entry["track_b_return"] - 0.012) < 0.0001
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py -v 2>&1 | head -15
```

- [ ] **Step 3.3: Create ai_pm_counterfactual.py**

```python
# ascent/monitoring/ai_pm_counterfactual.py
"""
Four-track counterfactual engine.

Track A★ — Pure Quant (no Phase 1 priors)
Track A  — Quant + Phase 1 sleeve priors
Track B  — Actual portfolio (Alpaca last_equity)
Track C  — SPY benchmark
Track D  — Pure AI PM at 100% weight (diagnostic only)
"""
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent
QUANT_STAR_LOG = _REPO / "logs" / "counterfactual_quant_star_snapshots.jsonl"
QUANT_LOG      = _REPO / "logs" / "counterfactual_quant_snapshots.jsonl"
AI_PM_LOG      = _REPO / "logs" / "counterfactual_ai_snapshots.jsonl"
DAILY_LOG      = _REPO / "logs" / "counterfactual_daily.jsonl"


def _idempotent_write(path: Path, date_str: str, record: dict) -> bool:
    """Write record only if no entry for date_str exists. Returns True if written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                if json.loads(line).get("date") == date_str:
                    return False  # already written
            except Exception:
                pass
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def snapshot_quant_star(run_date: date, weights: Dict[str, float]) -> None:
    """Track A★: quant with default regime weights, zero Phase 1 influence."""
    _idempotent_write(QUANT_STAR_LOG, run_date.isoformat(), {
        "date": run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    })


def snapshot_quant(run_date: date, weights: Dict[str, float]) -> None:
    """Track A: quant after Phase 1 sleeve priors applied, before Phase 2 blend."""
    _idempotent_write(QUANT_LOG, run_date.isoformat(), {
        "date": run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    })


def snapshot_ai_pm(run_date: date, weights: Dict[str, float]) -> None:
    """Track D: AI PM proposed portfolio, normalized to sum=1.0."""
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    _idempotent_write(AI_PM_LOG, run_date.isoformat(), {
        "date": run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    })


def _portfolio_return(weights: Dict[str, float], prices: Dict[str, dict]) -> float:
    """Compute weighted portfolio return from prev/curr price dict."""
    ret = 0.0
    for sym, w in weights.items():
        p = prices.get(sym)
        if p and p.get("prev", 0) > 0:
            ret += w * (p["curr"] - p["prev"]) / p["prev"]
    return ret


def _load_last_snapshot(path: Path) -> Optional[dict]:
    """Load the most recent snapshot from a jsonl file."""
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        try:
            last = json.loads(line)
        except Exception:
            pass
    return last


def score_daily(
    run_date: date,
    quant_star_weights: Optional[Dict[str, float]],
    quant_weights: Optional[Dict[str, float]],
    ai_pm_weights: Optional[Dict[str, float]],
    track_b_return: float,
    spy_return: float,
    prices: Dict[str, dict],
) -> dict:
    """Compute all track daily returns and append to DAILY_LOG."""
    as_ret = _portfolio_return(quant_star_weights or {}, prices) if quant_star_weights else 0.0
    a_ret  = _portfolio_return(quant_weights or {}, prices) if quant_weights else 0.0
    d_ret  = _portfolio_return(ai_pm_weights or {}, prices) if ai_pm_weights else 0.0

    record = {
        "date":               run_date.isoformat(),
        "track_astar_return": round(as_ret, 6),
        "track_a_return":     round(a_ret, 6),
        "track_b_return":     round(track_b_return, 6),
        "track_c_return":     round(spy_return, 6),
        "track_d_return":     round(d_ret, 6),
    }

    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DAILY_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_snapshots() -> tuple[dict, dict, dict]:
    """Returns (quant_star_weights, quant_weights, ai_pm_weights) from last rebalance."""
    def _w(snap):
        return snap.get("weights", {}) if snap else {}
    return (
        _w(_load_last_snapshot(QUANT_STAR_LOG)),
        _w(_load_last_snapshot(QUANT_LOG)),
        _w(_load_last_snapshot(AI_PM_LOG)),
    )


def load_daily_records() -> list[dict]:
    if not DAILY_LOG.exists():
        return []
    records = []
    for line in DAILY_LOG.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def print_cumulative_report() -> None:
    records = load_daily_records()
    if not records:
        print("[Counterfactual] No data yet")
        return

    def cumret(key):
        v = 1.0
        for r in records:
            v *= (1 + r.get(key, 0.0))
        return (v - 1) * 100

    as_cum = cumret("track_astar_return")
    a_cum  = cumret("track_a_return")
    b_cum  = cumret("track_b_return")
    c_cum  = cumret("track_c_return")
    d_cum  = cumret("track_d_return")

    start = records[0]["date"]
    end   = records[-1]["date"]
    n     = len(records)

    print(f"[Counterfactual] Since AI PM live ({start} → {end}, {n} days):")
    print(f"  Track A★ (Pure Quant):    {as_cum:+.2f}%")
    print(f"  Track A  (Quant+P1):      {a_cum:+.2f}%")
    print(f"  Track B  (Actual):        {b_cum:+.2f}%")
    print(f"  Track C  (SPY):           {c_cum:+.2f}%")
    print(f"  Track D  (Pure AI PM):    {d_cum:+.2f}%")
    print(f"  AI value add (B−A★):      {b_cum - as_cum:+.2f}pp vs pure quant")
    print(f"  AI signal quality (D−A★): {d_cum - as_cum:+.2f}pp — what full authority would add")
```

- [ ] **Step 3.4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add ascent/monitoring/ai_pm_counterfactual.py tests/test_ai_pm_counterfactual.py
git commit -m "feat: add ai_pm_counterfactual.py — Track A★/A/B/C/D daily scoring"
```

---

## Task 4: Create ai_pm_perf_feedback.py

**Files:**
- Create: `ascent/strategy/ai_pm_perf_feedback.py`
- Test: `tests/test_ai_pm_perf_feedback.py`

- [ ] **Step 4.1: Write failing tests**

```python
# tests/test_ai_pm_perf_feedback.py
import json, tempfile
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch

def _make_decision(sym="STRL", ai_w=0.09, quant_w=0.07, days_ago=12, ov_type="amplify"):
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return {
        "date": d, "level": 1, "ai_weight": 0.05,
        "overrides_applied": [{"symbol": sym, "type": ov_type, "ai_w": ai_w, "quant_w": quant_w}],
    }

def test_compute_feedback_writes_file():
    from ascent.strategy.ai_pm_perf_feedback import compute_feedback
    with tempfile.TemporaryDirectory() as tmp:
        fb_path = Path(tmp) / "feedback.json"
        dec_path = Path(tmp) / "decisions.jsonl"
        dec_path.write_text(json.dumps(_make_decision()) + "\n")
        with patch("ascent.strategy.ai_pm_perf_feedback.FEEDBACK_PATH", fb_path):
            with patch("ascent.strategy.ai_pm_perf_feedback.DECISION_LOG", dec_path):
                with patch("ascent.strategy.ai_pm_perf_feedback._fetch_price_return", return_value=0.03):
                    compute_feedback()
        assert fb_path.exists()
        fb = json.loads(fb_path.read_text())
        assert "level" in fb
        assert "promotion_gates" in fb

def test_incremental_alpha_uses_delta_weight():
    from ascent.strategy.ai_pm_perf_feedback import _incremental_alpha
    # AI PM held 9%, quant held 7%, stock returned +10%
    # Incremental = (0.09 - 0.07) * 0.10 = 0.002
    result = _incremental_alpha(ai_w=0.09, quant_w=0.07, stock_return=0.10)
    assert abs(result - 0.002) < 0.0001

def test_fade_detection():
    from ascent.strategy.ai_pm_perf_feedback import _is_fade
    assert _is_fade(outcome_10d=0.02, outcome_21d=-0.01) is True
    assert _is_fade(outcome_10d=0.02, outcome_21d=0.03) is False
    assert _is_fade(outcome_10d=-0.01, outcome_21d=-0.02) is False  # not a fade if already negative at 10d

def test_confidence_label():
    from ascent.strategy.ai_pm_perf_feedback import _confidence
    assert _confidence(n=2) == "low"
    assert _confidence(n=8) == "medium"
    assert _confidence(n=20) == "high"
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_perf_feedback.py -v 2>&1 | head -10
```

- [ ] **Step 4.3: Create ai_pm_perf_feedback.py**

```python
# ascent/strategy/ai_pm_perf_feedback.py
"""
Daily Python-computed learning brief. Zero LLM cost.
Reads decision_log + counterfactual_daily + earned_authority.
Writes data_cache/ai_pm_perf_feedback.json.
Called after _log_holdings() in run_all_agents.py every day.
"""
from __future__ import annotations
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPO        = Path(__file__).resolve().parent.parent.parent
FEEDBACK_PATH = _REPO / "data_cache" / "ai_pm_perf_feedback.json"
DECISION_LOG  = _REPO / "logs" / "ai_pm_decision_log.jsonl"
DAILY_LOG     = _REPO / "logs" / "counterfactual_daily.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence(n: int) -> str:
    if n < 5:  return "low"
    if n < 15: return "medium"
    return "high"


def _incremental_alpha(ai_w: float, quant_w: float, stock_return: float) -> float:
    """(ai_weight - quant_weight) * return — the AI PM's true contribution."""
    return (ai_w - quant_w) * stock_return


def _is_fade(outcome_10d: Optional[float], outcome_21d: Optional[float]) -> bool:
    """A fade: positive at 10d but negative at 21d."""
    if outcome_10d is None or outcome_21d is None:
        return False
    return outcome_10d > 0 and outcome_21d < 0


def _sortino(returns: List[float]) -> float:
    if len(returns) < 5:
        return 0.0
    mean = sum(returns) / len(returns)
    neg  = [r for r in returns if r < 0]
    if not neg:
        return mean * math.sqrt(252) * 100
    dd = math.sqrt(sum(r ** 2 for r in neg) / len(returns))
    return 0.0 if dd == 0 else mean / dd * math.sqrt(252)


def _fetch_price_return(symbol: str, as_of: str, days_forward: int) -> Optional[float]:
    """Fetch total return from as_of date + days_forward. Returns None if unavailable."""
    try:
        import yfinance as yf
        import pandas as pd
        start = (date.fromisoformat(as_of)).isoformat()
        end   = (date.fromisoformat(as_of) + timedelta(days=days_forward + 5)).isoformat()
        df    = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return None
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        # Get return at approximately days_forward trading days out
        target_idx = min(days_forward, len(closes) - 1)
        return float((closes.iloc[target_idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as exc:
        log.debug("[PerfFeedback] Price fetch failed %s: %s", symbol, exc)
        return None


def _load_decisions() -> List[dict]:
    if not DECISION_LOG.exists():
        return []
    rows = []
    for line in DECISION_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return sorted(rows, key=lambda x: x.get("date", ""))


def _load_daily_records() -> List[dict]:
    if not DAILY_LOG.exists():
        return []
    rows = []
    for line in DAILY_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _load_authority_state() -> dict:
    from ascent.strategy.earned_authority import get_state
    return get_state()


# ── Score pending decisions ────────────────────────────────────────────────────

def _score_decisions(decisions: List[dict]) -> List[dict]:
    """For each override, fill in outcome_5d/10d/21d/63d if enough days have passed."""
    today = date.today()
    scored = []
    for dec in decisions:
        dec_date = date.fromisoformat(dec["date"])
        for ov in dec.get("overrides_applied", []):
            sym        = ov["symbol"]
            days_since = (today - dec_date).days

            record = {
                "date":       dec["date"],
                "symbol":     sym,
                "type":       ov.get("type", "amplify"),
                "ai_w":       ov.get("ai_w", 0.0),
                "quant_w":    ov.get("quant_w", 0.0),
                "outcome_5d":  None,
                "outcome_10d": None,
                "outcome_21d": None,
                "outcome_63d": None,
                "verdict":     None,
                "fade":        False,
                "early":       False,
            }

            for horizon, min_days, key in [(5, 5, "outcome_5d"), (10, 10, "outcome_10d"),
                                           (21, 21, "outcome_21d"), (63, 63, "outcome_63d")]:
                if days_since >= min_days:
                    raw = _fetch_price_return(sym, dec["date"], horizon)
                    if raw is not None:
                        record[key] = round(_incremental_alpha(ov.get("ai_w", 0.0),
                                                               ov.get("quant_w", 0.0), raw), 6)
                    else:
                        record[key] = 0.0  # orphaned — score as 0 (constraint 22)

            # Classify
            r10 = record["outcome_10d"]
            r21 = record["outcome_21d"]
            r63 = record["outcome_63d"]

            if r10 is not None:
                if _is_fade(r10, r21):
                    record["verdict"] = "fade"
                    record["fade"]    = True
                elif r10 < 0 and r63 is not None and r63 > 0:
                    record["verdict"] = "early"
                    record["early"]   = True
                elif r10 >= 0:
                    record["verdict"] = "win"
                else:
                    record["verdict"] = "miss"

            scored.append(record)

    return scored


# ── Main compute ───────────────────────────────────────────────────────────────

def compute_feedback() -> dict:
    """Compute and write the daily learning brief. Returns the feedback dict."""
    today    = date.today()
    state    = _load_authority_state()
    level    = state.get("level", 0)
    decisions = _load_decisions()
    daily    = _load_daily_records()

    scored   = _score_decisions(decisions)
    evaluated = [s for s in scored if s["outcome_10d"] is not None]
    pending   = [s for s in scored if s["outcome_10d"] is None]

    # ── Sortino metrics ──
    d_rets  = [r.get("track_d_return", 0.0) for r in daily[-21:]]
    as_rets = [r.get("track_astar_return", 0.0) for r in daily[-21:]]
    sortino_d  = _sortino(d_rets)
    sortino_as = _sortino(as_rets)
    n_days = len(d_rets)

    # ── Hit rate, profit factor, fade rate ──
    wins   = [s for s in evaluated if s["verdict"] == "win"]
    misses = [s for s in evaluated if s["verdict"] == "miss"]
    fades  = [s for s in evaluated if s["verdict"] == "fade"]

    hit_rate      = len(wins) / len(evaluated) if evaluated else 0.0
    gross_wins    = sum(abs(s["outcome_10d"]) for s in wins)
    gross_losses  = sum(abs(s["outcome_10d"]) for s in misses)
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (1.5 if gross_wins > 0 else 1.0)
    fade_rate     = len(fades) / len(evaluated) if evaluated else 0.0

    # ── Override type breakdown ──
    def _type_stats(ov_type):
        sub = [s for s in evaluated if s["type"].lower() == ov_type and s["outcome_10d"] is not None]
        avg = sum(s["outcome_10d"] for s in sub) / len(sub) if sub else 0.0
        return avg, len(sub)

    amp_avg, amp_n   = _type_stats("amplify")
    red_avg, red_n   = _type_stats("reduce")
    new_avg, new_n   = _type_stats("new")

    # ── Best / worst calls ──
    best  = max(evaluated, key=lambda x: x.get("outcome_10d", 0), default=None)
    worst = min(evaluated, key=lambda x: x.get("outcome_10d", 0), default=None)

    # ── Promotion gates ──
    cfg = {
        (1,2): {"sortino_edge":0.20,"hit_rate":0.52,"profit_factor":1.2,"min_decisions":5},
        (2,3): {"sortino_edge":0.30,"hit_rate":0.55,"profit_factor":1.3,"min_decisions":8},
        (3,4): {"sortino_edge":0.40,"hit_rate":0.55,"profit_factor":1.3,"min_decisions":10},
        (4,5): {"sortino_edge":0.50,"hit_rate":0.58,"profit_factor":1.4,"min_decisions":15},
    }.get((level, level + 1), {})

    promotion_gates = {}
    if cfg:
        edge = sortino_d - sortino_as
        promotion_gates = {
            "sortino_edge":  {"pass": edge > cfg["sortino_edge"],   "value": round(edge, 3),         "threshold": cfg["sortino_edge"]},
            "hit_rate":      {"pass": hit_rate >= cfg["hit_rate"],  "value": round(hit_rate, 3),     "threshold": cfg["hit_rate"]},
            "profit_factor": {"pass": profit_factor > cfg["profit_factor"], "value": round(profit_factor, 3), "threshold": cfg["profit_factor"]},
            "min_decisions": {"pass": len(evaluated) >= cfg["min_decisions"], "value": len(evaluated), "threshold": cfg["min_decisions"]},
            "fade_rate":     {"pass": fade_rate <= 0.30,            "value": round(fade_rate, 3),    "threshold": 0.30},
            "regime_gate":   {"pass": True,                         "value": "not yet evaluated"},
            "cooldown":      {"pass": not state.get("in_cooldown"), "value": "active" if state.get("in_cooldown") else "clear"},
        }

    # ── Cooldown state ──
    cooldown_until = state.get("cooldown_until")
    cooldown_days_remaining = 0
    if cooldown_until:
        delta = (date.fromisoformat(cooldown_until) - today).days
        cooldown_days_remaining = max(0, delta)

    days_stuck = state.get("days_stuck", 0)

    feedback = {
        "as_of":                  today.isoformat(),
        "level":                  level,
        "title":                  state.get("title", "Shadow"),
        "ai_weight":              state.get("ai_weight", 0.0),
        "days_at_level":          state.get("days_at_level", 0),
        "in_cooldown":            state.get("in_cooldown", False),
        "cooldown_days_remaining": cooldown_days_remaining,
        "days_stuck":             days_stuck,
        "stuck_alert":            days_stuck >= 63,
        "sortino_21d_d":          round(sortino_d, 3),
        "sortino_21d_astar":      round(sortino_as, 3),
        "sortino_edge":           round(sortino_d - sortino_as, 3),
        "sortino_n_days":         n_days,
        "hit_rate_21d":           round(hit_rate, 3),
        "profit_factor":          round(profit_factor, 3),
        "fade_rate":              round(fade_rate, 3),
        "override_win_rate":      round(hit_rate, 3),
        "amplify_avg_alpha_10d":  round(amp_avg, 6),
        "amplify_n":              amp_n,
        "amplify_confidence":     _confidence(amp_n),
        "reduce_avg_alpha_10d":   round(red_avg, 6),
        "reduce_n":               red_n,
        "reduce_ban_active":      (red_n >= 5 and red_avg < 0),
        "new_position_avg_alpha_10d": round(new_avg, 6),
        "new_position_n":         new_n,
        "n_decisions_evaluated":  len(evaluated),
        "n_decisions_pending":    len(pending),
        "last_5_decisions":       scored[-5:],
        "best_call_10d":          {"symbol": best["symbol"], "type": best["type"], "alpha": best["outcome_10d"], "n_basis": len(evaluated)} if best else None,
        "worst_call_10d":         {"symbol": worst["symbol"], "type": worst["type"], "alpha": worst["outcome_10d"], "n_basis": len(evaluated)} if worst else None,
        "promotion_gates":        promotion_gates,
        "phase1_accuracy":        {"regime_accuracy_rate": None, "sleeve_prior_value": None},
    }

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(feedback, indent=2))
    log.info("[PerfFeedback] Feedback written: Level %d, edge %.3f, n_eval=%d",
             level, feedback["sortino_edge"], len(evaluated))

    if feedback["stuck_alert"]:
        print(f"[AIPMAuthority] WARNING: AI PM at Level {level} for 63+ days without promoting — review promotion gates.")

    return feedback
```

- [ ] **Step 4.4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_perf_feedback.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add ascent/strategy/ai_pm_perf_feedback.py tests/test_ai_pm_perf_feedback.py
git commit -m "feat: add ai_pm_perf_feedback.py — daily Python learning brief, zero LLM cost"
```

---

## Task 5: Update run_all_agents.py — daily integration

**Files:**
- Modify: `run_all_agents.py`

Read the current file before editing:

- [ ] **Step 5.1: Add Haiku daily view function**

Find the section near line 900 where `_ai_prethesis` is initialized. Add a new function `_run_daily_haiku_view()` just before the AI PM section:

```python
def _run_daily_haiku_view(positions: list, feedback: dict) -> None:
    """Lightweight Haiku daily conviction update on non-rebalance days. ~$0.005/day."""
    try:
        from ascent.llm.client import HAIKU_MODEL
        import anthropic
        client = anthropic.Anthropic()

        held_summary = ", ".join(f"{p['symbol']}({p.get('weight',0):.1%})" for p in positions[:10])
        level  = feedback.get("level", 0)
        worst  = feedback.get("worst_call_10d") or {}

        prompt = (
            f"SYSTEM CONTEXT: Today {date.today().isoformat()}. "
            f"Current AI PM level: {level}. "
            f"Worst recent call: {worst.get('symbol','none')} ({worst.get('alpha',0):+.1%} over 10d).\n\n"
            f"Held positions: {held_summary}\n\n"
            "In 2-3 sentences: which held name changed most today and why? "
            "State your directional conviction (bullish/bearish/neutral) on each held name. "
            "Cite only information from today's price moves — no outside knowledge."
        )
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        view_text = resp.content[0].text if resp.content else ""

        _DAILY_VIEWS_LOG = Path("logs/ai_pm_daily_views.jsonl")
        _DAILY_VIEWS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DAILY_VIEWS_LOG, "a") as f:
            f.write(json.dumps({
                "date":  date.today().isoformat(),
                "level": level,
                "view":  view_text,
            }) + "\n")
        print(f"[Runner] Daily AI PM view logged ({len(view_text)} chars, Haiku)")
    except Exception as e:
        print(f"[Runner] Daily AI PM view skipped: {e}")
```

- [ ] **Step 5.2: Add quant Track A★ snapshot call**

In the rebalance day path, BEFORE Phase 1 runs (before `_ai_prethesis = run_ai_pm_prethesis()`), add:

```python
# Track A★: snapshot quant weights BEFORE any Phase 1 influence
from ascent.monitoring.ai_pm_counterfactual import snapshot_quant_star
_quant_star_weights = dict(merged_weights)  # pure quant, no Phase 1 yet
snapshot_quant_star(today, _quant_star_weights)
```

- [ ] **Step 5.3: Add quant Track A snapshot and AI PM Track D snapshot**

After `merged_weights = authority_blend(ai_pm_result.portfolio, merged_weights)` (the blend call), add:

```python
# Track A: quant after Phase 1 sleeve priors, before Phase 2 blend
from ascent.monitoring.ai_pm_counterfactual import snapshot_quant, snapshot_ai_pm
snapshot_quant(today, _quant_pre_blend)  # capture merged_weights before blend
# Track D: pure AI PM portfolio (diagnostic)
if not ai_pm_result.fallback and ai_pm_result.portfolio:
    snapshot_ai_pm(today, ai_pm_result.portfolio)
```

Note: you need to capture `_quant_pre_blend = dict(merged_weights)` BEFORE the blend call.

- [ ] **Step 5.4: Write decision log entry on rebalance days**

After the blend, write the decision log. Find where `format_thesis(...)` is called and add after it:

```python
# Decision log
_DECISION_LOG = Path("logs/ai_pm_decision_log.jsonl")
_DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
with open(_DECISION_LOG, "a") as f:
    _overrides = ai_pm_result.thesis.get("quant_overrides", []) if ai_pm_result.thesis else []
    f.write(json.dumps({
        "date":                  today.isoformat(),
        "level":                 get_authority_state().get("level", 0),
        "title":                 get_authority_state().get("title", "Shadow"),
        "ai_weight":             get_authority_state().get("ai_weight", 0.0),
        "phase2_model":          _phase2_model_used,  # set this variable when Phase 2 runs
        "perf_feedback_injected": _perf_feedback_injected,
        "quant_proposed":        _quant_pre_blend,
        "ai_pm_proposed":        ai_pm_result.portfolio,
        "overrides_applied":     _overrides,
        "final_blended":         merged_weights,
        "thesis_summary":        str(ai_pm_result.thesis.get("market_view", ""))[:200] if ai_pm_result.thesis else "",
    }) + "\n")
```

- [ ] **Step 5.5: Add daily counterfactual scoring and feedback in `_log_holdings()`**

Find `_log_holdings()`. After the Alpaca equity fetch and SPY computation, add:

```python
# Counterfactual daily scoring
try:
    from ascent.monitoring.ai_pm_counterfactual import score_daily, load_snapshots, print_cumulative_report
    _as_w, _a_w, _d_w = load_snapshots()
    _prices_for_cf = {}
    if _as_w:
        syms = list(set(_as_w) | set(_a_w or {}) | set(_d_w or {}))
        try:
            import yfinance as yf
            _raw = yf.download(syms, period="5d", auto_adjust=True, progress=False)
            if not _raw.empty and len(_raw) >= 2:
                _cls = _raw["Close"] if hasattr(_raw.columns, '__len__') and "Close" in _raw else _raw
                for sym in syms:
                    if sym in _cls.columns:
                        _prices_for_cf[sym] = {"prev": float(_cls[sym].iloc[-2]), "curr": float(_cls[sym].iloc[-1])}
        except Exception:
            pass
    score_daily(
        run_date=today,
        quant_star_weights=_as_w,
        quant_weights=_a_w,
        ai_pm_weights=_d_w,
        track_b_return=day_ret,
        spy_return=spy_ret,
        prices=_prices_for_cf,
    )
    print_cumulative_report()
except Exception as _cf_e:
    print(f"[Runner] Counterfactual scoring skipped: {_cf_e}")

# Daily learning brief
try:
    from ascent.strategy.ai_pm_perf_feedback import compute_feedback
    _fb = compute_feedback()
except Exception as _fb_e:
    print(f"[Runner] Perf feedback skipped: {_fb_e}")
```

- [ ] **Step 5.6: Wire Haiku daily view and update_authority on non-rebalance days**

In the non-rebalance path (currently `[Runner] AI PM skipped`), replace that print with:

```python
# Non-rebalance: Haiku daily view + update authority
try:
    _fb_data = json.loads(Path("data_cache/ai_pm_perf_feedback.json").read_text()) if Path("data_cache/ai_pm_perf_feedback.json").exists() else {}
    _run_daily_haiku_view(positions=_current_positions, feedback=_fb_data)
except Exception as _dv_e:
    print(f"[Runner] Daily view skipped: {_dv_e}")
```

And ensure `update_authority()` is called on both rebalance AND non-rebalance days. The current call is at line ~1236. Move it to always run (not gated by `is_rebalance_day`). Pass Track D and Track A★ daily returns from the counterfactual log's latest entry:

```python
# Always update authority with today's Track D vs Track A★
try:
    from ascent.monitoring.ai_pm_counterfactual import load_daily_records
    _cf_today = [r for r in load_daily_records() if r.get("date") == today.isoformat()]
    if _cf_today:
        _d_ret  = _cf_today[-1].get("track_d_return", 0.0)
        _as_ret = _cf_today[-1].get("track_astar_return", 0.0)
    else:
        _d_ret, _as_ret = 0.0, 0.0
    _fb_data = json.loads(Path("data_cache/ai_pm_perf_feedback.json").read_text()) if Path("data_cache/ai_pm_perf_feedback.json").exists() else {}
    update_authority(
        track_d_return=_d_ret,
        track_astar_return=_as_ret,
        n_decisions_evaluated=_fb_data.get("n_decisions_evaluated", 0),
        hit_rate=_fb_data.get("hit_rate_21d"),
        profit_factor=_fb_data.get("profit_factor"),
        fade_rate=_fb_data.get("fade_rate"),
    )
except Exception as _ua_e:
    print(f"[Runner] Authority update skipped: {_ua_e}")
```

- [ ] **Step 5.7: Add smart Opus trigger**

In the Phase 2 call, determine model before calling `run_ai_pm()`:

```python
from ascent.regime.engine import get_latest_regime
_current_regime = get_latest_regime() if hasattr(get_latest_regime, '__call__') else "calm_bull"
_last_regime    = _quant_star_weights  # placeholder — read from regime_signal.json

_use_opus = any([
    str(_current_regime).lower() == "crisis",
    # regime change: compare to last stored regime in earned_authority.json
    get_authority_state().get("last_regime", "") != str(_current_regime),
    len((_ai_prethesis.high_conviction_names if _ai_prethesis else [])) >= 4,
    get_authority_state().get("in_cooldown") is False and get_authority_state().get("days_at_level", 0) == 0,
])
_phase2_model_used = "claude-opus-4-6" if _use_opus else "claude-sonnet-4-6"
```

Pass `_phase2_model_used` to `run_ai_pm(model_override=_phase2_model_used)` — you'll add `model_override` param to that function in Task 6.

- [ ] **Step 5.8: Verify pipeline runs clean**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "import run_all_agents; print('imports OK')"
```
Expected: imports OK with no errors.

- [ ] **Step 5.9: Commit**

```bash
git add run_all_agents.py
git commit -m "feat: wire AI PM daily run, Track A★/D snapshots, counterfactual scoring, decision log"
```

---

## Task 6: Update agents/ai_pm_agent.py — sector thesis + Sharpe objective + anti-hallucination

**Files:**
- Modify: `agents/ai_pm_agent.py`

- [ ] **Step 6.1: Add Sharpe objective + temporal context to Phase 1 prompt**

Find `_SYSTEM_PROMPT` or the Phase 1 system prompt constant. Add at the very top of the prompt (before existing content):

```python
_SHARPE_OBJECTIVE_HEADER = """
OBJECTIVE: Sharpe ratio, not raw return.
For every position you propose, state:
  - Expected 3-month return (with basis in cited data)
  - Expected volatility (high/medium/low with reason)
  - What would make you wrong (one specific falsifiable condition)

A 15% position in a volatile name hurts Sharpe more than 8% in a stable name.
When in doubt, choose the lower-volatility expression of the same thesis.
"""

def _build_temporal_context(feedback: dict | None = None) -> str:
    from datetime import date
    import pandas as pd
    regime = "unknown"
    try:
        import json
        rp = Path("dashboard/regime_signal.json")
        if rp.exists():
            regime = json.loads(rp.read_text()).get("label", "unknown")
    except Exception:
        pass

    worst = ""
    if feedback:
        wc = feedback.get("worst_call_10d")
        if wc and wc.get("symbol"):
            worst = f"\nYour worst recent call: {wc['symbol']} ({wc.get('alpha', 0):+.1%} over 10d)"

    cutoff = (date.today() - __import__('datetime').timedelta(days=45)).isoformat()
    return (
        f"SYSTEM CONTEXT (authoritative — do not contradict):\n"
        f"Today: {date.today().isoformat()}\n"
        f"Current regime: {regime}\n"
        f"Data freshness cutoff: {cutoff} (do not cite anything older as current)\n"
        f"Your last rebalance: {_last_rebalance_date()}{worst}\n"
    )
```

- [ ] **Step 6.2: Add required sector_thesis field to Phase 1 output schema**

Find where `propose_prethesis` tool is defined (around line 407). Add `sector_thesis` as a required top-level field in the tool's input schema:

```python
{
    "name": "sector_thesis",
    "type": "array",
    "description": "REQUIRED before stock selection. Sector-level over/underweight calls with sources.",
    "items": {
        "type": "object",
        "required": ["sector", "view", "conviction", "reason", "source", "data_date"],
        "properties": {
            "sector":          {"type": "string"},
            "view":            {"type": "string", "enum": ["overweight", "underweight", "neutral"]},
            "conviction":      {"type": "string", "enum": ["high", "medium", "low"]},
            "reason":          {"type": "string"},
            "avoid_subsectors":{"type": "array", "items": {"type": "string"}},
            "prefer_subsectors":{"type": "array", "items": {"type": "string"}},
            "source":          {"type": "string"},
            "data_date":       {"type": "string"},
        }
    }
}
```

Also add to `conviction_reasons` items a required `source` and `data_date` field so the source tagging constraint is enforced.

- [ ] **Step 6.3: Add Phase 1 → Phase 2 context strip**

Find where `_ai_prethesis` data is passed to `run_ai_pm`. Add a helper that strips freeform prose:

```python
def _strip_prethesis_for_phase2(prethesis) -> dict:
    """Only pass structured, sourced fields to Phase 2. No freeform prose."""
    if prethesis is None:
        return {}
    return {
        "high_conviction_names":  getattr(prethesis, "high_conviction_names", []),
        "sector_thesis":          getattr(prethesis, "sector_thesis", []),
        "conviction_reasons":     [
            r for r in getattr(prethesis, "conviction_reasons", [])
            if r.get("source") and r.get("data_date")  # sourced only
        ],
        "regime_assessment":      getattr(prethesis, "regime_assessment", {}),
        "causal_mechanisms":      getattr(prethesis, "causal_mechanisms", []),
        # Explicitly omit: market_character prose, sleeve_weight_prior freeform text
    }
```

- [ ] **Step 6.4: Add recency gate parser**

Add a function that strips stale claims before Phase 2 receives them:

```python
from datetime import date, timedelta

_RECENCY_THRESHOLDS = {
    "price":    5,
    "earnings": 45,
    "filings":  45,
    "analyst":  30,
    "default":  45,
}

def _apply_recency_gate(conviction_reasons: list) -> tuple[list, list]:
    """Strip claims older than threshold. Returns (valid_claims, stale_claims)."""
    today = date.today()
    valid, stale = [], []
    for claim in conviction_reasons:
        data_date_str = claim.get("data_date")
        if not data_date_str:
            stale.append({**claim, "strip_reason": "missing data_date"})
            continue
        try:
            data_date = date.fromisoformat(data_date_str)
        except ValueError:
            stale.append({**claim, "strip_reason": "invalid data_date format"})
            continue
        source = claim.get("source", "").lower()
        for key, days in _RECENCY_THRESHOLDS.items():
            if key in source:
                threshold = days
                break
        else:
            threshold = _RECENCY_THRESHOLDS["default"]
        if (today - data_date).days > threshold:
            stale.append({**claim, "strip_reason": f"stale: {(today - data_date).days}d > {threshold}d threshold"})
        else:
            valid.append(claim)
    return valid, stale
```

- [ ] **Step 6.5: Add numeric cross-reference check (post-Phase 2)**

Add after Phase 2 completes, before guardrail processing:

```python
def _check_numeric_claims(thesis: dict, hallucination_log: Path) -> dict:
    """Cross-check numeric claims against data cache. Log and reduce conviction on mismatches."""
    import re
    summary = thesis.get("market_view", "") + " " + str(thesis.get("quant_overrides", ""))
    # Simple regex for revenue/growth claims: "STRL revenue +23%"
    pattern = re.compile(r'([A-Z]{2,5})\s+\w+\s+([+-]?\d+(?:\.\d+)?%)')
    incidents = []
    for match in pattern.finditer(summary):
        sym, claimed_pct_str = match.group(1), match.group(2)
        # Check against fundamentals cache if available
        try:
            import pandas as pd
            fp = Path(f"data_cache/fundamentals_{sym}.parquet")
            if fp.exists():
                df = pd.read_parquet(fp)
                if "revenue_growth" in df.columns:
                    actual = float(df["revenue_growth"].iloc[-1])
                    claimed = float(claimed_pct_str.replace("%", "")) / 100
                    if abs(claimed - actual) / (abs(actual) + 1e-6) > 0.15:
                        incidents.append({"symbol": sym, "claimed": claimed, "actual": actual})
        except Exception:
            pass

    if incidents:
        hallucination_log.parent.mkdir(parents=True, exist_ok=True)
        with open(hallucination_log, "a") as f:
            for inc in incidents:
                f.write(json.dumps({
                    "date": date.today().isoformat(),
                    "symbol": inc["symbol"],
                    "claimed": inc["claimed"],
                    "actual": inc["actual"],
                }) + "\n")

    return thesis  # thesis is modified in-place conviction levels elsewhere
```

- [ ] **Step 6.6: Add model_override parameter to run_ai_pm**

Find `def run_ai_pm(` and add `model_override: str | None = None` parameter. Inside the function, where the Opus/Sonnet model is selected for Phase 2, use `model_override` if provided:

```python
def run_ai_pm(
    quant_outputs=None,
    prethesis=None,
    model_override: str | None = None,
    **kwargs,
):
    # ... existing code ...
    from ascent.llm.client import DEFAULT_MODEL, SONNET_MODEL
    _phase2_model = model_override or DEFAULT_MODEL
    # pass _phase2_model to tool_completion call
```

- [ ] **Step 6.7: Verify ai_pm_agent.py parses cleanly**

```bash
.venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); print('OK')"
```

- [ ] **Step 6.8: Run existing AI PM tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | tail -10
```
Expected: existing tests pass.

- [ ] **Step 6.9: Commit**

```bash
git add agents/ai_pm_agent.py
git commit -m "feat: ai_pm_agent — sector thesis schema, Sharpe objective, source tagging, Phase1→2 strip, Opus trigger"
```

---

## Task 7: Update generate_performance_page.py — dashboard sections

**Files:**
- Modify: `scripts/generate_performance_page.py`

- [ ] **Step 7.1: Add new data loaders**

Find the comment `# ── Extra local data loaders` (around line 289). After the existing loaders, add:

```python
def load_counterfactual() -> list[dict]:
    """Load logs/counterfactual_daily.jsonl — Track A★/A/B/C/D daily returns."""
    path = Path("logs/counterfactual_daily.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def load_perf_feedback() -> dict:
    """Load data_cache/ai_pm_perf_feedback.json — gates, metrics, confidence."""
    path = Path("data_cache/ai_pm_perf_feedback.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_ai_pm_decisions() -> list[dict]:
    """Load logs/ai_pm_decision_log.jsonl — per-rebalance override records."""
    path = Path("logs/ai_pm_decision_log.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return sorted(rows, key=lambda x: x.get("date", ""))
```

- [ ] **Step 7.2: Rewrite `_earned_authority_html` for 5-level ladder**

Replace the existing `_earned_authority_html(auth)` function with:

```python
def _earned_authority_html(auth: dict, feedback: dict) -> str:
    if not auth:
        return '<p class="empty">data_cache/earned_authority.json not found</p>'

    level     = auth.get("level", auth.get("phase", 0))
    title     = auth.get("title", ["Shadow","Analyst","Associate","Manager","Director","CEO"][min(level,5)])
    ai_weight = auth.get("ai_weight", 0.0)
    days      = auth.get("days_at_level", 0)
    cooldown  = auth.get("in_cooldown", False)
    stuck     = feedback.get("stuck_alert", False)
    edge      = feedback.get("sortino_edge", 0.0)

    titles = ["Shadow","Analyst","Associate","Manager","Director","CEO"]
    weights = ["0%","5%","15%","30%","50%","75%"]
    windows = ["-","21d","21d","42d","63d","-"]

    steps_html = ""
    for i, (t, w) in enumerate(zip(titles, weights)):
        cls = "ph-active" if i == level else ("ph-done" if i < level else "ph-future")
        steps_html += f'<div class="ph-step {cls}"><div class="ph-dot"></div><div class="ph-name">{t}<br><small>{w}</small></div></div>'
        if i < 5:
            steps_html += f'<div class="ph-line {"ph-done" if i < level else ""}"></div>'

    window = windows[min(level, 5)]
    pct_done = min(100, round(days / 21 * 100)) if level < 3 else min(100, round(days / 42 * 100))

    alert_html = ""
    if stuck:
        alert_html = '<div class="stuck-alert">⚠ AI PM stuck at this level 63+ days — review promotion gates</div>'
    if cooldown:
        cd_rem = feedback.get("cooldown_days_remaining", 0)
        alert_html += f'<div class="cooldown-banner">❄ Cooldown active — {cd_rem} trading days remaining</div>'

    edge_color = "#3fb950" if edge >= 0 else "#f85149"
    edge_sign  = "+" if edge >= 0 else ""

    return f"""
{alert_html}
<div class="level-badge">{title} — {weights[min(level,5)]} authority — Day {days} of {window}</div>
<div class="phase-steps">{steps_html}</div>
<div class="phase-progress-bar"><div class="phase-fill" style="width:{pct_done}%"></div></div>
<div class="auth-stats">
  <div class="auth-stat"><div class="as-val" style="color:{edge_color}">{edge_sign}{edge:.3f}</div><div class="as-lbl">Sortino edge</div></div>
  <div class="auth-stat"><div class="as-val">{feedback.get('hit_rate_21d',0):.0%}</div><div class="as-lbl">Hit rate</div></div>
  <div class="auth-stat"><div class="as-val">{feedback.get('profit_factor',0):.2f}x</div><div class="as-lbl">Profit factor</div></div>
  <div class="auth-stat"><div class="as-val">{feedback.get('n_decisions_evaluated',0)}</div><div class="as-lbl">Decisions scored</div></div>
</div>"""
```

- [ ] **Step 7.3: Add `_promotion_gates_html`**

Add after `_earned_authority_html`:

```python
def _promotion_gates_html(feedback: dict) -> str:
    if not feedback or "promotion_gates" not in feedback:
        return '<p class="empty">No promotion gate data yet — starts after first rebalance.</p>'
    gates = feedback["promotion_gates"]
    rows = ""
    labels = {
        "sortino_edge": "Sortino edge",
        "hit_rate": "Hit rate",
        "profit_factor": "Profit factor",
        "min_decisions": "Min decisions",
        "fade_rate": "Fade rate",
        "regime_gate": "Regime diversity",
        "cooldown": "Cooldown clear",
    }
    for key, label in labels.items():
        g = gates.get(key, {})
        passed = g.get("pass", False)
        val    = g.get("value", "—")
        thr    = g.get("threshold", "")
        icon   = "✓" if passed else "✗"
        color  = "#3fb950" if passed else "#f85149"
        thr_str = f" / need {thr}" if thr else ""
        rows += f'<div class="gate-row"><span style="color:{color}">{icon}</span> <span class="gate-label">{label}</span><span class="gate-val">{val}{thr_str}</span></div>'
    return f'<div class="gates-container">{rows}</div>'
```

- [ ] **Step 7.4: Add `_counterfactual_chart_html`**

```python
def _counterfactual_chart_html(cfdata: list[dict]) -> str:
    if not cfdata:
        return '<p class="empty">No AI PM data yet — starts after next rebalance.</p>'

    def cumulative(key):
        v, vals = 1.0, []
        for r in cfdata:
            v *= (1 + r.get(key, 0.0))
            vals.append(round((v - 1) * 100, 3))
        return vals

    dates  = [r["date"] for r in cfdata]
    astar  = cumulative("track_astar_return")
    actual = cumulative("track_b_return")
    spy    = cumulative("track_c_return")
    ai_pm  = cumulative("track_d_return")

    d_val  = ai_pm[-1]  if ai_pm  else 0
    as_val = astar[-1]  if astar  else 0
    b_val  = actual[-1] if actual else 0
    signal_quality = round(d_val - as_val, 2)
    actual_impact  = round(b_val - as_val, 2)
    sq_color = "#3fb950" if signal_quality >= 0 else "#f85149"

    return f"""
<div class="cf-summary">
  AI signal quality (D−A★): <span style="color:{sq_color}">{'+' if signal_quality>=0 else ''}{signal_quality:.2f}pp</span> since live.
  Actual portfolio impact at current weight: {'+' if actual_impact>=0 else ''}{actual_impact:.2f}pp measurable.
</div>
<canvas id="cfChart" height="120"></canvas>
<script>
(function(){{
  var ctx = document.getElementById('cfChart');
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: {json.dumps(dates)},
      datasets: [
        {{label:'Pure Quant (A★)', data:{json.dumps(astar)},  borderColor:'#8b949e', borderDash:[4,2], pointRadius:0, fill:false}},
        {{label:'Actual (B)',       data:{json.dumps(actual)}, borderColor:'#3fb950', pointRadius:0,    fill:false}},
        {{label:'SPY (C)',          data:{json.dumps(spy)},    borderColor:'#58a6ff', borderDash:[4,2], pointRadius:0, fill:false}},
        {{label:'Pure AI PM (D)',   data:{json.dumps(ai_pm)},  borderColor:'#d29922', pointRadius:0,    fill:false}},
      ]
    }},
    options: {{responsive:true, plugins:{{legend:{{position:'bottom'}}}}, scales:{{y:{{ticks:{{callback:function(v){{return v.toFixed(1)+'%'}}}}}}}}}}
  }});
}})();
</script>"""
```

- [ ] **Step 7.5: Add `_override_scorecard_html`**

```python
def _override_scorecard_html(decisions: list[dict], feedback: dict) -> str:
    if not decisions:
        return '<p class="empty">No override decisions yet.</p>'
    # Flatten all overrides with outcomes from feedback last_5
    last5 = feedback.get("last_5_decisions", [])
    if not last5:
        return '<p class="empty">No scored overrides yet — outcomes available after 10 trading days.</p>'

    rows = ""
    for dec in last5[-5:]:
        sym   = dec.get("symbol", "?")
        ov_t  = dec.get("type", "?")
        ai_w  = dec.get("ai_w", 0)
        qt_w  = dec.get("quant_w", 0)
        r5    = dec.get("outcome_5d")
        r10   = dec.get("outcome_10d")
        r21   = dec.get("outcome_21d")
        verd  = dec.get("verdict", "pending")
        vcolor = {"win":"#3fb950","miss":"#f85149","fade":"#d29922","early":"#58a6ff"}.get(verd, "#8b949e")
        fmt = lambda v: f"{v:+.2%}" if v is not None else "—"
        rows += (f'<tr><td>{dec.get("date","")[:10]}</td><td><b>{sym}</b></td>'
                 f'<td>{ov_t}</td><td>{ai_w:.1%}</td><td>{qt_w:.1%}</td>'
                 f'<td>{fmt(r5)}</td><td>{fmt(r10)}</td><td>{fmt(r21)}</td>'
                 f'<td style="color:{vcolor}">{verd.upper()}</td></tr>')

    win_rate     = feedback.get("hit_rate_21d", 0)
    avg_alpha    = feedback.get("amplify_avg_alpha_10d", 0)
    fade_rate    = feedback.get("fade_rate", 0)

    return f"""
<div class="scorecard-summary">
  Win rate: <b>{win_rate:.0%}</b> · Avg incremental α (10d): <b>{avg_alpha:+.3%}</b> · Fade rate: <b>{fade_rate:.0%}</b>
</div>
<table class="scorecard-table">
  <thead><tr><th>Date</th><th>Symbol</th><th>Type</th><th>AI%</th><th>Quant%</th><th>+5d</th><th>+10d</th><th>+21d</th><th>Result</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""
```

- [ ] **Step 7.6: Wire everything into `build_html()`**

Find `build_html(` function (around line 617). Add these loaders near the top of the function and wire the HTML sections. Find where `authority_html = _earned_authority_html(auth)` is called and update:

```python
# New loaders
cfdata    = load_counterfactual()
feedback  = load_perf_feedback()
decisions = load_ai_pm_decisions()

# Updated + new sections
authority_html  = _earned_authority_html(auth, feedback)   # add feedback arg
gates_html      = _promotion_gates_html(feedback)
cf_chart_html   = _counterfactual_chart_html(cfdata)
scorecard_html  = _override_scorecard_html(decisions, feedback)
```

Then find the AI section HTML template (around line 908, inside the large HTML string) and add the new sections after `authority_html`:

```python
# In the f-string HTML, after {authority_html}:
{gates_html}
{cf_chart_html}
{scorecard_html}
```

- [ ] **Step 7.7: Verify dashboard generates without errors**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python scripts/generate_performance_page.py 2>&1 | tail -5
```
Expected: `[Dashboard] Written to docs/index.html` with no exceptions.

- [ ] **Step 7.8: Commit**

```bash
git add scripts/generate_performance_page.py
git commit -m "feat: dashboard — 5-level career ladder, promotion gates, counterfactual chart, override scorecard"
```

---

## Task 8: Bootstrap + full run verification

- [ ] **Step 8.1: Set Level 1 in earned_authority.json**

```bash
cat > data_cache/earned_authority.json << 'EOF'
{
  "level": 1,
  "title": "Analyst",
  "ai_weight": 0.05,
  "phase": 1,
  "level_start_date": "2026-06-04",
  "days_at_level": 0,
  "days_stuck": 0,
  "in_cooldown": false,
  "cooldown_until": null,
  "auto_revert_count": 0,
  "last_updated": "2026-06-03",
  "track_d_returns": [],
  "track_astar_returns": [],
  "ai_returns_21d": [],
  "quant_returns_21d": [],
  "disable_sleeve_priors": false,
  "phase_start_date": "2026-06-04"
}
EOF
```

- [ ] **Step 8.2: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -15
```
Expected: all tests pass (752+ passing, 0 failures).

- [ ] **Step 8.3: Verify new test files all pass**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_authority.py tests/test_ai_pm_counterfactual.py tests/test_ai_pm_perf_feedback.py -v 2>&1 | tail -20
```
Expected: all 13 new tests PASS.

- [ ] **Step 8.4: Verify module imports**

```bash
.venv/bin/python -c "
from ascent.strategy.earned_authority import get_state, blend, update_authority
from ascent.strategy.ai_pm_guardrails import apply_guardrails
from ascent.monitoring.ai_pm_counterfactual import snapshot_quant_star, score_daily
from ascent.strategy.ai_pm_perf_feedback import compute_feedback
s = get_state()
print(f'Level: {s[\"level\"]} ({s[\"title\"]}), ai_weight: {s[\"ai_weight\"]:.0%}')
"
```
Expected: `Level: 1 (Analyst), ai_weight: 5%`

- [ ] **Step 8.5: Run pipeline dry-run (no live API calls)**

```bash
.venv/bin/python -c "
import json
from ascent.strategy.ai_pm_perf_feedback import compute_feedback
fb = compute_feedback()
print('Feedback written:', json.dumps({k: fb[k] for k in ['level','title','ai_weight','n_decisions_evaluated']}, indent=2))
"
```
Expected: feedback file written with level=1, title=Analyst.

- [ ] **Step 8.6: Final commit**

```bash
git add data_cache/earned_authority.json
git commit -m "bootstrap: set AI PM to Level 1 (Analyst, 5% authority) — Day 1 evaluation begins"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|---|---|
| 5-level career ladder, Sortino promotion | Task 1 |
| Demotion: catastrophic/hard/soft on Track D vs A★ | Task 1 |
| Cooldown 5 days after demotion | Task 1 |
| Stuck alert at 63 days | Task 1, 4 |
| Level guardrails (weight change, types, overrides) | Task 2 |
| Tracking error cap per level | Task 2 |
| Override correlation check | Task 2 |
| Conviction inflation cap | Task 2 |
| Post-blend portfolio validation | Task 2 |
| Track A★ / Track A / Track D snapshots | Task 3 |
| Idempotent snapshot writes | Task 3 |
| Daily Track A★/A/B/C/D scoring | Task 3 |
| Incremental alpha (delta weight × return) | Task 4 |
| Fade detection, early detection | Task 4 |
| Profit factor computation | Task 4 |
| All 7 promotion gates with n= fields | Task 4 |
| Confidence labels (low/medium/high) | Task 4 |
| Outcome windows 5d/10d/21d/63d/126d | Task 4 |
| Orphaned decisions scored as 0 | Task 4 |
| REDUCE ban requires n≥5 | Task 4 |
| Haiku daily view (non-rebalance) | Task 5 |
| Quant A★/A snapshot calls | Task 5 |
| Decision log write | Task 5 |
| update_authority called daily | Task 5 |
| Counterfactual daily scoring after _log_holdings | Task 5 |
| Smart Opus trigger (crisis always, + 4 conditions) | Task 5 |
| Sector thesis required field in Phase 1 | Task 6 |
| Sharpe objective in every prompt | Task 6 |
| Temporal context header injected | Task 6 |
| Source tagging + recency gate | Task 6 |
| Phase 1 → 2 context strip | Task 6 |
| Numeric cross-reference check | Task 6 |
| model_override for Opus trigger | Task 6 |
| Dashboard 5-level ladder | Task 7 |
| Dashboard promotion gates checklist | Task 7 |
| Dashboard Track A★/B/C/D chart | Task 7 |
| Dashboard override scorecard | Task 7 |
| Bootstrap Level 1 | Task 8 |

**Constraints not covered by a task (deferred):**
- Phase 1 accuracy tracking (regime call scoring vs actual 10d regime) — requires 10d look-forward after each rebalance; add to `ai_pm_perf_feedback.py` in a follow-up once there is data.
- `disable_sleeve_priors` flag enforcement in quant pipeline — requires touching `ascent/main.py`; add as follow-up.
- Level 4+ architecture flip (AI PM proposes, quant validates) — requires significant orchestrator changes; implement when AI PM reaches Level 4.
- 126d outcome window — add to `_score_decisions()` in Task 4 code alongside 63d; omitted from tests for brevity but the pattern is identical.
