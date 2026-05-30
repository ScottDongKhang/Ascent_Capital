# ascent/strategy/earned_authority.py
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data_cache" / "earned_authority.json"
SHADOW_RETURNS_PATH = Path(__file__).resolve().parent.parent.parent / "data_cache" / "ai_pm_shadow_returns.jsonl"

ADVANCE_EDGE = 0.05
ADVANCE_WINDOW = 10        # rebalance periods (~5 months); was 21 daily returns (wrong unit)
REVERT_DRAWDOWN_EDGE = 0.05
MIN_WEIGHT = 0.02
HARD_CAP = 0.80
# Phases 0-3: shadow → 25% → 50% → 75%. HARD_CAP is the absolute ceiling.
PHASE_WEIGHTS = [0.0, 0.25, 0.50, 0.75]
# Each return is a ~10-business-day holding-period return. ~26 rebalances/year.
_PERIODS_PER_YEAR = 26


def get_state() -> dict:
    """Load state from JSON. Returns defaults if file missing or corrupt."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as exc:
            log.warning("[EarnedAuthority] Corrupt state file, resetting to defaults: %s", exc)
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
    return 0.0 if stdev == 0 else mean / stdev * (_PERIODS_PER_YEAR ** 0.5)


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

    if state.get("last_updated") == today:
        log.debug("[EarnedAuthority] Already updated today (%s), skipping", today)
        return state

    ai_buf: List[float] = (state.get("ai_returns_21d", []) + [float(ai_daily_return)])[-ADVANCE_WINDOW:]
    qt_buf: List[float] = (state.get("quant_returns_21d", []) + [float(quant_daily_return)])[-ADVANCE_WINDOW:]
    state["ai_returns_21d"] = ai_buf
    state["quant_returns_21d"] = qt_buf
    _log_shadow_return(today, ai_daily_return, quant_daily_return, state["ai_weight"])

    # Auto-revert: AI drawdown more than 5pp worse than quant
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

    # Advance: full 21-day buffer with AI Sharpe edge
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
    """Weight-average over union, drop < MIN_WEIGHT=0.02, renormalize."""
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
        if not ai_portfolio and not quant_portfolio:
            log.error("[EarnedAuthority] blend() called with both portfolios empty")
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
