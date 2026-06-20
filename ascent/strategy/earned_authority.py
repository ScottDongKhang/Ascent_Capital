# ascent/strategy/earned_authority.py
from __future__ import annotations
import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data_cache" / "earned_authority.json"
SHADOW_RETURNS_PATH = Path(__file__).resolve().parent.parent.parent / "data_cache" / "ai_pm_shadow_returns.jsonl"

LEVEL_WEIGHTS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]
PHASE_WEIGHTS = LEVEL_WEIGHTS  # legacy alias — old code used 4-phase system
LEVEL_TITLES  = ["Shadow", "Analyst", "Associate", "Manager", "Director", "CEO"]
HARD_CAP      = 0.80
MIN_WEIGHT    = 0.02
_TRADING_DAYS_PER_YEAR = 252

# Promotion config per transition (from_level, to_level)
PROMOTION_CONFIG = {
    (1, 2): {"window": 21, "sortino_edge": 0.20, "hit_rate": 0.52, "profit_factor": 1.2, "min_decisions": 5,  "primary_window": 10},
    (2, 3): {"window": 21, "sortino_edge": 0.30, "hit_rate": 0.55, "profit_factor": 1.3, "min_decisions": 8,  "primary_window": 10},
    (3, 4): {"window": 42, "sortino_edge": 0.40, "hit_rate": 0.55, "profit_factor": 1.3, "min_decisions": 10, "primary_window": 63},
    (4, 5): {"window": 63, "sortino_edge": 0.50, "hit_rate": 0.58, "profit_factor": 1.4, "min_decisions": 15, "primary_window": 63},
}


def _sortino(returns: List[float]) -> float:
    """Sortino ratio annualised. Only penalises downside deviation."""
    if len(returns) < 5:
        return 0.0
    mean = sum(returns) / len(returns)
    neg  = [r for r in returns if r < 0]
    if not neg:
        return mean * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100  # cap at large value
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
    """Load state from JSON. Migrates old phase-based schema automatically."""
    if STATE_PATH.exists():
        try:
            s = json.loads(STATE_PATH.read_text())
            # Migrate old phase-based state to new level-based schema
            if "phase" in s and "level" not in s:
                phase = min(s["phase"], 5)
                s["level"]               = phase
                s["title"]               = LEVEL_TITLES[phase]
                s["level_start_date"]    = s.get("phase_start_date", str(date.today()))
                s.setdefault("days_at_level", 0)
                s.setdefault("days_stuck", 0)
                s.setdefault("in_cooldown", False)
                s.setdefault("cooldown_until", None)
                # Carry over old buffers into new track fields
                s.setdefault("track_d_returns",     s.get("ai_returns_21d", []))
                s.setdefault("track_astar_returns",  s.get("quant_returns_21d", []))
                s.setdefault("disable_sleeve_priors", False)
            # Always migrate empty track buffers from legacy arrays (runs even if level present)
            if not s.get("track_d_returns") and s.get("ai_returns_21d"):
                s["track_d_returns"] = list(s["ai_returns_21d"])
            if not s.get("track_astar_returns") and s.get("quant_returns_21d"):
                s["track_astar_returns"] = list(s["quant_returns_21d"])
            s.setdefault("track_d_returns", [])
            s.setdefault("track_astar_returns", [])
            s.setdefault("in_cooldown", False)
            s.setdefault("cooldown_until", None)
            s.setdefault("days_at_level", 0)
            s.setdefault("days_stuck", 0)
            s.setdefault("disable_sleeve_priors", False)
            return s
        except Exception as exc:
            log.warning("[EarnedAuthority] Corrupt state, resetting: %s", exc)
    return {
        "level": 0, "title": "Shadow", "ai_weight": 0.0,
        "phase": 0,  # legacy compat
        "level_start_date": str(date.today()),
        "phase_start_date": str(date.today()),
        "days_at_level": 0, "days_stuck": 0,
        "in_cooldown": False, "cooldown_until": None,
        "auto_revert_count": 0, "last_updated": str(date.today()),
        "track_d_returns": [], "track_astar_returns": [],
        "ai_returns_21d": [], "quant_returns_21d": [],  # legacy compat
        "disable_sleeve_priors": False,
    }


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def rebuild_buffers_from_counterfactual() -> int:
    """Reconcile the Track D / Track A★ rolling buffers to the counterfactual log —
    the single source of truth for D-vs-A★.

    The buffers drive Sortino-based promotion/demotion, but they drifted: they were
    seeded from a different source (the shadow log, with its own A★ series and a
    duplicated entry) and update_authority only appends on days where BOTH tracks are
    non-null, so the bad seed was never reconciled. This rebuilds both buffers from the
    log's common-window — the (track_d, track_astar) pairs where both are non-null, in
    date order, last 63 — so the buffer always matches what get_cumulative_returns
    reports. Idempotent. Returns the number of common-window observations written."""
    from ascent.monitoring.ai_pm_counterfactual import load_daily_records
    d_buf, as_buf = [], []
    for r in load_daily_records():
        d, a = r.get("track_d_return"), r.get("track_astar_return")
        if d is None or a is None:
            continue
        d_buf.append(float(d))
        as_buf.append(float(a))
    d_buf, as_buf = d_buf[-63:], as_buf[-63:]
    state = get_state()
    state["track_d_returns"]     = d_buf
    state["track_astar_returns"] = as_buf
    state["ai_returns_21d"]      = d_buf[-21:]
    state["quant_returns_21d"]   = as_buf[-21:]
    _save_state(state)
    log.info("[EarnedAuthority] Buffers rebuilt from counterfactual log: %d common-window obs", len(d_buf))
    return len(d_buf)


def update_authority(
    track_d_return: Optional[float],
    track_astar_return: Optional[float],
    n_decisions_evaluated: int = 0,
    hit_rate: Optional[float] = None,
    profit_factor: Optional[float] = None,
    fade_rate: Optional[float] = None,
    regime_gate_pass: bool = True,
) -> dict:
    """Append daily Track D / Track A★ returns. Check demotion then promotion. Returns updated state.
    If either track_d_return or track_astar_return is None, skips buffer append and returns unchanged state."""
    state = get_state()
    today = str(date.today())

    if state.get("last_updated") == today:
        log.debug("[EarnedAuthority] Already updated today (%s), skipping", today)
        return state

    level = state.get("level", 0)

    # Guard: only append to buffers when both returns are real numbers (not None)
    if track_d_return is None or track_astar_return is None:
        log.debug("[EarnedAuthority] Skipping buffer append — Track D or A★ return is None")
        state["last_updated"] = today
        _save_state(state)
        return state

    # Update rolling buffers (keep last 63 days for Level 3+ windows)
    d_buf  = (state.get("track_d_returns", [])     + [float(track_d_return)])[-63:]
    as_buf = (state.get("track_astar_returns", []) + [float(track_astar_return)])[-63:]
    state["track_d_returns"]     = d_buf
    state["track_astar_returns"] = as_buf
    # Legacy compat buffers
    state["ai_returns_21d"]    = d_buf[-21:]
    state["quant_returns_21d"] = as_buf[-21:]

    _log_shadow_return(today, track_d_return, track_astar_return, state.get("ai_weight", 0.0))

    # ── Cooldown check ──────────────────────────────────────────────────────
    cooldown_until = state.get("cooldown_until")
    if cooldown_until and today <= cooldown_until:
        state["days_at_level"] = state.get("days_at_level", 0) + 1
        state["days_stuck"]    = state.get("days_stuck", 0) + 1
        state["in_cooldown"]   = True
        state["last_updated"]  = today
        _save_state(state)
        return state
    elif cooldown_until and today > cooldown_until:
        state["in_cooldown"]    = False
        state["cooldown_until"] = None
        log.info("[EarnedAuthority] Cooldown expired")

    # ── Demotion checks (Track D vs Track A★) ──────────────────────────────
    if level > 0:
        daily_diff = track_d_return - track_astar_return

        # Catastrophic: single day ≥10pp worse (use 0.099 to handle float precision)
        if daily_diff <= -0.099:
            log.warning("[EarnedAuthority] CATASTROPHIC demotion (%.2fpp): Track D %.3f vs A★ %.3f",
                        abs(daily_diff) * 100, track_d_return, track_astar_return)
            _apply_demotion(state, today, target_level=0, cooldown_days=5)
            _save_state(state)
            return state

        # Hard: single day ≥5pp worse
        if daily_diff <= -0.05:
            log.warning("[EarnedAuthority] HARD demotion (%.2fpp worse than A★)", abs(daily_diff) * 100)
            _apply_demotion(state, today, target_level=max(0, level - 1), cooldown_days=5)
            _save_state(state)
            return state

        # Soft: drawdown gap over rolling 21-day window (requires ≥10 days)
        if len(d_buf) >= 10:
            dd_d  = _max_drawdown(d_buf[-21:])
            dd_as = _max_drawdown(as_buf[-21:])
            if dd_d > dd_as + 0.03:
                log.warning("[EarnedAuthority] SOFT demotion: DD gap %.2fpp", (dd_d - dd_as) * 100)
                _apply_demotion(state, today, target_level=max(0, level - 1), cooldown_days=5)
                _save_state(state)
                return state

    # ── Promotion check ─────────────────────────────────────────────────────
    cfg = PROMOTION_CONFIG.get((level, level + 1))
    if cfg and not state.get("in_cooldown"):
        win = cfg["window"]
        if len(d_buf) >= win:
            sortino_d  = _sortino(d_buf[-win:])
            sortino_as = _sortino(as_buf[-win:])
            edge       = sortino_d - sortino_as

            gates = {
                "sortino_edge":  edge > cfg["sortino_edge"],
                "hit_rate":      (hit_rate or 0.0) >= cfg["hit_rate"],
                "profit_factor": (profit_factor or 0.0) > cfg["profit_factor"],
                "min_decisions": n_decisions_evaluated >= cfg["min_decisions"],
                "fade_rate":     (fade_rate or 0.0) <= 0.30,
                "regime_gate":   regime_gate_pass,
            }
            if all(gates.values()):
                new_level = level + 1
                log.info("[EarnedAuthority] PROMOTED Level %d → %d (%s, %.0f%%)",
                         level, new_level, LEVEL_TITLES[new_level], LEVEL_WEIGHTS[new_level] * 100)
                state.update({
                    "level":             new_level,
                    "title":             LEVEL_TITLES[new_level],
                    "ai_weight":         min(LEVEL_WEIGHTS[new_level], HARD_CAP),
                    "phase":             new_level,  # legacy compat
                    "level_start_date":  today,
                    "phase_start_date":  today,
                    "days_at_level":     0,
                    "days_stuck":        0,
                    "track_d_returns":   [],
                    "track_astar_returns": [],
                    "ai_returns_21d":    [],
                    "quant_returns_21d": [],
                })
                state["last_updated"] = today
                _save_state(state)
                return state

    # ── Increment day counters ───────────────────────────────────────────────
    state["days_at_level"] = state.get("days_at_level", 0) + 1
    state["days_stuck"]    = state.get("days_stuck", 0) + 1
    if is_stuck(state):
        log.warning("[AIPMAuthority] WARNING: AI PM at Level %d for 63+ days without promoting", level)

    state["last_updated"] = today
    _save_state(state)
    return state


def _apply_demotion(state: dict, today: str, target_level: int, cooldown_days: int) -> None:
    cooldown_date = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=cooldown_days)).date()
    state.update({
        "level":               target_level,
        "title":               LEVEL_TITLES[target_level],
        "ai_weight":           LEVEL_WEIGHTS[target_level],
        "phase":               target_level,  # legacy compat
        "level_start_date":    today,
        "phase_start_date":    today,
        "days_at_level":       0,
        "days_stuck":          0,
        "in_cooldown":         True,
        "cooldown_until":      str(cooldown_date),
        "auto_revert_count":   state.get("auto_revert_count", 0) + 1,
        "track_d_returns":     [],
        "track_astar_returns": [],
        "ai_returns_21d":      [],
        "quant_returns_21d":   [],
    })
    state["last_updated"] = today


def blend(ai_portfolio: Dict[str, float], quant_portfolio: Dict[str, float]) -> Dict[str, float]:
    """Apply AI PM changes as active-weight budget against quant portfolio.

    ai_weight is the max one-way tracking-error budget (e.g. 0.05 = 5pp).
    Deltas are scaled so gross one-way deviation stays within budget,
    then dust positions (<0.5%) are dropped and weights renormalized.
    """
    if not ai_portfolio:
        return dict(quant_portfolio)

    state = get_state()
    budget = state.get("ai_weight", 0.0)

    if budget <= 0.0:
        return dict(quant_portfolio)

    all_syms = set(ai_portfolio) | set(quant_portfolio)
    deltas = {
        s: ai_portfolio.get(s, 0.0) - quant_portfolio.get(s, 0.0)
        for s in all_syms
    }
    gross = sum(abs(d) for d in deltas.values()) / 2.0  # one-way deviation
    scale = min(1.0, budget / gross) if gross > 0.0 else 0.0

    blended = {
        s: max(0.0, quant_portfolio.get(s, 0.0) + scale * deltas[s])
        for s in all_syms
    }
    # Drop dust after scaling, then renormalize
    DUST_THRESHOLD = 0.005  # 0.5% — lower than old MIN_WEIGHT to allow budgeted new names
    blended = {s: w for s, w in blended.items() if w >= DUST_THRESHOLD}
    total = sum(blended.values())
    if total <= 0:
        return dict(quant_portfolio)
    return {s: w / total for s, w in blended.items()}


def _log_shadow_return(today: str, d_ret: float, as_ret: float, ai_weight: float) -> None:
    try:
        SHADOW_RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_RETURNS_PATH, "a") as f:
            f.write(json.dumps({
                "date": today,
                "ai_return": d_ret,
                "quant_return": as_ret,
                "ai_weight_at_time": ai_weight,
            }) + "\n")
    except Exception as exc:
        log.warning("[EarnedAuthority] Could not log shadow return: %s", exc)
