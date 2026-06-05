"""
tests/test_ai_pm_authority.py
Tests for earned_authority.py (5-level career ladder) and ai_pm_guardrails.py.
"""
import json
import tempfile
import pytest
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


# ── earned_authority tests ────────────────────────────────────────────────────

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
        shadow = Path(tmp) / "shadow.jsonl"
        with patch.object(ea, "STATE_PATH", sp):
            with patch.object(ea, "SHADOW_RETURNS_PATH", shadow):
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
                for _ in range(5):
                    ea.update_authority(0.05, -0.01)
                s = ea.get_state()
    assert s["level"] == 1  # no promotion during cooldown


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


def test_blend_shadow_returns_pure_quant():
    """At ai_weight=0, blend returns quant portfolio."""
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        sp = _write_state(tmp, _default_state(level=0))
        shadow = Path(tmp) / "shadow.jsonl"
        with patch.object(ea, "STATE_PATH", sp):
            with patch.object(ea, "SHADOW_RETURNS_PATH", shadow):
                quant = {"AAPL": 0.50, "MSFT": 0.50}
                ai    = {"GOOG": 0.60, "AMZN": 0.40}
                result = ea.blend(ai, quant)
    assert "AAPL" in result
    assert "GOOG" not in result
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_blend_renormalizes():
    import ascent.strategy.earned_authority as ea
    with tempfile.TemporaryDirectory() as tmp:
        sp = _write_state(tmp, _default_state(level=2))
        shadow = Path(tmp) / "shadow.jsonl"
        with patch.object(ea, "STATE_PATH", sp):
            with patch.object(ea, "SHADOW_RETURNS_PATH", shadow):
                ai    = {f"A{i}": 0.1 for i in range(5)}
                quant = {f"Q{i}": 0.1 for i in range(5)}
                result = ea.blend(ai, quant)
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_legacy_migration_from_phase_to_level():
    """Old state with 'phase' key should migrate to 'level' seamlessly."""
    import ascent.strategy.earned_authority as ea
    old_state = {
        "phase": 1, "ai_weight": 0.25,
        "phase_start_date": "2026-05-01",
        "ai_returns_21d": [], "quant_returns_21d": [],
        "auto_revert_count": 0, "last_updated": "2026-05-01",
    }
    with tempfile.TemporaryDirectory() as tmp:
        sp = _write_state(tmp, old_state)
        with patch.object(ea, "STATE_PATH", sp):
            s = ea.get_state()
    assert "level" in s
    assert s["level"] == 1


# ── guardrail tests ────────────────────────────────────────────────────────────

def test_level1_blocks_reduce():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    quant = {"AAPL": 0.10, "MSFT": 0.10}
    ai_pm = {"AAPL": 0.07, "MSFT": 0.10}  # AAPL is a REDUCE
    alpha_scores = {"AAPL": 0.8, "MSFT": 0.9}
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    # AAPL REDUCE blocked → stays at quant weight
    assert result.get("AAPL", 0.10) >= quant["AAPL"] - 0.001
    assert any("REDUCE" in v or "reduce" in v.lower() or "not allowed" in v.lower() for v in violations)


def test_level1_blocks_amplify_bottom_50pct_alpha():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    quant = {"AAPL": 0.07, "WEAK": 0.05}
    ai_pm = {"AAPL": 0.09, "WEAK": 0.09}  # amplifying WEAK (low alpha)
    alpha_scores = {"AAPL": 0.80, "WEAK": 0.20}  # WEAK below median
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    # WEAK amplification blocked — stays at quant weight
    assert abs(result.get("WEAK", 0.05) - 0.05) < 0.002
    assert any("WEAK" in v for v in violations)


def test_level1_max_weight_change_capped_at_2pp():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    quant = {"AAPL": 0.07}
    ai_pm = {"AAPL": 0.15}  # +8pp, above 2pp cap
    alpha_scores = {"AAPL": 0.90}
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    assert result.get("AAPL", 0.09) <= 0.07 + 0.02 + 1e-5


def test_max_overrides_enforced():
    from ascent.strategy.ai_pm_guardrails import apply_guardrails
    # Level 1 allows max 2 overrides
    quant = {f"S{i}": 0.07 for i in range(5)}
    ai_pm = {f"S{i}": 0.09 for i in range(5)}  # 5 amplifications
    alpha_scores = {f"S{i}": 0.90 for i in range(5)}
    result, violations = apply_guardrails(ai_pm, quant, alpha_scores, level=1)
    n_changed = sum(1 for sym in quant if abs(result.get(sym, quant[sym]) - quant[sym]) > 0.001)
    assert n_changed <= 2


def test_check_conviction_inflation():
    from ascent.strategy.ai_pm_guardrails import check_conviction_inflation
    # 5 names all high = 100% > 40% threshold → downgrade excess
    proposals = {f"S{i}": "high" for i in range(5)}
    result = check_conviction_inflation(proposals)
    high_count = sum(1 for v in result.values() if v == "high")
    assert high_count <= max(1, int(len(proposals) * 0.40) + 1)
