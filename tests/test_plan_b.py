# tests/test_plan_b.py
def test_em_commodity_cap_enforced():
    """Merged weights must not exceed 20% in EM+commodity+gold combined."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {
        "EEM":  0.12, "VWO":  0.10, "GLD":  0.08, "PDBC": 0.07,
        "AAPL": 0.25, "MSFT": 0.20, "JPM":  0.18,
    }
    capped = _cap_em_commodity(weights)

    em_commodity_total = (
        capped.get("EEM", 0) + capped.get("VWO", 0) +
        capped.get("GLD", 0) + capped.get("PDBC", 0)
    )
    assert em_commodity_total <= 0.201, f"EM+commodity {em_commodity_total:.1%} exceeds 20%"
    assert abs(sum(capped.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


def test_em_commodity_cap_no_op_when_under():
    """Cap must be a no-op when EM+commodity is already under 20%."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {"EEM": 0.05, "GLD": 0.06, "AAPL": 0.50, "MSFT": 0.39}
    capped = _cap_em_commodity(weights)

    assert abs(capped["EEM"] - 0.05) < 0.0001
    assert abs(capped["GLD"] - 0.06) < 0.0001


def test_em_commodity_cap_preserves_non_em():
    """Non-EM symbols should gain weight when EM is trimmed."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {"EEM": 0.20, "GLD": 0.15, "AAPL": 0.40, "JPM": 0.25}
    capped = _cap_em_commodity(weights)

    em_total = capped.get("EEM", 0) + capped.get("GLD", 0)
    assert em_total <= 0.201
    assert capped.get("AAPL", 0) > 0.40


def test_reduce_size_enforces_actual_reduction():
    """reduce_size must produce a measurably SMALLER BOOK, not a rotation.

    Rewritten 2026-07-28. This test previously asserted the old contract —
    `sum(enforced) == 1.0` plus ">=3 positions each cut by >=1pp" — which is
    exactly the behaviour the 2026-07-27 audit identified as the defect: gross
    exposure was 1.0000 before and after on all three dates reduce_size ever
    fired, so it never reduced anything. It concentrated 2pp cuts on the largest
    five names and recycled the freed weight into the rest, which on 07-27 pushed
    10pp out of dollar, duration and T-bills into equities 48h before FOMC under
    a de-risking verdict.

    The contract now is: gross falls, and the freed weight leaves the book as cash
    rather than moving to other positions. A per-position ">=1pp cut" assertion is
    not meaningful under proportional de-grossing of a 20-name book — a 10%
    de-gross moves the largest 10% position by exactly 1pp and everything else by
    less — so the assertions are on gross and on direction instead.
    """
    from ascent.execution.eod_runner import REDUCE_SIZE_GROSS_TARGET, _enforce_reduce_size

    original = {"AAPL": 0.10, "MSFT": 0.09, "AMZN": 0.08, "NVDA": 0.07, "JPM": 0.06,
                "V": 0.06, "MA": 0.05, "UNH": 0.05, "HD": 0.05, "PG": 0.05,
                "BRK": 0.04, "XOM": 0.04, "LLY": 0.04, "JNJ": 0.04, "AVGO": 0.04,
                "META": 0.04, "GOOGL": 0.04, "TSLA": 0.03, "COST": 0.03, "NKE": 0.03}

    # Haiku returns the same weights (no reduction) — enforcement must kick in.
    enforced = _enforce_reduce_size(original, dict(original))

    # 1. The book is genuinely smaller. This is the assertion that would have
    #    caught the original bug, and the old version of this test did not make it.
    assert sum(enforced.values()) < sum(original.values()) - 0.001
    assert sum(enforced.values()) <= REDUCE_SIZE_GROSS_TARGET + 1e-6

    # 2. No position GREW. A rotation shows up here as some name gaining weight.
    grew = {s: (original[s], w) for s, w in enforced.items() if w > original[s] + 1e-9}
    assert not grew, f"reduce_size increased positions (rotation, not reduction): {grew}"

    # 3. Every position shrank, so the reduction is spread rather than
    #    concentrated on the largest few.
    assert all(enforced[s] < original[s] for s in original)


def test_reduce_size_passes_through_a_genuine_reduction():
    """A real reduction — one that lowers gross — is left exactly as Haiku set it.

    Scaling it further would re-gross a book that had already been cut, the same
    class of bug as the old renormalize-to-1.0.
    """
    from ascent.execution.eod_runner import _enforce_reduce_size

    original = {"AAPL": 0.12, "MSFT": 0.10, "AMZN": 0.08, "NVDA": 0.07, "OTHER": 0.63}
    # Sums to 0.85: the freed weight became cash instead of moving elsewhere.
    adjusted = {"AAPL": 0.08, "MSFT": 0.06, "AMZN": 0.05, "NVDA": 0.04, "OTHER": 0.62}

    result = _enforce_reduce_size(original, adjusted)
    assert abs(result["AAPL"] - 0.08) < 0.001
    assert abs(result["MSFT"] - 0.06) < 0.001
    assert abs(sum(result.values()) - 0.85) < 0.001


def test_reduce_size_caps_a_rotation_masquerading_as_a_reduction():
    """Rewritten 2026-07-28.

    The previous fixture here cut four names while growing OTHER from 0.63 to
    0.77 — gross unchanged at 1.0 — and asserted it passed through untouched.
    That is the 2026-07-27 defect exactly: names trimmed, proceeds recycled, no
    risk removed. Haiku's relative shape is still honoured, but gross is capped.
    """
    import pytest

    from ascent.execution.eod_runner import REDUCE_SIZE_GROSS_TARGET, _enforce_reduce_size

    original = {"AAPL": 0.12, "MSFT": 0.10, "AMZN": 0.08, "NVDA": 0.07, "OTHER": 0.63}
    rotation = {"AAPL": 0.08, "MSFT": 0.06, "AMZN": 0.05, "NVDA": 0.04, "OTHER": 0.77}

    result = _enforce_reduce_size(original, rotation)
    assert sum(result.values()) <= REDUCE_SIZE_GROSS_TARGET + 1e-6
    # Relative shape preserved: AAPL still twice MSFT, as Haiku intended.
    assert result["AAPL"] / result["MSFT"] == pytest.approx(0.08 / 0.06, rel=1e-6)


def test_reduce_size_empty_haiku_returns_original():
    """If haiku returns empty weights, fall back to original."""
    from ascent.execution.eod_runner import _enforce_reduce_size

    original = {"AAPL": 0.50, "MSFT": 0.50}
    result = _enforce_reduce_size(original, {})
    assert result == original


# ── B3: Regime staleness detection ───────────────────────────────────────────

from datetime import date, timedelta
import json
from pathlib import Path


def test_regime_staleness_detected(tmp_path, monkeypatch):
    """Regime signal older than 5 days must be flagged as stale."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    stale_date = (date.today() - timedelta(days=8)).isoformat()
    sig = {"regime": "stressed", "label": "stressed", "as_of": stale_date, "last_refit_date": stale_date}
    (tmp_path / "dashboard" / "regime_signal.json").write_text(json.dumps(sig))
    import importlib
    import run_all_agents
    importlib.reload(run_all_agents)
    assert run_all_agents._is_regime_stale() is True


def test_regime_staleness_fresh(tmp_path, monkeypatch):
    """Regime signal updated today must not be stale."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    sig = {"regime": "calm_bull", "as_of": date.today().isoformat(), "last_refit_date": date.today().isoformat()}
    (tmp_path / "dashboard" / "regime_signal.json").write_text(json.dumps(sig))
    import importlib
    import run_all_agents
    importlib.reload(run_all_agents)
    assert run_all_agents._is_regime_stale() is False


def test_regime_list_schema_detected_as_stale(tmp_path, monkeypatch):
    """Old list-schema regime_signal.json must be detected as stale (needs migration)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()
    old_list = [{"d": "2026-04-01", "label": "calm_bull"}, {"d": "2026-04-17", "label": "stressed"}]
    (tmp_path / "dashboard" / "regime_signal.json").write_text(json.dumps(old_list))
    import importlib
    import run_all_agents
    importlib.reload(run_all_agents)
    assert run_all_agents._is_regime_stale() is True


# ── Task 1: International equity cap ─────────────────────────────────────────

def test_intl_equity_cap_fires_on_ewy_efa_overweight():
    """EWY + EFA combined > 15% must be trimmed to exactly 15%."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {
        "EWY": 0.10, "EFA": 0.10,   # 20% intl — over cap
        "AAPL": 0.40, "MSFT": 0.40,
    }
    result = _cap_intl_equity(weights)
    intl_total = result.get("EWY", 0) + result.get("EFA", 0)
    assert intl_total <= 0.151, f"Intl {intl_total:.1%} exceeds 15%"
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_intl_equity_cap_noop_when_under():
    """Cap must be a no-op when international equity is already under 15%."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {"EWY": 0.05, "EFA": 0.05, "AAPL": 0.50, "MSFT": 0.40}
    result = _cap_intl_equity(weights)
    assert abs(result["EWY"] - 0.05) < 0.0001
    assert abs(result["EFA"] - 0.05) < 0.0001


def test_intl_equity_cap_catches_ewy_alone():
    """EWY alone at 20% must be capped to 15%."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {"EWY": 0.20, "AAPL": 0.50, "MSFT": 0.30}
    result = _cap_intl_equity(weights)
    assert result.get("EWY", 0) <= 0.151
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_intl_equity_cap_freed_weight_goes_to_domestic():
    """Freed weight from intl trim must flow to non-intl names."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {"EWY": 0.12, "EEM": 0.10, "AAPL": 0.40, "MSFT": 0.38}
    result = _cap_intl_equity(weights)
    # EEM is in em_equity bucket → also caught; AAPL+MSFT must gain
    assert result.get("AAPL", 0) > 0.40


# ── Task 2: Macro-equity regime divergence penalty ───────────────────────────

from datetime import date
import pandas as pd
from ascent.config.types import AgentOutput


def _make_agent_output(agent_id, regime):
    return AgentOutput(
        agent_id=agent_id,
        as_of_date=date.today(),
        target_weights={"SPY": 0.50, "TLT": 0.50},
        regime_signal=regime,
        alpha_scores=pd.DataFrame(),
        skill_score=None,
        metadata={},
    )


def test_macro_divergence_scales_down_when_macro_stressed():
    """When macro=stressed and equity=calm_bull, merged weights should sum to ~0.90."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [
        _make_agent_output("us_equities", "calm_bull"),
        _make_agent_output("macro", "stressed"),
    ]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total <= 0.91, f"Expected ~0.90 total after divergence penalty, got {total:.3f}"
    assert total >= 0.85, f"Total {total:.3f} dropped too far — expected ~0.90"


def test_macro_divergence_no_penalty_when_regimes_agree():
    """When macro and equity both calm_bull, merged weights should sum to ~1.0."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [
        _make_agent_output("us_equities", "calm_bull"),
        _make_agent_output("macro", "calm_bull"),
    ]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total >= 0.98, f"Expected ~1.0 (no penalty), got {total:.3f}"


def test_macro_divergence_no_penalty_when_macro_not_present():
    """If macro agent is absent, no penalty should apply."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [_make_agent_output("us_equities", "calm_bull")]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total >= 0.98, f"Expected ~1.0 with no macro agent, got {total:.3f}"


def test_macro_divergence_no_penalty_when_equity_also_stressed():
    """When both macro and equity are stressed, no divergence — no penalty."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [
        _make_agent_output("us_equities", "stressed"),
        _make_agent_output("macro", "stressed"),
    ]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total >= 0.98, f"Expected ~1.0 (same regime), got {total:.3f}"
