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
    """reduce_size must produce weights that are measurably smaller than originals."""
    from ascent.execution.eod_runner import _enforce_reduce_size

    original = {"AAPL": 0.10, "MSFT": 0.09, "AMZN": 0.08, "NVDA": 0.07, "JPM": 0.06,
                "V": 0.06, "MA": 0.05, "UNH": 0.05, "HD": 0.05, "PG": 0.05,
                "BRK": 0.04, "XOM": 0.04, "LLY": 0.04, "JNJ": 0.04, "AVGO": 0.04,
                "META": 0.04, "GOOGL": 0.04, "TSLA": 0.03, "COST": 0.03, "NKE": 0.03}

    # Haiku returns same weights (no reduction) — enforcement must kick in
    unchanged = dict(original)
    enforced = _enforce_reduce_size(original, unchanged)

    reduced_count = sum(1 for s, w in enforced.items() if w < original.get(s, 0) - 0.01)
    assert reduced_count >= 3, f"Expected >=3 positions reduced, got {reduced_count}"
    assert abs(sum(enforced.values()) - 1.0) < 0.001


def test_reduce_size_passes_through_genuine_reduction():
    """If Haiku genuinely reduced positions, pass through unchanged."""
    from ascent.execution.eod_runner import _enforce_reduce_size

    original = {"AAPL": 0.12, "MSFT": 0.10, "AMZN": 0.08, "NVDA": 0.07, "OTHER": 0.63}
    adjusted = {"AAPL": 0.08, "MSFT": 0.06, "AMZN": 0.05, "NVDA": 0.04, "OTHER": 0.77}

    result = _enforce_reduce_size(original, adjusted)
    assert abs(result["AAPL"] - 0.08) < 0.001
    assert abs(result["MSFT"] - 0.06) < 0.001


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
