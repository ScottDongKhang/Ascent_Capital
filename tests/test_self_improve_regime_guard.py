# tests/test_self_improve_regime_guard.py
"""
Regression test for the per-regime promotion fail-closed guard.

Bug (code review, 2026-08-25): run_self_improve() sets `current_calmar =
float("-inf")` as a fail-closed sentinel when no live-or-OOS Calmar baseline
is available, and correctly derives `edge = float("-inf")` for the MAIN
promotion decision from that sentinel. But the PER-REGIME promotion block
further down recomputed its own edge from scratch as
`regime_edge = live_calmar - current_calmar`, which becomes
`live_calmar - (-inf) == +inf` for ANY live_calmar -- always clearing
MIN_CALMAR_EDGE and promoting a per-regime variant with zero valid baseline
comparison, the opposite of "fail closed: no promotion this run".

Fix: guard the per-regime block with `math.isfinite(current_calmar)` so it is
skipped entirely whenever the sentinel is in play, mirroring the main path.
"""
import json
from datetime import date

import pytest

import ascent.research.self_improve as si


def _variant(variant_id, oos_calmar, oos_sharpe=0.5, weights=None):
    return {
        "variant_id": variant_id,
        "alpha_weights": weights or {"meanrev": 0.5, "statarb": 0.5},
        "oos_calmar": oos_calmar,
        "oos_sharpe": oos_sharpe,
    }


def _patch_common(monkeypatch, tmp_path, variants, current_calmar, current_sharpe=0.4):
    """Wire run_self_improve's dependencies to return controlled values, and
    redirect file writes into tmp_path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    monkeypatch.setattr(si, "LOG_PATH", tmp_path / "logs" / "self_improve_log.jsonl")
    monkeypatch.setattr(si, "ACTIVE_CONFIG_PATH", tmp_path / "data_cache" / "active_alpha_config.json")

    monkeypatch.setattr(si, "SELF_MODIFY_ENABLED", True)
    monkeypatch.setattr(si, "_load_active_config", lambda: {"alpha_weights": dict(si.DEFAULT_ALPHA_WEIGHTS)})
    monkeypatch.setattr(si, "generate_variants", lambda base_config, n=si.N_VARIANTS, regime=None: variants)
    monkeypatch.setattr(si, "was_previously_rejected", lambda weights: None)
    monkeypatch.setattr(si, "record_verdict", lambda **kwargs: None)

    # _evaluate_variant_full is called per-variant in the normal (non-rejected)
    # path, returning each variant's own preset oos_calmar/oos_sharpe as
    # "score"/"sharpe". It is also called once on the *active* config when
    # get_baseline_calmar()/get_baseline_sharpe() return None (the
    # no-baseline scenario) -- `active` has no oos_calmar/oos_sharpe keys, so
    # fall back to None there (which is what drives the -inf sentinel path).
    def _fake_evaluate(v):
        return {"score": v.get("oos_calmar"), "sharpe": v.get("oos_sharpe")}
    monkeypatch.setattr(si, "_evaluate_variant_full", _fake_evaluate)

    monkeypatch.setattr(si, "get_baseline_calmar", lambda: current_calmar)
    monkeypatch.setattr(si, "get_baseline_sharpe", lambda: current_sharpe)


def test_no_baseline_blocks_per_regime_promotion(monkeypatch, tmp_path):
    """current_calmar == None (no live/OOS baseline) -> sentinel -inf -> the
    per-regime block must NOT call _promote_regime_variant, no matter how good
    the best regime variant looks."""
    variants = [_variant("v0", oos_calmar=1.5, oos_sharpe=1.2)]  # looks great
    _patch_common(monkeypatch, tmp_path, variants, current_calmar=None)

    promote_calls = []
    monkeypatch.setattr(si, "_promote_regime_variant", lambda *a, **kw: promote_calls.append((a, kw)))

    results = si.run_self_improve(current_regime="bull")

    assert results, "expected variants to be evaluated"
    assert promote_calls == [], (
        "per-regime promotion fired with no valid Calmar baseline -- "
        "the -inf sentinel was misused as a real number (live_calmar - (-inf) == +inf)"
    )


def test_valid_baseline_still_promotes_per_regime(monkeypatch, tmp_path):
    """Regression safety: a genuine finite baseline with a real edge above
    MIN_CALMAR_EDGE must still promote via the per-regime path."""
    current_calmar = 0.10
    variants = [_variant("v0", oos_calmar=current_calmar + si.MIN_CALMAR_EDGE + 0.05, oos_sharpe=0.6)]
    _patch_common(monkeypatch, tmp_path, variants, current_calmar=current_calmar)

    promote_calls = []
    monkeypatch.setattr(si, "_promote_regime_variant", lambda *a, **kw: promote_calls.append((a, kw)))

    results = si.run_self_improve(current_regime="bull")

    assert results
    assert len(promote_calls) == 1, "expected per-regime promotion with a genuine positive edge"


def test_no_regime_specified_skips_block_regardless_of_baseline(monkeypatch, tmp_path):
    """Sanity: without current_regime, the per-regime block never runs at all."""
    variants = [_variant("v0", oos_calmar=5.0, oos_sharpe=1.0)]
    _patch_common(monkeypatch, tmp_path, variants, current_calmar=None)

    promote_calls = []
    monkeypatch.setattr(si, "_promote_regime_variant", lambda *a, **kw: promote_calls.append((a, kw)))

    results = si.run_self_improve(current_regime=None)

    assert results
    assert promote_calls == []
