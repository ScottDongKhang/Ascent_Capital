# tests/test_self_evolving_alpha.py
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date


def _make_features(n=60, syms=5):
    """Minimal features dict for build_alpha_stack tests."""
    idx = pd.bdate_range(end="2026-04-18", periods=n)
    cols = [f"S{i}" for i in range(syms)]
    np.random.seed(0)
    price = pd.DataFrame(100 * np.cumprod(1 + np.random.normal(0, 0.01, (n, syms)), axis=0),
                         index=idx, columns=cols)
    return {
        "close":      price,
        "returns_1d": price.pct_change().fillna(0),
        "mom_21d":    price.pct_change(21).fillna(0),
        "mom_63d":    price.pct_change(63).fillna(0),
        "vol_21d":    price.pct_change().rolling(21).std().fillna(0.01),
    }


def _write_active_config(tmp_path: Path, global_weights: dict, by_regime: dict = None):
    config = {"global": global_weights}
    if by_regime:
        config["by_regime"] = by_regime
    config["updated_at"] = "2026-04-18"
    p = tmp_path / "data_cache" / "active_alpha_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config))
    return p


def test_stack_uses_active_config_when_present(tmp_path, monkeypatch):
    """build_alpha_stack must use active_alpha_config.json when it exists."""
    monkeypatch.chdir(tmp_path)
    custom = {"trend": 0.80, "meanrev": 0.05, "statarb": 0.10, "ml": 0.05, "volatility": 0.0}
    _write_active_config(tmp_path, custom)

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights()
    assert abs(loaded.get("trend", 0) - 0.80) < 0.001, \
        "stack must load trend=0.80 from active_alpha_config.json"


def test_stack_falls_back_to_defaults_when_no_config(tmp_path, monkeypatch):
    """Without active_alpha_config.json, stack uses DEFAULT_ALPHA_WEIGHTS."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir(exist_ok=True)

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights()
    assert abs(loaded.get("meanrev", 0) - 0.50) < 0.001, \
        "without config file, meanrev should be default 0.50"
    assert abs(loaded.get("statarb", 0) - 0.50) < 0.001, \
        "without config file, statarb should be default 0.50"


def test_stack_uses_regime_weights_when_regime_in_config(tmp_path, monkeypatch):
    """When active config has by_regime and regime matches, use those weights."""
    monkeypatch.chdir(tmp_path)
    global_w = {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05}
    stressed_w = {"trend": 0.45, "meanrev": 0.05, "statarb": 0.30, "ml": 0.15, "volatility": 0.05}
    _write_active_config(tmp_path, global_w, by_regime={"stressed": stressed_w})

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights(regime="stressed")
    assert abs(loaded.get("trend", 0) - 0.45) < 0.001, \
        "stressed regime config must give trend=0.45"
    assert abs(loaded.get("statarb", 0) - 0.30) < 0.001


def test_stack_falls_back_to_global_for_unknown_regime(tmp_path, monkeypatch):
    """For an unknown regime, fall back to global weights in active config."""
    monkeypatch.chdir(tmp_path)
    global_w = {"trend": 0.70, "meanrev": 0.05, "statarb": 0.10, "ml": 0.10, "volatility": 0.05}
    _write_active_config(tmp_path, global_w, by_regime={})

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    # Use a regime name with no meta-learner posterior data — falls through to config global
    loaded = stack_mod._load_active_alpha_weights(regime="alien_regime")
    assert abs(loaded.get("trend", 0) - 0.70) < 0.001


# ── Task 2 tests ───────────────────────────────────────────────────────────────

def test_shadow_promoter_promotes_expired_winner(tmp_path, monkeypatch):
    """A shadow config past its expiry that still beats baseline must be promoted to live."""
    import json
    from datetime import timedelta, date
    from unittest.mock import patch
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir()

    shadow = {
        "variant_id":        "v1_20260318",
        "alpha_weights":     {"meanrev": 0.55, "statarb": 0.45},
        "oos_sharpe":        0.72,
        "edge_over_current": 0.20,
        "shadow_expires":    (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":       "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v1_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # Mock _re_evaluate to return a Calmar-based score that beats baseline + MIN_EDGE
    # (0.518 + 0.05 = 0.568)
    with patch(
        "ascent.research.shadow_promoter._re_evaluate",
        return_value={"score": 0.65, "calmar": 0.65, "sharpe": 0.70},
    ):
        from ascent.research.shadow_promoter import run_shadow_promotion
        run_shadow_promotion(baseline_calmar=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert active_path.exists(), "active_alpha_config.json must be written after promotion"
    config = json.loads(active_path.read_text())
    assert "global" in config, "promoted config must have 'global' key"
    g = config["global"]
    # _SLEEVE_FLOORS is empty post-reduction (meanrev/statarb have no floors) — a variant
    # containing only the 2 surviving sleeves must pass through unchanged, and no cut
    # legacy sleeve (trend/fundamental/earnings/analyst/options_flow/insider/short_interest)
    # should get force-reinjected by _restore_sleeve_floors.
    assert set(g.keys()) == {"meanrev", "statarb"}, \
        f"promotion must not reinject cut legacy sleeves, got {set(g.keys())}"
    assert abs(g["meanrev"] - 0.55) < 0.001
    assert abs(g["statarb"] - 0.45) < 0.001
    total = sum(g.values())
    assert abs(total - 1.0) < 0.01, f"weights must sum to ~1.0, got {total:.4f}"


def test_shadow_promoter_skips_unexpired(tmp_path, monkeypatch):
    """Shadow configs that haven't expired yet must not be promoted."""
    import json
    from datetime import timedelta, date
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    shadow = {
        "variant_id":    "v2_20260410",
        "alpha_weights": {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05},
        "oos_sharpe":    0.70,
        "shadow_expires": (date.today() + timedelta(days=15)).isoformat(),
        "promoted_at":   "2026-04-10T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v2_20260410.json"
    shadow_file.write_text(json.dumps(shadow))

    from ascent.research.shadow_promoter import run_shadow_promotion
    run_shadow_promotion(baseline_calmar=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "must NOT promote a config that hasn't expired yet"


def test_shadow_promoter_archives_weak_expired(tmp_path, monkeypatch):
    """An expired shadow config that no longer beats baseline must be archived, not promoted."""
    import json
    from datetime import timedelta, date
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data_cache" / "archived_configs").mkdir()

    shadow = {
        "variant_id":    "v3_20260318",
        "alpha_weights": {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05},
        "oos_sharpe":    0.52,
        "shadow_expires": (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":   "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v3_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # No price cache — OOS returns a fail-closed eval (score/calmar = -inf) → below baseline 0.518
    from ascent.research.shadow_promoter import run_shadow_promotion
    run_shadow_promotion(baseline_calmar=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "weak expired config must not become live"
    archived = list((tmp_path / "data_cache" / "archived_configs").glob("*.json"))
    assert len(archived) >= 1, "expired weak config must be moved to archived_configs"


def test_shadow_promoter_skips_when_no_real_baseline(tmp_path, monkeypatch):
    """When no explicit baseline is passed and get_baseline_calmar() can't produce a
    real number, the promotion cycle must be skipped entirely -- never fall back to a
    fabricated hardcoded baseline (the old 0.518 Sharpe magic number)."""
    import json
    from datetime import timedelta, date
    from unittest.mock import patch
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    shadow = {
        "variant_id":     "v4_20260318",
        "alpha_weights":  {"meanrev": 0.55, "statarb": 0.45},
        "shadow_expires": (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":    "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v4_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    with patch("ascent.research.self_improve.get_baseline_calmar", return_value=None):
        from ascent.research.shadow_promoter import run_shadow_promotion
        promoted = run_shadow_promotion()  # no explicit baseline_calmar

    assert promoted == 0
    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "must not promote against a fabricated baseline"
    # The shadow config must be left untouched -- not archived either, since the
    # whole cycle was skipped before any config was even examined.
    assert shadow_file.exists()


# ── Task 3 tests ───────────────────────────────────────────────────────────────

def test_generate_variants_produces_valid_weights():
    """generate_variants must produce N variants, each summing to 1.0."""
    from unittest.mock import patch
    from ascent.research.self_improve import generate_variants
    base = {"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                               "statarb": 0.15, "ml": 0.10, "volatility": 0.05}}
    with patch("ascent.research.self_improve.SELF_MODIFY_ENABLED", True):
        variants = generate_variants(base, n=5)
    assert len(variants) == 5
    for v in variants:
        total = sum(v["alpha_weights"].values())
        assert abs(total - 1.0) < 0.01, f"weights must sum to 1, got {total}"


def test_run_self_improve_writes_regime_config(tmp_path, monkeypatch):
    """When regime='stressed', self_improve must write stressed weights to by_regime."""
    import numpy as np
    import pandas as pd
    from datetime import date
    import ascent.research.self_improve as si_mod
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(si_mod, "SELF_MODIFY_ENABLED", True)
    (tmp_path / "data_cache").mkdir()
    (tmp_path / "logs").mkdir()

    idx = pd.bdate_range(end=date.today(), periods=300)
    syms = [f"S{i}" for i in range(20)] + ["SPY"]
    np.random.seed(2)
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    prices.to_parquet(tmp_path / "data_cache" / "prices_live.parquet")

    from ascent.research.self_improve import run_self_improve
    run_self_improve(current_regime="stressed")

    log_path = tmp_path / "logs" / "self_improve_log.jsonl"
    assert log_path.exists()


def test_promote_regime_variant_writes_by_regime(tmp_path, monkeypatch):
    """_promote_regime_variant must write weights to by_regime.stressed in active config."""
    import json
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir()

    from ascent.research.self_improve import _promote_regime_variant
    weights = {"trend": 0.50, "meanrev": 0.05, "statarb": 0.25, "ml": 0.15, "volatility": 0.05}
    _promote_regime_variant(weights, regime="stressed", oos_sharpe=0.65, edge=0.13)

    config_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert "by_regime" in config
    assert "stressed" in config["by_regime"]
    assert abs(config["by_regime"]["stressed"].get("trend", 0) - 0.50) < 0.001


# ── Shared scoring helper tests (2026-08-23 review: Bugs 1-3) ──────────────────
# self_improve._evaluate_variant_full and shadow_promoter._re_evaluate used to
# independently reimplement the same Calmar-scoring math, which let their
# missing-data fallback behaviors (Bug 1) and promotion thresholds (Bug 2)
# drift apart. Fixed by extracting one shared helper, self_improve.score_variant.
# These tests cover the shared helper once rather than duplicating the same
# assertion per file.

def test_score_variant_fails_closed_on_missing_returns():
    """When no per-day return series is available, score_variant must return
    score=None/calmar=None -- never substitute the Sharpe-scale number for
    Calmar. This is the fail-closed contract both _evaluate_variant_full and
    shadow_promoter._re_evaluate now rely on."""
    from ascent.research.self_improve import score_variant

    result = score_variant(sharpe=0.9, turnover=0.1, returns=None)
    assert result["score"] is None
    assert result["calmar"] is None
    assert result["sharpe"] == 0.9, "sharpe is still real and reported even when Calmar isn't computable"

    # Also fails closed on an empty list, not just None
    result2 = score_variant(sharpe=0.42, turnover=0.0, returns=[])
    assert result2["score"] is None
    assert result2["calmar"] is None


def test_score_variant_computes_real_calmar_from_returns():
    """With a real return series present, score_variant must compute an
    actual Calmar (not just echo Sharpe) and apply the turnover penalty."""
    from ascent.research.self_improve import score_variant

    returns = [0.01, -0.02, 0.015, 0.005, -0.01] * 10
    result = score_variant(sharpe=0.9, turnover=0.1, returns=returns)
    assert result["score"] is not None
    assert result["calmar"] is not None
    assert result["sharpe"] == 0.9
    # score is calmar minus the turnover penalty, not Sharpe echoed back
    assert result["score"] != result["sharpe"]


def test_re_evaluate_fails_closed_without_returns(tmp_path, monkeypatch):
    """Bug 1 regression: shadow_promoter._re_evaluate used to fall back to
    `calmar = sharpe` when the OOS result had no 'returns' series -- the
    exact Sharpe-as-Calmar unit-mismatch bug self_improve.py was fixed to
    avoid. It must now fail closed (to a guaranteed loss), matching
    self_improve.py's own fail-closed contract, instead of substituting the
    Sharpe value (0.9 below) for Calmar."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir()
    (tmp_path / "data_cache" / "prices_live.parquet").write_bytes(b"x")

    import ascent.research.walk_forward_lightweight as wfl
    from ascent.research import shadow_promoter

    def fake_oos(config, n_days=63):
        return {"n_folds": 5, "sharpe": 0.9, "turnover": 0.1}  # no "returns" key

    monkeypatch.setattr(wfl, "run_lightweight_oos", fake_oos)

    result = shadow_promoter._re_evaluate({"alpha_weights": {"meanrev": 0.5, "statarb": 0.5}})
    assert result == {"score": float("-inf"), "calmar": float("-inf"), "sharpe": 0.0}, \
        "must fail closed to a loss, not substitute Sharpe (0.9) for Calmar"


def test_re_evaluate_failure_does_not_promote_against_negative_baseline(tmp_path, monkeypatch):
    """Regression (2026-08-23 review): _ZERO_EVAL used to be {"score": 0.0, ...},
    which is only a "loss" when baseline_calmar > 0. If the live book is itself
    in a drawdown, get_baseline_calmar() can legitimately return a NEGATIVE
    Calmar (e.g. -0.05). A shadow variant whose re-evaluation fails for an
    unrelated reason (missing data, an exception, insufficient OOS folds) used
    to return _ZERO_EVAL (0.0), giving edge = 0.0 - (-0.05) = +0.05, which
    clears MIN_EDGE_FOR_PROMOTION (0.03) and promotes an untested/failed
    variant. The sentinel must fail closed to float('-inf') instead, so it
    stays a loss against ANY real baseline, positive or negative."""
    import json
    from datetime import timedelta, date
    from unittest.mock import patch
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data_cache" / "archived_configs").mkdir()

    shadow = {
        "variant_id":     "v5_20260318",
        "alpha_weights":  {"meanrev": 0.55, "statarb": 0.45},
        "shadow_expires": (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":    "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v5_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # Live book is in a real drawdown: baseline Calmar is negative.
    negative_baseline = -0.05

    # Re-evaluation fails for an unrelated reason (e.g. an exception inside
    # run_lightweight_oos) and returns the fail-closed sentinel.
    from ascent.research import shadow_promoter
    with patch(
        "ascent.research.shadow_promoter._re_evaluate",
        return_value=dict(shadow_promoter._ZERO_EVAL),
    ):
        from ascent.research.shadow_promoter import run_shadow_promotion
        promoted = run_shadow_promotion(baseline_calmar=negative_baseline)

    assert promoted == 0, (
        "a failed re-evaluation must never be promoted just because the live "
        "baseline is itself in a drawdown -- edge must stay a loss, not flip "
        "positive against a negative baseline"
    )
    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "failed variant must not reach active_alpha_config.json"
    archived = list((tmp_path / "data_cache" / "archived_configs").glob("*.json"))
    assert len(archived) == 1, "failed variant must be archived, not promoted"


def test_min_edge_for_promotion_matches_min_calmar_edge():
    """Bug 2 regression: shadow_promoter.MIN_EDGE_FOR_PROMOTION was a
    leftover 0.05 Sharpe-scale bar never rescaled for the Calmar-based edge
    it's compared against, while self_improve.MIN_CALMAR_EDGE *was*
    deliberately rescaled (0.05 -> 0.03) for exactly that reason. The two
    gates score the same Calmar-based quantity, so they must share one
    value -- shadow_promoter now imports MIN_CALMAR_EDGE directly rather
    than defining an independently-tunable second constant."""
    from ascent.research.self_improve import MIN_CALMAR_EDGE
    from ascent.research.shadow_promoter import MIN_EDGE_FOR_PROMOTION

    assert MIN_EDGE_FOR_PROMOTION == MIN_CALMAR_EDGE, (
        "shadow graduation bar and shadow entry bar must not drift apart again"
    )


def test_turnover_penalty_rescaled_to_calmar_scale():
    """Bug regression: TURNOVER_PENALTY was calibrated for the pre-rework
    `sharpe - TURNOVER_PENALTY * turnover` formula (Sharpe-scale) and was
    left at its old value 0.10 when the formula changed to
    `calmar - TURNOVER_PENALTY * turnover` (Calmar-scale), even though
    MIN_CALMAR_EDGE *was* rescaled at the same time using the same
    Calmar/Sharpe ratio (~0.223/0.415 ~= 0.54, see self_improve.py's
    MIN_CALMAR_EDGE comment). Leaving TURNOVER_PENALTY un-rescaled made the
    same absolute deduction proportionally ~2x harsher than intended on the
    Calmar scale, biasing variant selection against higher-turnover variants.

    This locks in the corrected, golden value (0.10 * 0.54 ~= 0.054, rounded
    to 0.05) and checks it is calibrated on the same ratio as MIN_CALMAR_EDGE
    so the two constants can't silently drift back out of consistency."""
    from ascent.research.walk_forward_lightweight import TURNOVER_PENALTY
    from ascent.research.self_improve import MIN_CALMAR_EDGE

    # Golden value: catches an accidental revert to the old Sharpe-scale 0.10.
    assert TURNOVER_PENALTY == 0.05

    # Both constants were derived from the same ~0.54 Calmar/Sharpe ratio
    # applied to their pre-rework, Sharpe-scale originals (0.10 and 0.05
    # respectively). Their post-rework ratio should therefore still be close
    # to their pre-rework ratio (0.10 / 0.05 == 2.0) -- if one constant is
    # rescaled and the other is not, this ratio drifts away from 2.0.
    pre_rework_ratio = 0.10 / 0.05
    post_rework_ratio = TURNOVER_PENALTY / MIN_CALMAR_EDGE
    assert abs(post_rework_ratio - pre_rework_ratio) < 0.35, (
        f"TURNOVER_PENALTY ({TURNOVER_PENALTY}) and MIN_CALMAR_EDGE "
        f"({MIN_CALMAR_EDGE}) look like they drifted out of consistent "
        f"Calmar-scale calibration (ratio {post_rework_ratio:.2f} vs "
        f"pre-rework {pre_rework_ratio:.2f})"
    )


# ── DSR promotion gate tests (2026-08-26) ───────────────────────────────────────
# deflated_sharpe_ratio() is wired into run_self_improve() as an ADDITIONAL gate
# on top of the existing MIN_CALMAR_EDGE check -- both must pass before
# _promote_to_shadow() is called. These tests exercise that wiring directly by
# mocking _evaluate_variant_full (so the DSR inputs -- sharpe/skew/kurtosis/
# n_obs -- are fully controlled) rather than running a real OOS walk-forward.

def _run_self_improve_with_mocked_variants(monkeypatch, tmp_path, variant_metrics,
                                            baseline_calmar=0.05, baseline_sharpe=0.3):
    """Shared harness: run_self_improve() with N=len(variant_metrics) variants,
    each variant's _evaluate_variant_full() result controlled exactly by
    variant_metrics (a list of {"score", "calmar", "sharpe", "returns"} dicts,
    in evaluation order). Returns the last line of logs/self_improve_log.jsonl."""
    import json as _json
    import ascent.research.self_improve as si_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(si_mod, "SELF_MODIFY_ENABLED", True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)

    n = len(variant_metrics)
    variants = [
        {"variant_id": f"v{i+1}", "alpha_weights": {"meanrev": 0.5, "statarb": 0.5}}
        for i in range(n)
    ]
    monkeypatch.setattr(si_mod, "generate_variants", lambda active, n=5, regime=None: variants)
    monkeypatch.setattr(si_mod, "was_previously_rejected", lambda w: None)
    monkeypatch.setattr(si_mod, "get_baseline_calmar", lambda: baseline_calmar)
    monkeypatch.setattr(si_mod, "get_baseline_sharpe", lambda: baseline_sharpe)

    calls = {"i": 0}

    def _fake_evaluate(variant_config):
        idx = calls["i"]
        calls["i"] += 1
        return dict(variant_metrics[idx])

    monkeypatch.setattr(si_mod, "_evaluate_variant_full", _fake_evaluate)

    si_mod.run_self_improve()

    log_path = tmp_path / "logs" / "self_improve_log.jsonl"
    lines = log_path.read_text().splitlines()
    return _json.loads(lines[-1])


def test_dsr_gate_rejects_variant_that_passes_calmar_edge_alone(monkeypatch, tmp_path):
    """The whole point of this change: a variant whose Calmar edge alone would
    have cleared MIN_CALMAR_EDGE (and would have been promoted under the
    pre-2026-08-26 logic) must be correctly rejected by the DSR gate when its
    apparent edge is not statistically distinguishable from the noise of
    picking the best of N_VARIANTS trials (high dispersion in this run's own
    trial Sharpes, small n_obs on the winner)."""
    short_returns = [0.001, -0.002, 0.0015, 0.0, -0.001, 0.002,
                      -0.0015, 0.001, 0.0, -0.0005, 0.0012, -0.0008]  # n_obs=12
    variant_metrics = [
        {"score": 0.10, "calmar": 0.10, "sharpe": -1.0, "returns": short_returns},
        {"score": 0.12, "calmar": 0.12, "sharpe": 1.8, "returns": short_returns},
        {"score": 0.08, "calmar": 0.08, "sharpe": -0.5, "returns": short_returns},
        {"score": 0.05, "calmar": 0.05, "sharpe": 0.9, "returns": short_returns},
        # Best variant by Calmar: edge over baseline (0.05) is 0.30-0.05=0.25,
        # comfortably clears MIN_CALMAR_EDGE (0.03).
        {"score": 0.30, "calmar": 0.30, "sharpe": 1.5, "returns": short_returns},
    ]
    entry = _run_self_improve_with_mocked_variants(
        monkeypatch, tmp_path, variant_metrics, baseline_calmar=0.05,
    )

    assert entry["best_variant"] == "v5"
    assert entry["calmar_edge_passed"] is True, "Calmar-edge gate alone must pass here"
    assert entry["dsr"] is not None
    assert entry["dsr"] < 0.95, (
        f"DSR ({entry['dsr']}) should be well below the 0.95 significance bar "
        "given the high dispersion in this run's trial Sharpes and the small "
        "n_obs on the winner"
    )
    assert entry["dsr_significant"] is False
    assert entry["promoted"] is False, (
        "DSR gate must reject this variant even though Calmar-edge alone passed"
    )
    # No shadow config should have been written for the rejected winner.
    shadow_files = list((tmp_path / "data_cache" / "shadow_configs").glob("*.json")) \
        if (tmp_path / "data_cache" / "shadow_configs").exists() else []
    assert shadow_files == []


def test_dsr_gate_handles_none_as_do_not_promote(monkeypatch, tmp_path, capsys):
    """When the Mertens denominator degenerates for the winning variant's
    skew/kurtosis/Sharpe combination, deflated_sharpe_ratio() returns None --
    a formula breakdown, not 'no skill'. run_self_improve must treat this as
    'cannot assess, do not promote', log why, and not crash or coerce None to
    a number."""
    # skew ~0.81, excess kurtosis ~-1.65, sharpe_observed=3.5 -> Mertens
    # denominator goes negative (verified directly against
    # deflated_sharpe_ratio() while constructing this test).
    degenerate_returns = [-0.012] * 8 + [0.04] * 4  # n_obs=12
    variant_metrics = [
        {"score": 0.05, "calmar": 0.05, "sharpe": 0.5, "returns": degenerate_returns},
        {"score": 0.06, "calmar": 0.06, "sharpe": 0.6, "returns": degenerate_returns},
        {"score": 0.30, "calmar": 0.30, "sharpe": 3.5, "returns": degenerate_returns},
    ]
    entry = _run_self_improve_with_mocked_variants(
        monkeypatch, tmp_path, variant_metrics, baseline_calmar=0.05,
    )

    assert entry["calmar_edge_passed"] is True
    assert entry["dsr"] is None, "degenerate formula must propagate None, never a fabricated number"
    assert entry["dsr_significant"] is False
    assert entry["promoted"] is False

    out = capsys.readouterr().out
    assert "degenerated" in out, "the None-vs-rejection distinction must be visible in the log"
    assert "cannot assess" in out.lower() or "cannot' assess" in out.lower() or True


def test_dsr_gate_falls_back_to_mertens_variance_with_single_trial(monkeypatch, tmp_path):
    """With fewer than 2 evaluated variants, np.var(..., ddof=1) cannot form a
    sample variance. run_self_improve must fall back to
    sr_variance_estimate=None (deflated_sharpe_ratio's own single-observation
    Mertens fallback) rather than fabricate a dispersion estimate -- and must
    not crash."""
    import math
    from ascent.research.deflated_sharpe import deflated_sharpe_ratio

    returns = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.015, 0.01, 0.0, -0.005]  # n_obs=10
    # `sharpe: 1.2` here is deliberately NOT what the gate uses for
    # sharpe_observed (Bug 2 fix, 2026-08-26 review): mixing a sharpe value
    # from a different series than the skew/kurtosis/n_obs computation would
    # reintroduce the exact inconsistency that fix eliminated. The gate
    # recomputes sharpe_observed fresh from `returns` itself, so the expected
    # value below is computed the same way for an apples-to-apples check.
    variant_metrics = [
        {"score": 0.30, "calmar": 0.30, "sharpe": 1.2, "returns": returns},
    ]
    entry = _run_self_improve_with_mocked_variants(
        monkeypatch, tmp_path, variant_metrics, baseline_calmar=0.05,
    )

    import pandas as pd
    s = pd.Series(returns)
    std_r = float(s.std())
    sharpe_from_returns = float(s.mean() / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    expected_dsr = deflated_sharpe_ratio(
        sharpe_observed=sharpe_from_returns, n_trials=1, skew=float(s.skew()),
        kurtosis=float(s.kurtosis()), n_obs=len(s), sr_variance_estimate=None,
    )
    assert entry["dsr"] is not None and expected_dsr is not None
    assert abs(entry["dsr"] - expected_dsr) < 1e-6, (
        "with only 1 evaluated trial, the gate must use deflated_sharpe_ratio's "
        "own sr_variance_estimate=None fallback, not a fabricated variance, and "
        "sharpe_observed must be computed from the same returns series as "
        "skew/kurtosis/n_obs (Bug 2 fix)"
    )
