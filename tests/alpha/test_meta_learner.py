# tests/alpha/test_meta_learner.py
import json
import tempfile
from pathlib import Path

import pytest


def _make_learner(tmp_path, state=None):
    from ascent.alpha.meta_learner import SleeveMetaLearner
    p = Path(tmp_path) / "posteriors.json"
    learner = SleeveMetaLearner(posteriors_path=p)
    if state:
        learner._state = state
    return learner


def test_get_weights_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        learner = _make_learner(tmp)
        assert learner.get_weights("calm_bull", {"trend": 0.58, "meanrev": 0.05}) is None


def test_get_weights_returns_none_when_sparse():
    with tempfile.TemporaryDirectory() as tmp:
        learner = _make_learner(tmp, state={
            "calm_bull": {"trend": {"mu": 0.015, "var": 0.005, "n": 2}}
        })
        # n=2 < _MIN_OBS_TRUST=3 → None
        assert learner.get_weights("calm_bull", {"trend": 0.58, "meanrev": 0.05}) is None


def test_get_weights_with_sufficient_observations():
    with tempfile.TemporaryDirectory() as tmp:
        learner = _make_learner(tmp, state={
            "calm_bull": {
                "trend":   {"mu": 0.015, "var": 0.003, "n": 5},
                "meanrev": {"mu": 0.002, "var": 0.003, "n": 5},
                "statarb": {"mu": -0.005, "var": 0.003, "n": 5},
            }
        })
        defaults = {"trend": 0.58, "meanrev": 0.05, "statarb": 0.0}
        result = learner.get_weights("calm_bull", defaults)
        assert result is not None
        assert abs(sum(result.values()) - 1.0) < 0.02
        # statarb negative mu → zero Kelly contribution
        # at n=5, alpha_conf=0.25 → blends toward default 0.0 → still near 0
        assert result.get("statarb", 0) < 0.02


def test_negative_ic_has_less_weight_than_positive():
    with tempfile.TemporaryDirectory() as tmp:
        learner = _make_learner(tmp, state={
            "calm_bull": {
                "trend":   {"mu": 0.015, "var": 0.003, "n": 5},
                "statarb": {"mu": -0.010, "var": 0.003, "n": 5},
            }
        })
        result = learner.get_weights("calm_bull", {"trend": 0.58, "statarb": 0.15})
        assert result is not None
        assert result["trend"] > result["statarb"]


def test_update_rebalance_moves_posterior():
    with tempfile.TemporaryDirectory() as tmp:
        learner = _make_learner(tmp, state={
            "calm_bull": {"trend": {"mu": 0.0, "var": 0.005, "n": 0}}
        })
        learner.update_rebalance("calm_bull", {"trend": 0.020})
        s = learner._state["calm_bull"]["trend"]
        assert s["mu"] > 0.0
        assert s["n"] == 1
        assert s["var"] < 0.005  # posterior variance tightened


def test_update_creates_new_regime():
    with tempfile.TemporaryDirectory() as tmp:
        learner = _make_learner(tmp)
        learner.update_rebalance("stressed", {"trend": 0.010, "statarb": 0.015})
        assert "stressed" in learner._state
        assert "trend" in learner._state["stressed"]
        assert learner._state["stressed"]["trend"]["n"] == 1


def test_seed_from_ic_log_sets_mu():
    with tempfile.TemporaryDirectory() as tmp:
        ic_log = Path(tmp) / "sleeve_ic_log.jsonl"
        entries = [
            {"date": "2026-05-01", "sleeves": {"trend": {"mean_ic": 0.015}, "statarb": {"mean_ic": -0.002}}},
            {"date": "2026-05-02", "sleeves": {"trend": {"mean_ic": 0.012}, "statarb": {"mean_ic": -0.003}}},
        ]
        ic_log.write_text("\n".join(json.dumps(e) for e in entries))
        learner = _make_learner(tmp)
        count = learner.seed_from_ic_log(ic_log_path=ic_log)
        assert count == 2
        assert "calm_bull" in learner._state
        assert learner._state["calm_bull"]["trend"]["mu"] == pytest.approx(0.0135, abs=0.001)


def test_ai_prior_affects_single_call_not_posterior():
    with tempfile.TemporaryDirectory() as tmp:
        state = {
            "calm_bull": {
                "trend":   {"mu": 0.010, "var": 0.003, "n": 5},
                "statarb": {"mu": 0.005, "var": 0.003, "n": 5},
            }
        }
        learner = _make_learner(tmp, state=state)
        defaults = {"trend": 0.58, "statarb": 0.15}
        w_no_prior = learner.get_weights("calm_bull", defaults)
        w_with_prior = learner.get_weights("calm_bull", defaults, ai_prior={"trend": 0.008})
        # AI prior pushed trend mu up → trend gets more weight
        assert w_with_prior["trend"] > w_no_prior["trend"]
        # Posterior unchanged — calling again without prior gives same result
        w_again = learner.get_weights("calm_bull", defaults)
        assert abs(w_again["trend"] - w_no_prior["trend"]) < 1e-6


def test_posteriors_persist_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.json"
        from ascent.alpha.meta_learner import SleeveMetaLearner
        learner = SleeveMetaLearner(posteriors_path=path)
        learner.update_rebalance("calm_bull", {"trend": 0.015})
        learner2 = SleeveMetaLearner(posteriors_path=path)
        assert "calm_bull" in learner2._state
        assert learner2._state["calm_bull"]["trend"]["n"] == 1


def test_weights_sum_to_one():
    with tempfile.TemporaryDirectory() as tmp:
        state = {
            "stressed": {
                s: {"mu": 0.01, "var": 0.003, "n": 5}
                for s in ["trend", "meanrev", "statarb", "ml", "fundamental", "earnings"]
            }
        }
        learner = _make_learner(tmp, state=state)
        defaults = {"trend": 0.35, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10,
                    "fundamental": 0.08, "earnings": 0.05}
        result = learner.get_weights("stressed", defaults)
        assert result is not None
        assert abs(sum(result.values()) - 1.0) < 0.02
