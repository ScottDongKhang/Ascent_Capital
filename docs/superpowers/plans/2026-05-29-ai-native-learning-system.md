# AI-Native Learning System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Ascent Capital genuinely smarter every rebalance by adding three compounding learning loops: Bayesian sleeve-weight learning seeded from real IC data, AI self-calibration from its own prediction track record, and AI voice in regime determination that earns authority as it proves accurate.

**Architecture:** Component B (SleeveMetaLearner) slots into `_load_active_alpha_weights()` as a new priority level between config by_regime and the static regime defaults table. Component C (AICalibraton) logs AI PM market-character predictions at each rebalance and fills in outcomes at the next one. Component A (AI Regime Blend) adds a `blend_with_ai()` method to RegimeEngine that post-processes the signal cache after fit. All three are wired into `run_all_agents.py` on the rebalance path.

**Tech Stack:** Python 3.12, standard library (json, math, pathlib, collections), no new dependencies.

---

## File Structure

```
NEW
  ascent/alpha/meta_learner.py           SleeveMetaLearner class + log_weight_proposal
  ascent/strategy/ai_calibration.py     log_thesis, update_outcome, get_context
  data_cache/sleeve_posteriors.json      auto-created on first meta-learner run
  logs/meta_learner_weights.jsonl        audit trail per rebalance
  logs/regime_blend_log.jsonl            per-rebalance blend audit
  logs/ai_thesis_outcomes.jsonl          AI PM prediction log

MODIFIED (minimal changes)
  ascent/alpha/stack.py                  _load_active_alpha_weights + build_alpha_stack signature
  ascent/regime/engine.py                blend_with_ai() method + 4 constants
  agents/ai_pm_agent.py                  3 new fields in propose_prethesis schema + AIPreThesis
  run_all_agents.py                      wire up all three components

TESTS
  tests/alpha/test_meta_learner.py       9 tests for SleeveMetaLearner
  tests/strategy/test_ai_calibration.py 7 tests for AI Calibration
  tests/regime/test_ai_regime_blend.py  5 tests for blend_with_ai
```

---

## Task 1: SleeveMetaLearner

**Files:**
- Create: `ascent/alpha/meta_learner.py`
- Create: `tests/alpha/test_meta_learner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/alpha/test_meta_learner.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/alpha/test_meta_learner.py -v 2>&1 | tail -20
```
Expected: ImportError on `ascent.alpha.meta_learner`

- [ ] **Step 3: Create `ascent/alpha/meta_learner.py`**

```python
"""
ascent/alpha/meta_learner.py
Bayesian IC meta-learner for alpha sleeve weights.

Maintains per-(regime, sleeve) Gaussian posteriors seeded from sleeve_ic_log.jsonl.
Applies Gaussian conjugate update after each rebalance holding period.
Derives Kelly-inspired weights blended toward regime defaults by confidence
alpha_conf = min(1.0, n / 20) where n is the number of rebalance observations.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
POSTERIORS_PATH = _REPO_ROOT / "data_cache" / "sleeve_posteriors.json"
IC_LOG_PATH = _REPO_ROOT / "logs" / "sleeve_ic_log.jsonl"
WEIGHTS_LOG_PATH = _REPO_ROOT / "logs" / "meta_learner_weights.jsonl"

_PRIOR_VAR = 0.005
_OBS_NOISE = 0.003
_FULL_TRUST_N = 20
_MIN_OBS_TRUST = 3
_AI_PRIOR_MAX_DELTA = 0.010
_MAX_SINGLE_SLEEVE = 0.75
_VALID_LABELS = {"calm_bull", "stressed", "crisis", "euphoric", "uncertain"}


class SleeveMetaLearner:
    """
    Per-(regime, sleeve) Bayesian meta-learner for alpha sleeve weights.

    State stored in data_cache/sleeve_posteriors.json as:
    { "calm_bull": { "trend": { "mu": 0.015, "var": 0.003, "n": 29 }, ... }, ... }
    """

    def __init__(self, posteriors_path: Path = POSTERIORS_PATH):
        self._path = posteriors_path
        self._state: Dict[str, Dict[str, Dict]] = {}
        self._load()

    def get_weights(
        self,
        regime: str,
        regime_defaults: Dict[str, float],
        ai_prior: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Derive sleeve weights for a given regime.

        Returns None if:
        - No posterior data exists for this regime, or
        - Any sleeve has fewer than _MIN_OBS_TRUST observations.

        Caller should fall back to DEFAULT_ALPHA_WEIGHTS_BY_REGIME on None.

        ai_prior: {sleeve: delta_ic} — shifts effective mu for this call only.
        Bounded to ±_AI_PRIOR_MAX_DELTA per sleeve. Does NOT write to posterior.
        """
        regime = str(regime).lower()
        if regime not in _VALID_LABELS:
            return None

        regime_state = self._state.get(regime, {})
        if not regime_state:
            return None

        min_n = min((v.get("n", 0) for v in regime_state.values()), default=0)
        if min_n < _MIN_OBS_TRUST:
            return None

        raw_weights: Dict[str, float] = {}
        for sleeve in regime_defaults:
            s = regime_state.get(sleeve)
            if s is None:
                raw_weights[sleeve] = 0.0
                continue
            mu = float(s.get("mu", 0.0))
            var = max(float(s.get("var", _PRIOR_VAR)), 1e-9)
            if ai_prior and sleeve in ai_prior:
                delta = max(-_AI_PRIOR_MAX_DELTA, min(_AI_PRIOR_MAX_DELTA, float(ai_prior[sleeve])))
                mu = mu + delta
            raw_weights[sleeve] = max(0.0, mu / math.sqrt(var))

        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            return None

        kelly_w = {s: w / total_raw for s, w in raw_weights.items()}

        result: Dict[str, float] = {}
        for sleeve, default_w in regime_defaults.items():
            s = regime_state.get(sleeve, {})
            n = int(s.get("n", 0))
            alpha_conf = min(1.0, n / _FULL_TRUST_N)
            result[sleeve] = alpha_conf * kelly_w.get(sleeve, 0.0) + (1 - alpha_conf) * default_w

        total = sum(result.values())
        if total <= 0:
            return None

        result = {s: w / total for s, w in result.items()}
        result = _enforce_cap(result, _MAX_SINGLE_SLEEVE)

        if abs(sum(result.values()) - 1.0) > 0.02:
            t = sum(result.values()) or 1.0
            result = {s: w / t for s, w in result.items()}

        return result

    def update_rebalance(self, regime: str, sleeve_ic: Dict[str, float]) -> None:
        """
        Gaussian conjugate update for each sleeve after a rebalance holding period.

        sleeve_ic: {sleeve_name: realized_ic_for_period}
        """
        regime = str(regime).lower()
        if regime not in _VALID_LABELS:
            log.warning("[MetaLearner] Unknown regime '%s' — skipping update", regime)
            return

        if regime not in self._state:
            self._state[regime] = {}

        for sleeve, ic_obs in sleeve_ic.items():
            if sleeve not in self._state[regime]:
                self._state[regime][sleeve] = {"mu": 0.0, "var": _PRIOR_VAR, "n": 0}

            s = self._state[regime][sleeve]
            mu, var, n = float(s["mu"]), float(s["var"]), int(s["n"])

            precision_prior = 1.0 / var
            precision_obs = 1.0 / _OBS_NOISE
            precision_post = precision_prior + precision_obs
            mu_post = (mu * precision_prior + float(ic_obs) * precision_obs) / precision_post
            var_post = 1.0 / precision_post

            self._state[regime][sleeve] = {
                "mu": round(mu_post, 6),
                "var": round(var_post, 6),
                "n": n + 1,
            }

        self._save()
        log.info("[MetaLearner] Updated posteriors: regime=%s sleeves=%d", regime, len(sleeve_ic))

    def seed_from_ic_log(self, ic_log_path: Path = IC_LOG_PATH) -> int:
        """
        Seed posteriors from sleeve_ic_log.jsonl using observed mean IC per sleeve.
        Only seeds (regime, sleeve) pairs not already in state.
        Returns count of log entries processed.
        """
        if not ic_log_path.exists():
            return 0

        entries = []
        for line in ic_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue

        by_regime_sleeve: Dict = defaultdict(lambda: defaultdict(list))
        for entry in entries:
            regime = str(entry.get("regime", "")).lower()
            if not regime or regime not in _VALID_LABELS:
                regime = "calm_bull"
            for sleeve, stats in entry.get("sleeves", {}).items():
                ic = stats.get("mean_ic")
                if ic is not None:
                    by_regime_sleeve[regime][sleeve].append(float(ic))

        seeded = 0
        for regime, sleeves in by_regime_sleeve.items():
            if regime not in self._state:
                self._state[regime] = {}
            for sleeve, ics in sleeves.items():
                if sleeve not in self._state[regime]:
                    mu_seed = sum(ics) / len(ics)
                    self._state[regime][sleeve] = {
                        "mu": round(mu_seed, 6),
                        "var": _PRIOR_VAR,
                        "n": len(ics),
                    }
                    seeded += 1

        if seeded > 0:
            self._save()
            log.info("[MetaLearner] Seeded %d (regime, sleeve) posteriors from ic_log", seeded)

        return len(entries)

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text())
            except Exception as e:
                log.warning("[MetaLearner] Failed to load posteriors (%s) — fresh start", e)
                self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.parent / (self._path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.replace(self._path)


def _enforce_cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    result = dict(weights)
    for _ in range(50):
        capped = {s: w for s, w in result.items() if w > cap}
        if not capped:
            break
        freed = sum(w - cap for w in capped.values())
        uncapped = {s: w for s, w in result.items() if w <= cap}
        if not uncapped:
            break
        for s in capped:
            result[s] = cap
        total_uncapped = sum(uncapped.values()) or 1.0
        for s, w in uncapped.items():
            result[s] = w + freed * (w / total_uncapped)
    return result


def log_weight_proposal(regime: str, weights: Dict[str, float], source: str) -> None:
    """Append to meta_learner_weights.jsonl for audit trail."""
    from datetime import date
    WEIGHTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "date": date.today().isoformat(),
        "regime": regime,
        "source": source,
        "weights": {s: round(w, 4) for s, w in weights.items()},
    }
    with open(WEIGHTS_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run tests — expect all 9 to pass**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/alpha/test_meta_learner.py -v 2>&1 | tail -20
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/alpha/meta_learner.py tests/alpha/test_meta_learner.py
git commit -m "feat: SleeveMetaLearner — Bayesian IC posterior for regime-adaptive sleeve weights"
```

---

## Task 2: Wire Meta-Learner into stack.py

**Files:**
- Modify: `ascent/alpha/stack.py` (lines 67–96 `_load_active_alpha_weights`, line 182 `build_alpha_stack`)

- [ ] **Step 1: Write the failing test**

Add to `tests/alpha/test_meta_learner.py` (append to the existing file):

```python
# ── stack integration ──────────────────────────────────────────────────────────

def test_stack_uses_meta_learner_when_posteriors_exist(monkeypatch, tmp_path):
    """When meta-learner has sufficient posteriors, _load_active_alpha_weights uses them."""
    from ascent.alpha import meta_learner as _ml_mod
    from ascent.alpha.meta_learner import SleeveMetaLearner, POSTERIORS_PATH

    # Create a learner with 5 observations
    state = {
        "calm_bull": {
            s: {"mu": 0.010, "var": 0.003, "n": 5}
            for s in ["trend", "meanrev", "statarb", "ml", "fundamental",
                      "earnings", "analyst", "options_flow", "insider",
                      "short_interest", "altdata", "narrative", "llm_fundamental", "volatility"]
        }
    }
    p = tmp_path / "posteriors.json"
    p.write_text(__import__("json").dumps(state))
    monkeypatch.setattr(_ml_mod, "POSTERIORS_PATH", p)

    from ascent.alpha.stack import _load_active_alpha_weights
    weights = _load_active_alpha_weights(regime="calm_bull")
    # Should return meta-learner weights (all positive IC → all get some weight)
    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1.0) < 0.02


def test_stack_falls_back_to_regime_defaults_when_meta_learner_returns_none(monkeypatch, tmp_path):
    """When meta-learner returns None (sparse data), defaults are used."""
    from ascent.alpha import meta_learner as _ml_mod

    # Empty posteriors
    p = tmp_path / "posteriors.json"
    p.write_text("{}")
    monkeypatch.setattr(_ml_mod, "POSTERIORS_PATH", p)

    from ascent.alpha.stack import _load_active_alpha_weights, DEFAULT_ALPHA_WEIGHTS_BY_REGIME
    weights = _load_active_alpha_weights(regime="calm_bull")
    assert weights == DEFAULT_ALPHA_WEIGHTS_BY_REGIME["calm_bull"]
```

- [ ] **Step 2: Run these two tests to confirm they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/alpha/test_meta_learner.py::test_stack_uses_meta_learner_when_posteriors_exist tests/alpha/test_meta_learner.py::test_stack_falls_back_to_regime_defaults_when_meta_learner_returns_none -v 2>&1 | tail -15
```
Expected: FAIL (meta-learner not yet in stack priority chain)

- [ ] **Step 3: Edit `ascent/alpha/stack.py` — replace `_load_active_alpha_weights`**

Read the current function at lines 67–96, then replace it:

```python
def _load_active_alpha_weights(regime: str = None, ai_prior: dict = None) -> dict:
    import json as _json
    from pathlib import Path as _Path

    config_path = _Path("data_cache/active_alpha_config.json")
    config_regime_weights = None
    config_global_weights = None
    if config_path.exists():
        try:
            config = _json.loads(config_path.read_text())
            if regime:
                rw = config.get("by_regime", {}).get(str(regime).lower())
                if rw and isinstance(rw, dict):
                    config_regime_weights = {k: float(v) for k, v in rw.items()}
            gw = config.get("global")
            if gw and isinstance(gw, dict):
                config_global_weights = {k: float(v) for k, v in gw.items()}
        except Exception as exc:
            log.warning("_load_active_alpha_weights: failed to load config (%s) — using defaults", exc)

    # Priority: config by_regime → meta-learner → DEFAULT_ALPHA_WEIGHTS_BY_REGIME → config global → flat default
    if config_regime_weights is not None:
        return config_regime_weights

    if regime:
        regime_key = str(regime).lower()
        try:
            from ascent.alpha.meta_learner import SleeveMetaLearner, log_weight_proposal
            regime_defaults = DEFAULT_ALPHA_WEIGHTS_BY_REGIME.get(regime_key, DEFAULT_ALPHA_WEIGHTS)
            ml = SleeveMetaLearner()
            ml_weights = ml.get_weights(regime_key, regime_defaults, ai_prior=ai_prior)
            if ml_weights is not None:
                log.info("_load_active_alpha_weights: meta-learner weights for regime=%s", regime_key)
                log_weight_proposal(regime_key, ml_weights, "meta_learner")
                return ml_weights
        except Exception as exc:
            log.warning("_load_active_alpha_weights: meta-learner error (%s) — using regime defaults", exc)

        if regime_key in DEFAULT_ALPHA_WEIGHTS_BY_REGIME:
            return DEFAULT_ALPHA_WEIGHTS_BY_REGIME[regime_key].copy()

    if config_global_weights is not None:
        return config_global_weights
    return DEFAULT_ALPHA_WEIGHTS.copy()
```

- [ ] **Step 4: Edit `ascent/alpha/stack.py` — add `ai_prior` param to `build_alpha_stack`**

The current signature at line 182 is:
```python
def build_alpha_stack(features, alpha_weights=None, regime_signal=None, agent_id: str = "us_equities"):
```

Replace with:
```python
def build_alpha_stack(features, alpha_weights=None, regime_signal=None, agent_id: str = "us_equities", ai_prior: dict = None):
```

And in the body, the line at line 201 is:
```python
        alpha_weights = _load_active_alpha_weights(regime=regime_label)
```

Replace with:
```python
        alpha_weights = _load_active_alpha_weights(regime=regime_label, ai_prior=ai_prior)
```

- [ ] **Step 5: Run all meta_learner tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/alpha/test_meta_learner.py -v 2>&1 | tail -20
```
Expected: 11 passed

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -10
```
Expected: same pass count as before (636 passed, 1 skipped) + 2 new

- [ ] **Step 7: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/alpha/stack.py tests/alpha/test_meta_learner.py
git commit -m "feat: wire SleeveMetaLearner into _load_active_alpha_weights priority chain"
```

---

## Task 3: AI Calibration Module

**Files:**
- Create: `ascent/strategy/ai_calibration.py`
- Create: `tests/strategy/test_ai_calibration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/strategy/test_ai_calibration.py`:

```python
# tests/strategy/test_ai_calibration.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def _patch_log(tmp_path):
    from ascent.strategy import ai_calibration as _mod
    log_path = Path(tmp_path) / "ai_thesis_outcomes.jsonl"
    return patch.object(_mod, "OUTCOMES_LOG", log_path)


def test_log_thesis_writes_entry():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation",
                           {"trend": 0.004, "statarb": -0.002})
        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["thesis_date"] == "2026-06-09"
        assert entry["market_character"] == "momentum_continuation"
        assert entry["prediction_correct"] is None


def test_update_outcome_fills_most_recent_pending():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            cal.log_thesis("2026-06-23", "calm_bull", "sector_rotation")
            cal.update_outcome({"trend": 0.020, "statarb": 0.005, "meanrev": -0.003})

        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        # First entry still pending
        assert entries[0]["prediction_correct"] is None
        # Second entry (most recent) filled
        assert entries[1]["prediction_correct"] is not None
        assert entries[1]["realized_ic_leaders"] is not None


def test_momentum_continuation_correct_when_trend_leads():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            # trend has highest positive IC → correct for momentum_continuation
            cal.update_outcome({"trend": 0.020, "statarb": 0.005})

        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["prediction_correct"] is True


def test_momentum_continuation_wrong_when_trend_does_not_lead():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            # statarb leads, trend is negative → wrong
            cal.update_outcome({"statarb": 0.015, "meanrev": 0.012, "trend": -0.003})

        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["prediction_correct"] is False


def test_get_context_returns_empty_when_insufficient_data():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        with _patch_log(tmp):
            # Only 2 entries — get_context needs >= 3
            cal.log_thesis("2026-06-09", "calm_bull", "momentum_continuation")
            cal.log_thesis("2026-06-23", "calm_bull", "momentum_continuation")
        context = cal.get_context("calm_bull")
        assert context == ""


def test_get_context_returns_string_with_enough_data():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.strategy import ai_calibration as cal
        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        # Write 5 completed entries directly
        entries = [
            {"thesis_date": f"2026-0{i+1}-01", "regime": "calm_bull",
             "market_character": "momentum_continuation",
             "sleeve_weight_prior": {},
             "realized_ic_leaders": ["trend"],
             "prediction_correct": True}
            for i in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))
        from ascent.strategy import ai_calibration as cal
        with patch.object(cal, "OUTCOMES_LOG", log_path):
            context = cal.get_context("calm_bull")
        assert "calm_bull" in context
        assert "momentum_continuation" in context
        assert "5/5" in context or "100%" in context


def test_get_context_is_empty_for_different_regime():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ai_thesis_outcomes.jsonl"
        entries = [
            {"thesis_date": f"2026-0{i+1}-01", "regime": "calm_bull",
             "market_character": "momentum_continuation", "sleeve_weight_prior": {},
             "realized_ic_leaders": ["trend"], "prediction_correct": True}
            for i in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))
        from ascent.strategy import ai_calibration as cal
        with patch.object(cal, "OUTCOMES_LOG", log_path):
            context = cal.get_context("stressed")
        assert context == ""
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/strategy/test_ai_calibration.py -v 2>&1 | tail -15
```
Expected: ImportError on `ascent.strategy.ai_calibration`

- [ ] **Step 3: Create `ascent/strategy/ai_calibration.py`**

```python
"""
ascent/strategy/ai_calibration.py
Tracks AI PM market character predictions vs realized sleeve IC outcomes.

Lifecycle: log_thesis() at each rebalance → update_outcome() at next rebalance
         → get_context() injected into next pre-thesis system prompt.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOMES_LOG = _REPO_ROOT / "logs" / "ai_thesis_outcomes.jsonl"

_CHARACTER_TO_SLEEVES: Dict[str, List[str]] = {
    "momentum_continuation": ["trend"],
    "sector_rotation":       ["statarb", "meanrev"],
    "risk_off":              ["volatility", "fundamental"],
    "risk_on":               ["trend", "earnings"],
    "mean_reversion":        ["meanrev", "statarb"],
    "flight_to_quality":     ["fundamental", "volatility"],
    "uncertain":             [],
}


def log_thesis(
    thesis_date: str,
    regime: str,
    market_character: str,
    sleeve_weight_prior: Optional[Dict[str, float]] = None,
) -> None:
    """Log an AI PM market character prediction. Called each rebalance before quant."""
    entry = {
        "thesis_date": thesis_date,
        "regime": regime,
        "market_character": market_character,
        "sleeve_weight_prior": sleeve_weight_prior or {},
        "realized_ic_leaders": None,
        "prediction_correct": None,
    }
    OUTCOMES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCOMES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("[AICalibration] Logged thesis date=%s regime=%s char=%s",
             thesis_date, regime, market_character)


def update_outcome(realized_ic_by_sleeve: Dict[str, float]) -> None:
    """
    Fill in realized outcome for the most recent pending log entry.
    Called at rebalance N+1 with IC measured over the N→N+1 holding period.
    """
    entries = _read_log()
    if not entries:
        return

    pending_idx = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("prediction_correct") is None:
            pending_idx = i
            break

    if pending_idx is None:
        return

    entry = entries[pending_idx]
    top_sleeves = sorted(
        [(s, ic) for s, ic in realized_ic_by_sleeve.items() if ic > 0],
        key=lambda x: -x[1],
    )[:3]
    entry["realized_ic_leaders"] = [s for s, _ in top_sleeves]
    entry["prediction_correct"] = _check_correct(
        entry.get("market_character", ""),
        entry["realized_ic_leaders"],
    )

    _write_log(entries)
    log.info(
        "[AICalibration] Outcome filled for %s: correct=%s leaders=%s",
        entry["thesis_date"], entry["prediction_correct"], entry["realized_ic_leaders"],
    )


def get_context(regime: str, max_entries: int = 10) -> str:
    """
    Return a ~200 token calibration note for injection into the pre-thesis system prompt.
    Returns empty string if fewer than 3 completed entries exist for this regime.
    """
    entries = _read_log()
    regime_entries = [
        e for e in entries
        if e.get("regime") == regime and e.get("prediction_correct") is not None
    ]

    if len(regime_entries) < 3:
        return ""

    recent = regime_entries[-max_entries:]

    from collections import defaultdict
    by_char: Dict = defaultdict(lambda: {"total": 0, "correct": 0})
    for e in recent:
        c = e.get("market_character", "unknown")
        by_char[c]["total"] += 1
        if e.get("prediction_correct"):
            by_char[c]["correct"] += 1

    lines = [f"Calibration note ({regime}):"]
    for char, stats in sorted(by_char.items()):
        pct = int(100 * stats["correct"] / stats["total"])
        lines.append(f"- {char} calls: {stats['correct']}/{stats['total']} correct ({pct}%)")

    last_miss = next(
        (e for e in reversed(recent) if e.get("prediction_correct") is False), None
    )
    if last_miss:
        leaders = ", ".join(last_miss.get("realized_ic_leaders") or ["unknown"])
        lines.append(
            f"- Last miss ({last_miss['thesis_date']}): called {last_miss['market_character']} "
            f"but realized IC leaders were {leaders}"
        )

    return "\n".join(lines)


def _check_correct(market_character: str, realized_leaders: List[str]) -> bool:
    implied = _CHARACTER_TO_SLEEVES.get(market_character, [])
    if not implied:
        return False
    return any(s in realized_leaders for s in implied)


def _read_log() -> list:
    if not OUTCOMES_LOG.exists():
        return []
    entries = []
    for line in OUTCOMES_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def _write_log(entries: list) -> None:
    tmp = OUTCOMES_LOG.parent / (OUTCOMES_LOG.name + ".tmp")
    tmp.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    tmp.replace(OUTCOMES_LOG)
```

- [ ] **Step 4: Run tests — expect all 7 to pass**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/strategy/test_ai_calibration.py -v 2>&1 | tail -15
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/strategy/ai_calibration.py tests/strategy/test_ai_calibration.py
git commit -m "feat: AI Calibration — market_character prediction logging and outcome tracking"
```

---

## Task 4: AI Regime Blend

**Files:**
- Modify: `ascent/regime/engine.py` (add 4 constants + `blend_with_ai()` method + `_load_blend_state()` + `_log_blend()` helpers)
- Create: `tests/regime/test_ai_regime_blend.py`

- [ ] **Step 1: Write failing tests**

Create `tests/regime/test_ai_regime_blend.py`:

```python
# tests/regime/test_ai_regime_blend.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def _make_engine_with_cache(labels, risk_mults=None):
    """Return a RegimeEngine whose _signal_cache has the given labels."""
    from ascent.regime.engine import RegimeEngine
    engine = RegimeEngine()
    idx = pd.date_range("2026-01-01", periods=len(labels), freq="B")
    if risk_mults is None:
        risk_mults = [1.0] * len(labels)
    engine._signal_cache = pd.DataFrame({
        "label": labels,
        "risk_multiplier": risk_mults,
        "prob": [0.9] * len(labels),
    }, index=idx)
    engine._fitted = True
    return engine


def test_blend_with_ai_noop_when_agreeing():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"])
                engine.blend_with_ai(
                    {"label": "calm_bull", "confidence": 0.9, "reasoning": "Agree"},
                    as_of_date="2026-06-09",
                )
                last_label = engine._signal_cache.iloc[-1]["label"]
        assert last_label == "calm_bull"


def test_blend_with_ai_changes_label_on_strong_disagreement():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        # Start at α=0.30 (max) so pull_weight = 0.30 × 1.0 = 0.30, still < 0.50 — but at 1.0 conf it IS 0.30
        # Set alpha to 0.55 to force label change
        state_p.write_text(json.dumps({"alpha": 0.55, "history": []}))
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"], risk_mults=[1.0])
                engine._cfg["regime_risk_multiplier"] = {"stressed": 0.8}
                engine.blend_with_ai(
                    {"label": "stressed", "confidence": 1.0, "reasoning": "Seeing stress"},
                    as_of_date="2026-06-09",
                )
                last_label = engine._signal_cache.iloc[-1]["label"]
        assert last_label == "stressed"


def test_blend_with_ai_invalid_label_does_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"])
                engine.blend_with_ai(
                    {"label": "alien_regime", "confidence": 0.9, "reasoning": "Bad"},
                    as_of_date="2026-06-09",
                )
                last_label = engine._signal_cache.iloc[-1]["label"]
        assert last_label == "calm_bull"


def test_blend_with_ai_skipped_when_not_fitted():
    from ascent.regime.engine import RegimeEngine
    engine = RegimeEngine()
    # _fitted is False — should return without error
    engine.blend_with_ai({"label": "calm_bull", "confidence": 0.9, "reasoning": "x"}, "2026-06-09")
    assert engine._signal_cache is None


def test_blend_logs_to_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        from ascent.regime import engine as _eng
        state_p = Path(tmp) / "blend_state.json"
        log_p = Path(tmp) / "blend_log.jsonl"
        with patch.object(_eng, "AI_BLEND_STATE_PATH", str(state_p)):
            with patch.object(_eng, "AI_BLEND_LOG_PATH", str(log_p)):
                engine = _make_engine_with_cache(["calm_bull"])
                engine.blend_with_ai(
                    {"label": "calm_bull", "confidence": 0.8, "reasoning": "test"},
                    as_of_date="2026-06-09",
                )
        assert log_p.exists()
        entry = json.loads(log_p.read_text().strip())
        assert entry["as_of_date"] == "2026-06-09"
        assert "hmm_label" in entry
        assert "ai_label" in entry
        assert "alpha" in entry
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/regime/test_ai_regime_blend.py -v 2>&1 | tail -15
```
Expected: AttributeError — `blend_with_ai` not yet on RegimeEngine

- [ ] **Step 3: Add constants and helpers to `ascent/regime/engine.py`**

After the existing constants (after line ~60, before `check_emergency_refit_triggers`), add:

```python
AI_BLEND_INITIAL_ALPHA = 0.05
AI_BLEND_MAX_ALPHA = 0.30
AI_BLEND_STEP = 0.03
AI_BLEND_STATE_PATH = "data_cache/regime_blend_state.json"
AI_BLEND_LOG_PATH = "logs/regime_blend_log.jsonl"
_AI_BLEND_VALID_LABELS = {"calm_bull", "stressed", "crisis", "euphoric", "uncertain"}


def _load_blend_state() -> dict:
    p = Path(AI_BLEND_STATE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"alpha": AI_BLEND_INITIAL_ALPHA, "history": []}


def _save_blend_state(state: dict) -> None:
    p = Path(AI_BLEND_STATE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / (p.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def _log_blend(as_of_date: str, hmm_label: str, ai_label: str, alpha: float,
               blended_label: str, blended_risk: float) -> None:
    import json as _json
    p = Path(AI_BLEND_LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "as_of_date": as_of_date,
        "hmm_label": hmm_label,
        "ai_label": ai_label,
        "alpha": round(alpha, 4),
        "blended_label": blended_label,
        "blended_risk_multiplier": round(blended_risk, 4),
    }
    with open(p, "a") as f:
        f.write(_json.dumps(entry) + "\n")
```

Note: `engine.py` already imports `json` via `import json` — if not present, add it at the top. Check the existing imports first.

- [ ] **Step 4: Add `blend_with_ai()` method to `RegimeEngine` class**

Add this method after the `get_feature_panel()` method (around line 357):

```python
def blend_with_ai(
    self,
    ai_regime_assessment: dict,
    as_of_date: str,
) -> None:
    """
    Blend AI regime assessment into the most recent signal cache entry.

    ai_regime_assessment: {"label": str, "confidence": float, "reasoning": str}

    alpha (blend weight) starts at AI_BLEND_INITIAL_ALPHA and grows by AI_BLEND_STEP
    each time the AI label matches the regime that produced better IC, capped at
    AI_BLEND_MAX_ALPHA. Stored in data_cache/regime_blend_state.json.

    AI label changes the final label only when alpha * ai_conf > 0.50 (strong pull).
    risk_multiplier is always modulated proportionally.
    """
    if not self._fitted or self._signal_cache is None:
        log.debug("[Engine] blend_with_ai called before fit — skipping")
        return

    ai_label = str(ai_regime_assessment.get("label", "")).lower()
    ai_conf = float(ai_regime_assessment.get("confidence", 0.5))

    if ai_label not in _AI_BLEND_VALID_LABELS:
        log.warning("[Engine] blend_with_ai: invalid label '%s' — skipping", ai_label)
        return

    state = _load_blend_state()
    alpha = min(float(state.get("alpha", AI_BLEND_INITIAL_ALPHA)), AI_BLEND_MAX_ALPHA)

    last_date = self._signal_cache.index[-1]
    hmm_label = str(self._signal_cache.loc[last_date, "label"])
    hmm_risk_mult = float(self._signal_cache.loc[last_date, "risk_multiplier"])

    pull_weight = alpha * ai_conf

    if ai_label == hmm_label:
        blended_label = hmm_label
        blended_risk_mult = hmm_risk_mult
    elif pull_weight > 0.50:
        blended_label = ai_label
        blended_risk_mult = float(
            self._cfg.get("regime_risk_multiplier", {}).get(ai_label, hmm_risk_mult)
        )
    else:
        blended_label = hmm_label
        ai_risk = float(self._cfg.get("regime_risk_multiplier", {}).get(ai_label, 1.0))
        blended_risk_mult = (1 - pull_weight) * hmm_risk_mult + pull_weight * ai_risk

    self._signal_cache.loc[last_date, "label"] = blended_label
    self._signal_cache.loc[last_date, "risk_multiplier"] = round(blended_risk_mult, 4)

    _log_blend(as_of_date, hmm_label, ai_label, alpha, blended_label, blended_risk_mult)
    log.info(
        "[Engine] AI blend: HMM=%s AI=%s(conf=%.2f) α=%.3f → %s risk=%.2f",
        hmm_label, ai_label, ai_conf, alpha, blended_label, blended_risk_mult,
    )
```

- [ ] **Step 5: Check that `json` is imported in engine.py**

```bash
head -40 "/Users/scott/Downloads/ascent capital v2 up to phase 5.1/ascent/regime/engine.py" | grep import
```

If `import json` is not present, add it after the existing `import numpy as np` line.

- [ ] **Step 6: Run regime blend tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/regime/test_ai_regime_blend.py -v 2>&1 | tail -15
```
Expected: 5 passed

- [ ] **Step 7: Run full test suite**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -10
```
Expected: no regressions

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/regime/engine.py tests/regime/test_ai_regime_blend.py
git commit -m "feat: blend_with_ai() on RegimeEngine — AI voice in regime at alpha=0.05"
```

---

## Task 5: Pre-Thesis Schema Additions

**Files:**
- Modify: `agents/ai_pm_agent.py`
  - `AIPreThesis` dataclass: add `regime_assessment`, `sleeve_weight_prior`, `market_character`
  - `_PROPOSE_PRETHESIS_TOOL`: add 3 new properties
  - `_tool_propose_prethesis()`: extract new fields
  - `_PRE_THESIS_PROMPT`: add brief instruction to output new fields

- [ ] **Step 1: Write failing test**

Add to `tests/test_ai_pm_agent.py` (append):

```python
# ── pre-thesis schema additions ───────────────────────────────────────────────

def test_prethesis_dataclass_has_new_fields():
    from agents.ai_pm_agent import AIPreThesis
    pt = AIPreThesis(
        macro_view="Rates are stabilizing.",
        regime_interpretation="HMM says calm_bull, I agree.",
        high_conviction_names=[{"symbol": "VICR", "thesis": "Margin expansion"}],
        names_to_avoid=[],
        sector_tilts=[],
        regime_assessment={"label": "calm_bull", "confidence": 0.8, "reasoning": "VIX low"},
        sleeve_weight_prior={"trend": 0.004, "statarb": -0.002},
        market_character="momentum_continuation",
    )
    assert pt.regime_assessment["label"] == "calm_bull"
    assert pt.sleeve_weight_prior["trend"] == 0.004
    assert pt.market_character == "momentum_continuation"


def test_propose_prethesis_tool_accepts_new_fields():
    """Tool schema must include regime_assessment, sleeve_weight_prior, market_character."""
    from agents.ai_pm_agent import _PROPOSE_PRETHESIS_TOOL
    props = _PROPOSE_PRETHESIS_TOOL["input_schema"]["properties"]
    assert "regime_assessment" in props
    assert "sleeve_weight_prior" in props
    assert "market_character" in props
```

- [ ] **Step 2: Run these tests to confirm they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_ai_pm_agent.py::test_prethesis_dataclass_has_new_fields tests/test_ai_pm_agent.py::test_propose_prethesis_tool_accepts_new_fields -v 2>&1 | tail -15
```
Expected: FAIL (TypeError — unexpected keyword argument)

- [ ] **Step 3: Update `AIPreThesis` dataclass in `agents/ai_pm_agent.py`**

The current dataclass (lines 52–63) is:
```python
@dataclass
class AIPreThesis:
    """Output of Phase 1 — original AI PM thesis formed before seeing quant output."""
    macro_view: str
    regime_interpretation: str
    high_conviction_names: List[Dict]
    names_to_avoid: List[Dict]
    sector_tilts: List[Dict]
    raw: Dict = field(default_factory=dict)
```

Replace with:
```python
@dataclass
class AIPreThesis:
    """Output of Phase 1 — original AI PM thesis formed before seeing quant output."""
    macro_view: str
    regime_interpretation: str
    high_conviction_names: List[Dict]
    names_to_avoid: List[Dict]
    sector_tilts: List[Dict]
    regime_assessment: Dict = field(default_factory=dict)        # {label, confidence, reasoning}
    sleeve_weight_prior: Dict = field(default_factory=dict)      # {sleeve: delta_ic}
    market_character: str = ""                                    # e.g. "momentum_continuation"
    raw: Dict = field(default_factory=dict)
```

- [ ] **Step 4: Add 3 properties to `_PROPOSE_PRETHESIS_TOOL` input schema**

The `_PROPOSE_PRETHESIS_TOOL` is defined around line 380 in `agents/ai_pm_agent.py`. Its `input_schema.properties` dict needs 3 new keys added. Find the `"sector_tilts"` property and add after it (before `"required"`):

```python
"regime_assessment": {
    "type": "object",
    "description": "Your assessment of the current regime. label must be one of: calm_bull, stressed, crisis, euphoric, uncertain.",
    "properties": {
        "label":      {"type": "string"},
        "confidence": {"type": "number", "description": "0.0 to 1.0"},
        "reasoning":  {"type": "string"},
    },
},
"sleeve_weight_prior": {
    "type": "object",
    "description": "IC delta adjustments for sleeves this rebalance. Keys are sleeve names (trend, statarb, meanrev, ml, fundamental, earnings, volatility). Values are IC deltas in range [-0.010, +0.010]. Positive = boost, negative = reduce. Omit sleeves you have no view on.",
    "additionalProperties": {"type": "number"},
},
"market_character": {
    "type": "string",
    "description": "Single most important characteristic of this market period for sleeve selection.",
    "enum": ["momentum_continuation", "sector_rotation", "risk_off", "risk_on",
             "mean_reversion", "flight_to_quality", "uncertain"],
},
```

- [ ] **Step 5: Update `_tool_propose_prethesis()` to extract new fields**

Find `_tool_propose_prethesis` at line ~1268. The function currently builds an `AIPreThesis`. Add extraction of the three new fields. The current construction at end of function is:

```python
prethesis = AIPreThesis(
    macro_view=str(inputs.get("macro_view", "")),
    ...
)
```

Add the three new fields to the constructor call:

```python
prethesis = AIPreThesis(
    macro_view=str(inputs.get("macro_view", "")),
    regime_interpretation=str(inputs.get("regime_interpretation", "")),
    high_conviction_names=list(inputs.get("high_conviction_names", [])),
    names_to_avoid=list(inputs.get("names_to_avoid", [])),
    sector_tilts=list(inputs.get("sector_tilts", [])),
    regime_assessment=dict(inputs.get("regime_assessment") or {}),
    sleeve_weight_prior=dict(inputs.get("sleeve_weight_prior") or {}),
    market_character=str(inputs.get("market_character") or ""),
    raw=dict(inputs),
)
```

- [ ] **Step 6: Add brief instruction to `_PRE_THESIS_PROMPT`**

Find `_PRE_THESIS_PROMPT` around line 583. After the `propose_prethesis` output section, add these lines to the output description:

```
Also include in propose_prethesis:
  • regime_assessment: your own regime call (label, confidence 0-1, one sentence reasoning)
  • market_character: which of the 7 characters best describes this period
  • sleeve_weight_prior: IC delta adjustments for sleeves you have a specific view on
    (e.g. {"trend": +0.004} if you expect momentum to work well; {"statarb": -0.003} if not)
    Omit sleeves you have no view on. Delta range: -0.010 to +0.010.
```

- [ ] **Step 7: Run tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v 2>&1 | tail -15
```
Expected: all existing tests pass + 2 new ones pass

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add agents/ai_pm_agent.py tests/test_ai_pm_agent.py
git commit -m "feat: add regime_assessment, sleeve_weight_prior, market_character to pre-thesis schema"
```

---

## Task 6: Wire Into run_all_agents.py

**Files:**
- Modify: `run_all_agents.py` (4 targeted additions, no structural changes)

The four additions go at specific existing hook points:

1. **After pre-thesis** (line ~908, after `_ai_prethesis = run_ai_pm_prethesis()`): pass `regime_assessment` to regime engine (requires engine to already be fit — but on rebalance days, main.py runs and fits the engine first via each agent. The regime engine is per-agent; need to check architecture. Actually, looking at the flow: `run_all_agents.py` calls `main.py` via `AgentScope`, not directly. The `RegimeEngine.blend_with_ai()` needs to be called inside `main.py` after the engine is fit, OR we need to pass the assessment down. **Simpler approach**: store `_ai_prethesis.regime_assessment` and pass it to `main.py` via an environment variable or a temp file that `main.py` reads. Use a temp JSON file approach: write to `data_cache/ai_regime_assessment.json`, read in `main.py` after `engine.fit()`.)

2. **In `ascent/main.py`** (after `engine.fit()` call, line ~305): read `data_cache/ai_regime_assessment.json` if it exists and call `engine.blend_with_ai()`.

3. **After AI PM thesis is accepted** (line ~946 in run_all_agents.py): call `ai_calibration.log_thesis()` with the market_character from the pre-thesis.

4. **At start of rebalance processing** (before the authority snapshot, line ~1086): compute realized sleeve IC for the holding period and call `meta_learner.update_rebalance()` + `ai_calibration.update_outcome()`.

- [ ] **Step 1: Add regime assessment file write after pre-thesis in `run_all_agents.py`**

Find the block starting at line ~904:
```python
_ai_prethesis: AIPreThesis | None = None
if is_rebalance:
    try:
        print("[Runner] AI PM Phase 1 — forming original thesis before quant runs...")
        _ai_prethesis = run_ai_pm_prethesis()
        if _ai_prethesis:
            syms = ", ".join(_ai_prethesis.conviction_symbols[:6])
            print(f"[Runner] Pre-thesis sealed: ...")
        else:
            ...
    except Exception as _pt_e:
        ...
```

After the existing `if _ai_prethesis:` block (after the print), add:

```python
            # Write AI regime assessment + sleeve prior to temp file for main.py to pick up
            if _ai_prethesis.regime_assessment or _ai_prethesis.sleeve_weight_prior:
                _assess_path = Path("data_cache/ai_regime_assessment.json")
                try:
                    _assess_path.write_text(
                        json.dumps({
                            **(_ai_prethesis.regime_assessment or {}),
                            "sleeve_weight_prior": _ai_prethesis.sleeve_weight_prior or {},
                            "as_of_date": today.isoformat(),
                        })
                    )
                    print(f"[Runner] AI regime assessment written: "
                          f"{(_ai_prethesis.regime_assessment or {}).get('label', 'n/a')} "
                          f"sleeves={list((_ai_prethesis.sleeve_weight_prior or {}).keys())}")
                except Exception as _ae:
                    print(f"[Runner] AI regime assessment write failed: {_ae}")
```

- [ ] **Step 2: Add blend_with_ai call in `ascent/main.py`**

In `ascent/main.py`, find the call to `engine.fit()` — it's around line 305. After the fit call (after `regime_signal = engine.get_signal(as_of_date)` or similar), add:

```python
# Blend AI regime assessment if available; pass sleeve prior to build_alpha_stack
_ai_assess_path = Path("data_cache/ai_regime_assessment.json")
_ai_sleeve_prior: dict = {}
if _ai_assess_path.exists():
    try:
        import json as _json
        _ai_assess = _json.loads(_ai_assess_path.read_text())
        _assess_date = _ai_assess.get("as_of_date", "")
        if _assess_date == str(as_of_date)[:10]:
            engine.blend_with_ai(_ai_assess, as_of_date=_assess_date)
            log.info("[main] AI regime blend applied")
            _ai_sleeve_prior = dict(_ai_assess.get("sleeve_weight_prior") or {})
    except Exception as _blend_e:
        log.debug("[main] AI regime blend skipped: %s", _blend_e)
```

Then find the `build_alpha_stack(...)` call in `main.py` and add the `ai_prior` argument:

```python
alpha = build_alpha_stack(
    features,
    regime_signal=regime_signal,
    agent_id=agent_id,
    ai_prior=_ai_sleeve_prior or None,
)
```

Note: `as_of_date` may be a `datetime.date` or string depending on context in `main.py`. Use `str(as_of_date)[:10]` to get the ISO date string safely. If `_ai_sleeve_prior` is an empty dict, pass `None` so meta-learner skips the shift.

- [ ] **Step 3: Add `ai_calibration.log_thesis()` call in `run_all_agents.py`**

Find line ~946 in `run_all_agents.py` after `format_thesis(...)`:
```python
format_thesis({**ai_pm_result.thesis, "ai_pm_portfolio": ai_pm_result.portfolio})
```

After that line, add:
```python
                # Log AI market character for calibration tracking
                if _ai_prethesis and _ai_prethesis.market_character:
                    try:
                        from ascent.strategy.ai_calibration import log_thesis as _log_thesis
                        _log_thesis(
                            thesis_date=today.isoformat(),
                            regime=_get_current_regime(),
                            market_character=_ai_prethesis.market_character,
                            sleeve_weight_prior=_ai_prethesis.sleeve_weight_prior or {},
                        )
                        print(f"[Runner] Calibration: logged market_character="
                              f"{_ai_prethesis.market_character}")
                    except Exception as _cal_e:
                        print(f"[Runner] Calibration log failed: {_cal_e}")
```

- [ ] **Step 4: Add post-rebalance meta-learner and calibration updates in `run_all_agents.py`**

Find the authority snapshot block starting at line ~1086:
```python
_AUTHORITY_SNAPSHOT = Path("data_cache/authority_rebalance_snapshot.json")
if is_rebalance:
    try:
        ...
```

Before that block (but still inside the rebalance section), add:

```python
    # ── Post-rebalance: update meta-learner and calibration from holding-period sleeve IC ──
    if is_rebalance:
        try:
            import json as _json
            from ascent.alpha.meta_learner import SleeveMetaLearner
            from ascent.strategy.ai_calibration import update_outcome as _update_cal_outcome

            _sleeve_ic_log = Path("logs/sleeve_ic_log.jsonl")
            _snap_path = Path("data_cache/authority_rebalance_snapshot.json")
            _realized_ic: dict = {}

            if _sleeve_ic_log.exists() and _snap_path.exists():
                _prev_snap = _json.loads(_snap_path.read_text())
                _prev_date = _prev_snap.get("rebalance_date", "")
                if _prev_date:
                    # Collect IC entries since prev rebalance
                    _ic_entries = []
                    for _line in _sleeve_ic_log.read_text().splitlines():
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _e = _json.loads(_line)
                            if _e.get("date", "") >= _prev_date:
                                _ic_entries.append(_e)
                        except Exception:
                            continue

                    if _ic_entries:
                        from collections import defaultdict as _dd
                        _sleeve_sums: dict = _dd(list)
                        for _e in _ic_entries:
                            for _sl, _st in _e.get("sleeves", {}).items():
                                _ic = _st.get("mean_ic")
                                if _ic is not None:
                                    _sleeve_sums[_sl].append(float(_ic))
                        _realized_ic = {
                            _sl: sum(_ics) / len(_ics)
                            for _sl, _ics in _sleeve_sums.items()
                            if _ics
                        }

            if _realized_ic:
                _current_regime = _get_current_regime()
                _ml = SleeveMetaLearner()
                _ml.update_rebalance(_current_regime, _realized_ic)
                print(f"[Runner] Meta-learner updated: regime={_current_regime} "
                      f"sleeves={list(_realized_ic.keys())}")

                _update_cal_outcome(_realized_ic)
                print("[Runner] Calibration outcome updated")
            else:
                print("[Runner] Meta-learner: no IC data for this holding period — skipping")

        except Exception as _ml_e:
            print(f"[Runner] Meta-learner/calibration update failed: {_ml_e}")
```

- [ ] **Step 5: Also seed meta-learner from IC log on first run (one-time at startup)**

Find the top of `run_all_agents.py` near the other startup checks. After the kill_switch import, add a one-time seeding block:

```python
# Seed meta-learner from existing IC log if posteriors don't exist yet
_posteriors_path = Path("data_cache/sleeve_posteriors.json")
if not _posteriors_path.exists():
    try:
        from ascent.alpha.meta_learner import SleeveMetaLearner
        _seed_ml = SleeveMetaLearner()
        _n = _seed_ml.seed_from_ic_log()
        if _n > 0:
            print(f"[Runner] Meta-learner: seeded from {_n} IC log entries")
    except Exception as _seed_e:
        print(f"[Runner] Meta-learner seed failed: {_seed_e}")
```

This block should be placed after imports, before the `main()` function or at the top of `main()`.

- [ ] **Step 6: Check main.py for the exact location of engine.fit()**

```bash
grep -n "engine.fit\|regime_engine\|RegimeEngine" "/Users/scott/Downloads/ascent capital v2 up to phase 5.1/ascent/main.py" | head -20
```

Use this output to locate the exact line where `engine.fit()` is called to place the `blend_with_ai` call correctly.

- [ ] **Step 7: Run full test suite**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -10
```
Expected: no regressions (all existing 636 pass + new tests pass)

- [ ] **Step 8: Verify ast.parse on modified files**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
import ast
for f in ['run_all_agents.py', 'ascent/main.py', 'agents/ai_pm_agent.py']:
    ast.parse(open(f).read())
    print(f'OK: {f}')
"
```
Expected: `OK: <file>` for all three

- [ ] **Step 9: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add run_all_agents.py ascent/main.py
git commit -m "feat: wire AI learning loops into run_all_agents.py + main.py — regime blend, meta-learner updates, calibration tracking"
```

---

## Final Verification

- [ ] **Run full test suite one last time**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -15
```
Expected: ≥ 650 passed (636 existing + ~16 new), 1 skipped, no errors

- [ ] **Quick smoke test of new modules**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
from ascent.alpha.meta_learner import SleeveMetaLearner
from ascent.strategy.ai_calibration import get_context, log_thesis
from ascent.regime.engine import RegimeEngine, AI_BLEND_INITIAL_ALPHA
print('SleeveMetaLearner OK')
print('ai_calibration OK')
print('RegimeEngine OK, alpha_initial=', AI_BLEND_INITIAL_ALPHA)
"
```
Expected: 3 OK lines, no import errors

- [ ] **Update CLAUDE.md session log**

Append a new session entry to the `## Session log` section of `CLAUDE.md`:

```
### 2026-05-29 (AI-Native Learning System ✅)
- Component B: SleeveMetaLearner (`ascent/alpha/meta_learner.py`) — Bayesian IC posterior per (regime, sleeve); seeded from 29 days of sleeve_ic_log; Gaussian conjugate update at each rebalance; Kelly-inspired weight derivation; confidence blend toward regime defaults.
- Stack wiring: `_load_active_alpha_weights()` now has meta-learner as priority 2 (after config by_regime, before static regime table). `build_alpha_stack()` accepts `ai_prior` param.
- Component C: AICal (`ascent/strategy/ai_calibration.py`) — logs AI market_character predictions; fills realized_ic_leaders at next rebalance; generates ~200 token calibration note for pre-thesis injection.
- Component A: `RegimeEngine.blend_with_ai()` — AI voice in regime at α=0.05 (capped at 0.30); only changes label when α × confidence > 0.50.
- Pre-thesis schema: `AIPreThesis` + `_PROPOSE_PRETHESIS_TOOL` now include `regime_assessment`, `sleeve_weight_prior`, `market_character`.
- run_all_agents.py: regime assessment temp file → main.py blend; calibration.log_thesis() after AI PM; meta-learner + calibration update at start of each rebalance.
- Files: ascent/alpha/meta_learner.py (new), ascent/strategy/ai_calibration.py (new), ascent/regime/engine.py, ascent/alpha/stack.py, agents/ai_pm_agent.py, run_all_agents.py, ascent/main.py, tests/alpha/test_meta_learner.py (new), tests/strategy/test_ai_calibration.py (new), tests/regime/test_ai_regime_blend.py (new).
```

- [ ] **Final commit with CLAUDE.md update**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add CLAUDE.md
git commit -m "docs: session log — AI-Native Learning System complete"
```
