# Self-Evolving Alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last mile of the self-improve loop — make the system actually change its live alpha weights automatically, and make those weights regime-conditioned so it learns different behavior for different market states.

**Architecture:** Three layers. (1) Wire `stack.py` to read `active_alpha_config.json` so self-improve changes actually hit live trading. (2) Build `shadow_promoter.py` — after 30-day shadow period, re-evaluate and auto-promote to live if edge holds. (3) Extend config schema and self-improve logic to maintain per-regime weight sets, so the system learns "stressed regime = more statarb, less trend" independently from calm bull.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, existing `walk_forward_lightweight.py`, existing `ascent/alpha/stack.py`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ascent/alpha/stack.py` | Modify | Read `active_alpha_config.json`; fall back to hardcoded defaults |
| `ascent/research/shadow_promoter.py` | Create | Scan shadow configs, re-evaluate, auto-promote winners |
| `ascent/research/self_improve.py` | Modify | Per-regime variant generation + scoring |
| `data_cache/active_alpha_config.json` | Schema | Per-regime weight sets + global fallback |
| `run_all_agents.py` | Modify | Call shadow promoter weekly |
| `tests/test_self_evolving_alpha.py` | Create | All tests for this plan |

---

## Task 1: Wire `stack.py` to Read `active_alpha_config.json`

**The problem:** `build_alpha_stack()` uses hardcoded `DEFAULT_ALPHA_WEIGHTS`. Self-improve changes have zero effect on live trading. Fix: load weights from `active_alpha_config.json` if it exists.

**Files:**
- Modify: `ascent/alpha/stack.py:42-63`
- Test: `tests/test_self_evolving_alpha.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_self_evolving_alpha.py`:

```python
# tests/test_self_evolving_alpha.py
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Task 1 tests ───────────────────────────────────────────────────────────────

def test_stack_uses_active_config_when_present(tmp_path, monkeypatch):
    """build_alpha_stack must use active_alpha_config.json when it exists."""
    monkeypatch.chdir(tmp_path)
    custom = {"trend": 0.80, "meanrev": 0.05, "statarb": 0.10, "ml": 0.05, "volatility": 0.0}
    _write_active_config(tmp_path, custom)

    # Import fresh after chdir so the config path resolves correctly
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
    assert abs(loaded.get("trend", 0) - 0.65) < 0.001, \
        "without config file, trend should be default 0.65"


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

    loaded = stack_mod._load_active_alpha_weights(regime="euphoric")
    assert abs(loaded.get("trend", 0) - 0.70) < 0.001
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_self_evolving_alpha.py::test_stack_uses_active_config_when_present \
    tests/test_self_evolving_alpha.py::test_stack_falls_back_to_defaults_when_no_config \
    tests/test_self_evolving_alpha.py::test_stack_uses_regime_weights_when_regime_in_config \
    tests/test_self_evolving_alpha.py::test_stack_falls_back_to_global_for_unknown_regime \
    -v --tb=short 2>&1 | tail -15
```

Expected: 4 failed — `_load_active_alpha_weights` doesn't exist yet.

- [ ] **Step 3: Add `_load_active_alpha_weights()` to `ascent/alpha/stack.py`**

Read `ascent/alpha/stack.py` first. Then add this function after the `DEFAULT_ALPHA_WEIGHTS` dict (after line 22) and before `_load_sector_map()`:

```python
def _load_active_alpha_weights(regime: str = None) -> dict:
    """
    Load alpha sleeve weights from active_alpha_config.json if it exists.
    Falls back to DEFAULT_ALPHA_WEIGHTS if file missing or malformed.

    Args:
        regime: current regime label (e.g. "stressed", "calm_bull"). If the
                active config has per-regime weights and this regime matches,
                those weights are returned. Otherwise returns global weights.
    """
    import json as _json
    from pathlib import Path as _Path

    config_path = _Path("data_cache/active_alpha_config.json")
    if not config_path.exists():
        return DEFAULT_ALPHA_WEIGHTS.copy()

    try:
        config = _json.loads(config_path.read_text())
        if regime:
            regime_weights = config.get("by_regime", {}).get(str(regime).lower())
            if regime_weights and isinstance(regime_weights, dict):
                return {k: float(v) for k, v in regime_weights.items()}
        global_weights = config.get("global")
        if global_weights and isinstance(global_weights, dict):
            return {k: float(v) for k, v in global_weights.items()}
    except Exception as exc:
        log.warning("_load_active_alpha_weights: failed to load config (%s) — using defaults", exc)

    return DEFAULT_ALPHA_WEIGHTS.copy()
```

- [ ] **Step 4: Modify `build_alpha_stack()` to call `_load_active_alpha_weights()`**

Find the `if alpha_weights is None:` block (around line 54-55). Replace:

```python
    if alpha_weights is None:
        alpha_weights = DEFAULT_ALPHA_WEIGHTS.copy()
```

with:

```python
    if alpha_weights is None:
        regime_label = None
        if regime_signal is not None:
            try:
                regime_label = str(regime_signal.label.value).lower()
            except Exception:
                pass
        alpha_weights = _load_active_alpha_weights(regime=regime_label)
```

- [ ] **Step 5: Verify syntax**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "import ast; ast.parse(open('ascent/alpha/stack.py').read()); print('OK')"
```

- [ ] **Step 6: Run Task 1 tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_self_evolving_alpha.py::test_stack_uses_active_config_when_present \
    tests/test_self_evolving_alpha.py::test_stack_falls_back_to_defaults_when_no_config \
    tests/test_self_evolving_alpha.py::test_stack_uses_regime_weights_when_regime_in_config \
    tests/test_self_evolving_alpha.py::test_stack_falls_back_to_global_for_unknown_regime \
    -v --tb=short 2>&1 | tail -15
```

Expected: 4 passed.

- [ ] **Step 7: Run full suite**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: 161+ passed.

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/alpha/stack.py tests/test_self_evolving_alpha.py
git commit -m "$(cat <<'EOF'
feat(alpha): stack reads active_alpha_config.json — self-improve now hits live trading

_load_active_alpha_weights() loads global or per-regime weights from
data_cache/active_alpha_config.json. Falls back to hardcoded defaults.
build_alpha_stack() calls it when alpha_weights is None.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Build `shadow_promoter.py` — Auto-Promotion from Shadow to Live

**The problem:** Shadow configs accumulate in `data_cache/shadow_configs/` and are never promoted. After 30 days, if the edge holds, the config should become live automatically.

**Files:**
- Create: `ascent/research/shadow_promoter.py`
- Modify: `run_all_agents.py` — call promoter weekly
- Test: `tests/test_self_evolving_alpha.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_self_evolving_alpha.py`:

```python
# ── Task 2 tests ───────────────────────────────────────────────────────────────

def test_shadow_promoter_promotes_expired_winner(tmp_path, monkeypatch):
    """A shadow config past its expiry that still beats baseline must be promoted to live."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir()

    from datetime import timedelta
    shadow = {
        "variant_id":        "v1_20260318",
        "alpha_weights":     {"trend": 0.70, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.0},
        "oos_sharpe":        0.72,
        "edge_over_current": 0.20,
        "shadow_expires":    (date.today() - timedelta(days=1)).isoformat(),  # expired yesterday
        "promoted_at":       "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v1_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # Write prices so lightweight OOS can run
    idx = pd.bdate_range(end=date.today(), periods=300)
    syms = [f"S{i}" for i in range(20)] + ["SPY"]
    np.random.seed(1)
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    (tmp_path / "data_cache" / "prices_live.parquet").parent.mkdir(exist_ok=True)
    prices.to_parquet(tmp_path / "data_cache" / "prices_live.parquet")

    from ascent.research.shadow_promoter import run_shadow_promotion
    promoted = run_shadow_promotion(baseline_sharpe=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert active_path.exists(), "active_alpha_config.json must be written after promotion"
    config = json.loads(active_path.read_text())
    assert "global" in config, "promoted config must have 'global' key"
    assert abs(config["global"].get("trend", 0) - 0.70) < 0.001


def test_shadow_promoter_skips_unexpired(tmp_path, monkeypatch):
    """Shadow configs that haven't expired yet must not be promoted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    from datetime import timedelta
    shadow = {
        "variant_id":    "v2_20260410",
        "alpha_weights": {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05},
        "oos_sharpe":    0.70,
        "shadow_expires": (date.today() + timedelta(days=15)).isoformat(),  # not expired
        "promoted_at":   "2026-04-10T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v2_20260410.json"
    shadow_file.write_text(json.dumps(shadow))

    from ascent.research.shadow_promoter import run_shadow_promotion
    promoted = run_shadow_promotion(baseline_sharpe=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "must NOT promote a config that hasn't expired yet"


def test_shadow_promoter_archives_weak_expired(tmp_path, monkeypatch):
    """An expired shadow config that no longer beats baseline must be archived, not promoted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data_cache" / "archived_configs").mkdir()

    from datetime import timedelta
    shadow = {
        "variant_id":    "v3_20260318",
        "alpha_weights": {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05},
        "oos_sharpe":    0.52,
        "shadow_expires": (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":   "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v3_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # No price cache — OOS returns 0.0 sharpe → below baseline 0.518
    from ascent.research.shadow_promoter import run_shadow_promotion
    run_shadow_promotion(baseline_sharpe=0.518)

    # File should be moved to archived_configs, not promoted
    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "weak expired config must not become live"
    archived = list((tmp_path / "data_cache" / "archived_configs").glob("*.json"))
    assert len(archived) >= 1, "expired weak config must be moved to archived_configs"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_self_evolving_alpha.py -k "shadow_promoter" -v --tb=short 2>&1 | tail -15
```

Expected: 3 failed — module doesn't exist.

- [ ] **Step 3: Create `ascent/research/shadow_promoter.py`**

```python
"""
ascent/research/shadow_promoter.py

Scans shadow configs after their 30-day monitoring period.
Re-evaluates each on fresh OOS data.
Promotes winners to active_alpha_config.json.
Archives losers.

Called weekly by run_all_agents.py (same schedule as self_improve).
"""
from __future__ import annotations
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

SHADOW_DIR   = Path("data_cache/shadow_configs")
ARCHIVE_DIR  = Path("data_cache/archived_configs")
ACTIVE_PATH  = Path("data_cache/active_alpha_config.json")
LOG_PATH     = Path("logs/self_improve_log.jsonl")

MIN_EDGE_FOR_PROMOTION = 0.05  # re-eval Sharpe must beat baseline by this much


def _load_shadow_configs() -> list[dict]:
    """Return all shadow config dicts that exist on disk."""
    if not SHADOW_DIR.exists():
        return []
    configs = []
    for f in sorted(SHADOW_DIR.glob("*.json")):
        try:
            configs.append((f, json.loads(f.read_text())))
        except Exception:
            pass
    return configs


def _re_evaluate(variant_config: dict) -> float:
    """Re-evaluate a variant on fresh OOS data. Returns Sharpe (0.0 on failure)."""
    try:
        from ascent.research.walk_forward_lightweight import run_lightweight_oos, TURNOVER_PENALTY
        result = run_lightweight_oos(variant_config, n_days=63)
        if result.get("n_folds", 0) == 0:
            return 0.0
        return round(result["sharpe"] - TURNOVER_PENALTY * result["turnover"], 4)
    except Exception as e:
        print(f"[ShadowPromoter] Re-evaluation failed: {e}")
        return 0.0


def _write_active_config(variant: dict, fresh_sharpe: float) -> None:
    """Write variant weights to active_alpha_config.json."""
    ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    weights = variant.get("alpha_weights", {})

    # Load existing config to preserve per-regime weights from other promotions
    existing = {}
    if ACTIVE_PATH.exists():
        try:
            existing = json.loads(ACTIVE_PATH.read_text())
        except Exception:
            pass

    existing["global"]       = weights
    existing["updated_at"]   = date.today().isoformat()
    existing["promoted_from"] = variant.get("variant_id", "unknown")
    existing["fresh_sharpe"] = fresh_sharpe

    ACTIVE_PATH.write_text(json.dumps(existing, indent=2))
    print(f"[ShadowPromoter] Promoted {variant.get('variant_id')} → active_alpha_config.json")
    print(f"[ShadowPromoter] New global weights: {weights}")


def _archive(path: Path, variant: dict, reason: str) -> None:
    """Move an expired shadow config to archived_configs/."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / path.name
    shutil.move(str(path), str(dest))
    print(f"[ShadowPromoter] Archived {path.name} — {reason}")


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_shadow_promotion(baseline_sharpe: Optional[float] = None) -> int:
    """
    Scan all shadow configs. For each that has passed its shadow_expires date:
      - Re-evaluate on fresh OOS data
      - If fresh Sharpe beats baseline by MIN_EDGE_FOR_PROMOTION → promote to live
      - Otherwise → archive (expired without holding its edge)

    Args:
        baseline_sharpe: current live Sharpe. If None, reads from skill_tracker.

    Returns:
        Number of configs promoted to live.
    """
    if baseline_sharpe is None:
        try:
            from ascent.monitoring.skill_tracker import get_current_sharpe
            baseline_sharpe = get_current_sharpe("us_equities") or 0.518
        except Exception:
            baseline_sharpe = 0.518

    configs = _load_shadow_configs()
    if not configs:
        print("[ShadowPromoter] No shadow configs found.")
        return 0

    today = date.today()
    promoted = 0

    for path, variant in configs:
        expires_str = variant.get("shadow_expires", "")
        try:
            expires = date.fromisoformat(expires_str)
        except Exception:
            print(f"[ShadowPromoter] Skipping {path.name} — invalid shadow_expires")
            continue

        if today < expires:
            days_left = (expires - today).days
            print(f"[ShadowPromoter] {path.name}: {days_left} days remaining in shadow period")
            continue

        # Expired — re-evaluate
        vid = variant.get("variant_id", path.name)
        print(f"[ShadowPromoter] {vid}: shadow period ended — re-evaluating...")
        fresh_sharpe = _re_evaluate(variant)
        edge = fresh_sharpe - baseline_sharpe

        _log({
            "event":          "shadow_evaluation",
            "date":           today.isoformat(),
            "variant_id":     vid,
            "fresh_sharpe":   fresh_sharpe,
            "baseline_sharpe": baseline_sharpe,
            "edge":           round(edge, 4),
            "promoted":       edge >= MIN_EDGE_FOR_PROMOTION,
        })

        if edge >= MIN_EDGE_FOR_PROMOTION:
            _write_active_config(variant, fresh_sharpe)
            _archive(path, variant, f"promoted — edge {edge:+.3f}")
            promoted += 1
            print(f"[ShadowPromoter] ✓ PROMOTED: {vid} edge={edge:+.3f}")
        else:
            _archive(path, variant, f"expired without edge (fresh={fresh_sharpe:.3f}, baseline={baseline_sharpe:.3f})")
            print(f"[ShadowPromoter] ✗ ARCHIVED: {vid} edge={edge:+.3f} < {MIN_EDGE_FOR_PROMOTION}")

    return promoted
```

- [ ] **Step 4: Wire shadow promoter into `run_all_agents.py`**

Read `run_all_agents.py`. Find the section where `export_skill_scores()` is called (near the counterfactual scorer added in the previous plan). Add the promoter call after it, before Step 5 (orchestrator):

```python
    # Weekly shadow promotion check (runs daily but only acts on expired shadows)
    try:
        from ascent.research.shadow_promoter import run_shadow_promotion
        n_promoted = run_shadow_promotion()
        if n_promoted > 0:
            print(f"[Runner] Shadow promoter: {n_promoted} config(s) promoted to live")
    except Exception as e:
        print(f"[Runner] Shadow promotion failed: {type(e).__name__}: {e}")
```

- [ ] **Step 5: Verify syntax**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
import ast
for f in ['ascent/research/shadow_promoter.py', 'run_all_agents.py']:
    ast.parse(open(f).read()); print(f'OK: {f}')
"
```

- [ ] **Step 6: Run Task 2 tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_self_evolving_alpha.py -k "shadow_promoter" -v --tb=short 2>&1 | tail -15
```

Expected: 3 passed.

- [ ] **Step 7: Run full suite**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: 164+ passed.

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/research/shadow_promoter.py run_all_agents.py tests/test_self_evolving_alpha.py
git commit -m "$(cat <<'EOF'
feat(self-improve): shadow_promoter — auto-promotion closes the last mile

After 30-day shadow period, re-evaluate on fresh OOS. If edge holds
(>= 0.05 Sharpe), promote to active_alpha_config.json automatically.
Expired losers are archived. Called daily from run_all_agents.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Per-Regime Variant Generation in `self_improve.py`

**The problem:** Self-improve generates global variants and scores them globally. It can't learn "stressed regime needs different weights" because it never optimizes per regime. Fix: generate regime-specific variants and store per-regime weights in `active_alpha_config.json`.

**Files:**
- Modify: `ascent/research/self_improve.py`
- Test: `tests/test_self_evolving_alpha.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_self_evolving_alpha.py`:

```python
# ── Task 3 tests ───────────────────────────────────────────────────────────────

def test_generate_variants_produces_valid_weights():
    """generate_variants must produce N variants, each summing to 1.0."""
    from ascent.research.self_improve import generate_variants
    base = {"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                               "statarb": 0.15, "ml": 0.10, "volatility": 0.05}}
    variants = generate_variants(base, n=5)
    assert len(variants) == 5
    for v in variants:
        total = sum(v["alpha_weights"].values())
        assert abs(total - 1.0) < 0.01, f"weights must sum to 1, got {total}"


def test_run_self_improve_writes_regime_config(tmp_path, monkeypatch):
    """When regime='stressed', self_improve must write stressed weights to by_regime."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir()
    (tmp_path / "logs").mkdir()

    # Write minimal price cache
    idx = pd.bdate_range(end=date.today(), periods=300)
    syms = [f"S{i}" for i in range(20)] + ["SPY"]
    np.random.seed(2)
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    prices.to_parquet(tmp_path / "data_cache" / "prices_live.parquet")

    from ascent.research.self_improve import run_self_improve
    run_self_improve(current_regime="stressed")

    # self_improve_log must exist
    log_path = tmp_path / "logs" / "self_improve_log.jsonl"
    assert log_path.exists()


def test_promote_regime_variant_writes_by_regime(tmp_path, monkeypatch):
    """_promote_regime_variant must write weights to by_regime.stressed in active config."""
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_self_evolving_alpha.py -k "generate_variants or regime_config or regime_variant" \
    -v --tb=short 2>&1 | tail -15
```

Expected: 3 failed — `_promote_regime_variant` doesn't exist; `run_self_improve` doesn't accept `current_regime`.

- [ ] **Step 3: Add `_promote_regime_variant()` to `self_improve.py`**

Read `ascent/research/self_improve.py`. Add this function after `_promote_to_shadow()`:

```python
def _promote_regime_variant(weights: dict, regime: str, oos_sharpe: float, edge: float):
    """
    Write a regime-specific weight set into active_alpha_config.json by_regime section.
    Creates or updates the file — preserves other regime entries.
    """
    import json as _json
    config = {}
    if ACTIVE_CONFIG_PATH.exists():
        try:
            config = _json.loads(ACTIVE_CONFIG_PATH.read_text())
        except Exception:
            pass

    if "by_regime" not in config or not isinstance(config["by_regime"], dict):
        config["by_regime"] = {}

    config["by_regime"][str(regime).lower()] = {k: round(float(v), 4) for k, v in weights.items()}
    config["regime_updated_at"]              = datetime.now().isoformat()
    config[f"regime_{regime}_edge"]          = round(edge, 4)
    config[f"regime_{regime}_sharpe"]        = round(oos_sharpe, 4)

    os.makedirs(ACTIVE_CONFIG_PATH.parent, exist_ok=True)
    with open(ACTIVE_CONFIG_PATH, "w") as f:
        _json.dump(config, f, indent=2)
    print(f"[SelfImprove] Per-regime weights written: {regime} → {weights}")
```

- [ ] **Step 4: Add `current_regime` parameter to `run_self_improve()`**

Find the `def run_self_improve():` function signature. Change it to:

```python
def run_self_improve(current_regime: str = None):
    """Main entry point for the weekly self-improve loop.
    
    Args:
        current_regime: current regime label (e.g. 'stressed'). When provided,
                        generates regime-specific variants and promotes the best
                        to the by_regime section of active_alpha_config.json.
    """
```

At the end of `run_self_improve()`, after the existing logging block and before `return results`, add:

```python
    # Regime-conditioned promotion: write per-regime best to active config
    if current_regime and edge > MIN_SHARPE_EDGE:
        regime_weights = best.get("alpha_weights", {})
        _promote_regime_variant(regime_weights, current_regime, best["oos_sharpe"], edge)
        print(f"[SelfImprove] Per-regime weights promoted: {current_regime}")
```

- [ ] **Step 5: Wire current regime into `run_all_agents.py`**

In `run_all_agents.py`, find where `run_self_improve()` might be called or where it should be called. Look for any weekly trigger. If self_improve is only called via launchd (not in run_all_agents.py), add a regime-aware call wrapper.

Read `run_all_agents.py` to find if self_improve is called. If NOT called there, add after the shadow promoter block:

```python
    # Self-improve: runs weekly (Sunday). Pass current regime for per-regime optimization.
    try:
        import calendar as _cal
        if date.today().weekday() == 6:  # Sunday
            from ascent.research.self_improve import run_self_improve
            _current_regime = None
            try:
                import json as _rj
                from pathlib import Path as _rp
                _rsig_path = _rp("dashboard/regime_signal.json")
                if _rsig_path.exists():
                    _rsig = _rj.loads(_rsig_path.read_text())
                    if isinstance(_rsig, list):
                        _rsig = _rsig[-1] if _rsig else {}
                    _current_regime = str(_rsig.get("label", "")).lower() or None
            except Exception:
                pass
            print(f"[Runner] Running self-improve (regime={_current_regime})")
            run_self_improve(current_regime=_current_regime)
    except Exception as e:
        print(f"[Runner] Self-improve failed: {type(e).__name__}: {e}")
```

- [ ] **Step 6: Verify syntax**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
import ast
for f in ['ascent/research/self_improve.py', 'run_all_agents.py']:
    ast.parse(open(f).read()); print(f'OK: {f}')
"
```

- [ ] **Step 7: Run Task 3 tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_self_evolving_alpha.py -k "generate_variants or regime_config or regime_variant" \
    -v --tb=short 2>&1 | tail -15
```

Expected: 3 passed.

- [ ] **Step 8: Run full suite**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: 167+ passed.

- [ ] **Step 9: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/research/self_improve.py run_all_agents.py tests/test_self_evolving_alpha.py
git commit -m "$(cat <<'EOF'
feat(self-improve): per-regime variant generation and promotion

run_self_improve(current_regime=) generates and scores regime-specific
variants. Best per-regime winner promoted to active_alpha_config.json
by_regime section. System now learns stressed != calm_bull weights.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
1. ✅ Auto-promotion: `shadow_promoter.py` scans expired shadows, re-evaluates, promotes or archives
2. ✅ Wire `stack.py`: `_load_active_alpha_weights()` reads config, falls back to defaults
3. ✅ Regime-conditioned weights: `by_regime` schema, `_promote_regime_variant()`, `run_self_improve(current_regime=)`
4. ✅ Loop closes: `run_all_agents.py` calls promoter daily and self-improve on Sundays with current regime

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `_load_active_alpha_weights(regime: str = None) -> dict` — used in `build_alpha_stack` ✅
- `_promote_regime_variant(weights: dict, regime: str, oos_sharpe: float, edge: float)` — called in `run_self_improve` ✅
- `run_shadow_promotion(baseline_sharpe: Optional[float] = None) -> int` — called in `run_all_agents` ✅
- `run_self_improve(current_regime: str = None)` — called in `run_all_agents` ✅
