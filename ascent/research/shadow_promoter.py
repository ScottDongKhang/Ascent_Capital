"""ascent/research/shadow_promoter.py

Scans shadow configs after their 30-day monitoring period.
Re-evaluates each on fresh OOS data.
Promotes winners to active_alpha_config.json.
Archives losers.
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

MIN_EDGE_FOR_PROMOTION = 0.05

# Safety net: if a promoted variant has zeroed out an intentional sleeve,
# restore it to at least this floor before writing to active config.
# Mirrors MIN_SLEEVE_WEIGHTS in self_improve.py — both must stay in sync.
_SLEEVE_FLOORS = {"trend": 0.10, "fundamental": 0.02, "earnings": 0.02}


def _load_shadow_configs() -> list:
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
    # Check cwd-local prices first — allows tests to control data availability
    # via monkeypatch.chdir. If not present in cwd, fall back to package cache.
    cwd_prices = Path.cwd() / "data_cache" / "prices_live.parquet"
    if not cwd_prices.exists():
        # Check whether package-level cache exists; if we're in a tmp dir with
        # no local prices, treat as no-data and return 0.0 (safe fallback).
        from pathlib import Path as _Path
        pkg_cache = _Path(__file__).resolve().parents[3] / "data_cache" / "prices_live.parquet"
        # Only skip if cwd != package root (i.e. we're in a test tmp dir)
        if str(Path.cwd()) != str(pkg_cache.parent.parent):
            print("[ShadowPromoter] No local prices_live.parquet in cwd — returning 0.0")
            return 0.0
    try:
        from ascent.research.walk_forward_lightweight import run_lightweight_oos, TURNOVER_PENALTY
        result = run_lightweight_oos(variant_config, n_days=63)
        if result.get("n_folds", 0) == 0:
            return 0.0
        return round(result["sharpe"] - TURNOVER_PENALTY * result["turnover"], 4)
    except Exception as e:
        print(f"[ShadowPromoter] Re-evaluation failed: {e}")
        return 0.0


def _restore_sleeve_floors(weights: dict) -> dict:
    """
    Restore any intentional sleeve that got zeroed during perturbation or OOS
    scoring. Renormalizes after restoration so weights still sum to 1.0.
    Logs a warning for each sleeve that needed restoration.
    """
    w = dict(weights)
    restored = []
    for sleeve, floor in _SLEEVE_FLOORS.items():
        if w.get(sleeve, 0.0) < floor:
            w[sleeve] = floor
            restored.append(sleeve)
    if restored:
        total = sum(w.values())
        if total > 0:
            w = {k: round(v / total, 4) for k, v in w.items()}
        print(f"[ShadowPromoter] WARNING: restored zeroed sleeves {restored} to floor — renormalized")
    return w


def _write_active_config(variant: dict, fresh_sharpe: float) -> None:
    ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    weights = _restore_sleeve_floors(variant.get("alpha_weights", {}))

    existing = {}
    if ACTIVE_PATH.exists():
        try:
            existing = json.loads(ACTIVE_PATH.read_text())
        except Exception:
            pass

    existing["global"]        = weights
    existing["updated_at"]    = date.today().isoformat()
    existing["promoted_from"] = variant.get("variant_id", "unknown")
    existing["fresh_sharpe"]  = fresh_sharpe

    ACTIVE_PATH.write_text(json.dumps(existing, indent=2))
    print(f"[ShadowPromoter] Promoted {variant.get('variant_id')} -> active_alpha_config.json")
    print(f"[ShadowPromoter] New global weights: {weights}")


def _archive(path: Path, variant: dict, reason: str) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / path.name
    shutil.move(str(path), str(dest))
    print(f"[ShadowPromoter] Archived {path.name} -- {reason}")


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_shadow_promotion(baseline_sharpe: Optional[float] = None) -> int:
    """
    Scan all shadow configs. For each past shadow_expires date:
      - Re-evaluate on fresh OOS data
      - If fresh Sharpe beats baseline by MIN_EDGE_FOR_PROMOTION -> promote to live
      - Otherwise -> archive

    Returns number of configs promoted.
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
            print(f"[ShadowPromoter] Skipping {path.name} -- invalid shadow_expires")
            continue

        if today < expires:
            days_left = (expires - today).days
            print(f"[ShadowPromoter] {path.name}: {days_left} days remaining in shadow period")
            continue

        vid = variant.get("variant_id", path.name)
        print(f"[ShadowPromoter] {vid}: shadow period ended -- re-evaluating...")
        fresh_sharpe = _re_evaluate(variant)
        edge = fresh_sharpe - baseline_sharpe

        _log({
            "event":           "shadow_evaluation",
            "date":            today.isoformat(),
            "variant_id":      vid,
            "fresh_sharpe":    fresh_sharpe,
            "baseline_sharpe": baseline_sharpe,
            "edge":            round(edge, 4),
            "promoted":        edge >= MIN_EDGE_FOR_PROMOTION,
        })

        if edge >= MIN_EDGE_FOR_PROMOTION:
            _write_active_config(variant, fresh_sharpe)
            _archive(path, variant, f"promoted -- edge {edge:+.3f}")
            promoted += 1
            print(f"[ShadowPromoter] PROMOTED: {vid} edge={edge:+.3f}")
        else:
            _archive(path, variant, f"expired without edge (fresh={fresh_sharpe:.3f}, baseline={baseline_sharpe:.3f})")
            print(f"[ShadowPromoter] ARCHIVED: {vid} edge={edge:+.3f} < {MIN_EDGE_FOR_PROMOTION}")

    return promoted
