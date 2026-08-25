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

# Graduation bar for the Calmar-based edge. Reuses self_improve.py's
# MIN_CALMAR_EDGE (2026-08-23 review) rather than defining an independently
# tunable number here: this value used to be a leftover 0.05 Sharpe-scale
# bar that was never rescaled when the edge it's compared against became
# Calmar-based, while MIN_CALMAR_EDGE *was* deliberately rescaled (0.05 ->
# 0.03, see its docstring in self_improve.py for the ~0.54 Calmar/Sharpe
# ratio derivation). The entry gate (shadow admission) and the graduation
# gate (shadow promotion) are scoring the same Calmar-based quantity, so
# there is no reason for them to carry separate bars -- a single named
# constant is imported here instead of a second copy that can drift again.
from ascent.research.self_improve import MIN_CALMAR_EDGE, score_variant

MIN_EDGE_FOR_PROMOTION = MIN_CALMAR_EDGE

# Safety net: if a promoted variant has zeroed out an intentional sleeve,
# restore it to at least this floor before writing to active config.
# Mirrors MIN_SLEEVE_WEIGHTS in self_improve.py — both must stay in sync.
# Post proof-audit reduction to meanrev/statarb (neither has a floor), this is empty:
# trend/fundamental/earnings/analyst/options_flow/insider/short_interest are cut sleeves.
# Leaving their floors here would force-reinject ~19pp of weight into cut sleeves on
# every promotion, silently reversing the reduction the moment SELF_MODIFY_ENABLED flips.
_SLEEVE_FLOORS = {}


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


# Fail-closed sentinel (2026-08-23 review, bug fix): must be unconditionally
# worse than ANY real baseline_calmar, not merely worse than a positive one.
# 0.0 only "loses" edge = fresh - baseline when baseline > 0; if the live
# book is itself in a drawdown, get_baseline_calmar() legitimately returns a
# negative Calmar (e.g. -0.05), and 0.0 - (-0.05) = +0.05 clears
# MIN_EDGE_FOR_PROMOTION, promoting an untested/failed re-evaluation. Use
# float('-inf') instead, matching self_improve.py's identical fail-closed
# contract for a None/missing Calmar. `sharpe` stays a plain reportable
# float (never used as a ranking key -- see _re_evaluate's docstring).
_ZERO_EVAL = {"score": float("-inf"), "calmar": float("-inf"), "sharpe": 0.0}


def _re_evaluate(variant_config: dict) -> dict:
    """Re-evaluate a shadow variant on fresh OOS data at graduation time.

    Scoring must match how self_improve.py now gates *entry* into shadow
    (2026-08-23 Calmar rework) -- otherwise a variant is admitted on a
    drawdown-aware basis and then graduated/archived 30 days later by an
    unrelated raw-Sharpe re-score, which silently reverses the rework for
    half the promotion pipeline. Delegates to the single shared
    ascent.research.self_improve.score_variant() helper for the actual
    Calmar math (calmar_ratio(returns) - TURNOVER_PENALTY * turnover, with
    Sharpe carried alongside as a secondary/reported value only, never the
    ranking key) so both halves of the promotion pipeline can never drift
    apart again -- see score_variant()'s docstring.

    Fail-closed contract (2026-08-23 review, bug fix): this used to fall
    back to `calmar = sharpe` when the OOS result had no 'returns' series --
    the exact Sharpe-as-Calmar unit-mismatch bug score_variant() is designed
    to prevent. Now, when score_variant() reports no real Calmar was
    computable (score/calmar None), this fails closed to _ZERO_EVAL (a
    guaranteed loss against any real baseline) instead of substituting
    Sharpe, matching self_improve.py's own fail-closed contract.

    Returns {"score": <calmar - TURNOVER_PENALTY * turnover, ranking key>,
             "calmar": <raw Calmar>, "sharpe": <raw Sharpe, secondary>}.
    """
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
            return dict(_ZERO_EVAL)
    try:
        from ascent.research.walk_forward_lightweight import run_lightweight_oos

        result = run_lightweight_oos(variant_config, n_days=63)
        if result.get("n_folds", 0) == 0:
            return dict(_ZERO_EVAL)

        metrics = score_variant(result["sharpe"], result["turnover"], result.get("returns"))
        if metrics["score"] is None:
            # No per-day OOS return series available -- there is no drawdown
            # signal to compute a real Calmar from. Fail closed rather than
            # substitute Sharpe (see docstring above): treat as a loss.
            print("[ShadowPromoter] _re_evaluate: no 'returns' in OOS result "
                  "-- no Calmar computable; failing closed to a loss")
            return dict(_ZERO_EVAL)
        return metrics
    except Exception as e:
        print(f"[ShadowPromoter] Re-evaluation failed: {e}")
        return dict(_ZERO_EVAL)


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


def _write_active_config(variant: dict, fresh_eval: dict) -> None:
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
    existing["fresh_calmar"]  = fresh_eval["calmar"]
    # Secondary/reported only -- see _re_evaluate(); ranking uses fresh_calmar.
    existing["fresh_sharpe"]  = fresh_eval["sharpe"]

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


def run_shadow_promotion(baseline_calmar: Optional[float] = None) -> int:
    """
    Scan all shadow configs. For each past shadow_expires date:
      - Re-evaluate on fresh OOS data (Calmar-based, matching the 2026-08-23
        entry-gate rework in self_improve.py -- see _re_evaluate())
      - If fresh Calmar beats baseline by MIN_EDGE_FOR_PROMOTION -> promote to live
      - Otherwise -> archive

    Returns number of configs promoted.

    If no explicit baseline_calmar is passed and no real Calmar baseline can
    be computed (ascent.research.self_improve.get_baseline_calmar(), which
    needs >=10 recent PnL observations), this SKIPS the entire promotion
    cycle rather than falling back to a fabricated number -- a previous
    version of this function used a hardcoded 0.518 Sharpe fallback with no
    artifact backing it, which is exactly the failure mode
    self_improve.py's _artifact_baseline_sharpe() docstring describes as
    already fixed elsewhere. Declining to promote is always safe; comparing
    against an invented baseline is not.
    """
    if baseline_calmar is None:
        try:
            from ascent.research.self_improve import get_baseline_calmar
            baseline_calmar = get_baseline_calmar()
        except Exception as e:
            print(f"[ShadowPromoter] Could not import get_baseline_calmar: {e}")
            baseline_calmar = None

        if baseline_calmar is None:
            print("[ShadowPromoter] No real Calmar baseline available (insufficient "
                  "live PnL history) -- skipping this promotion cycle rather than "
                  "comparing against a fabricated number.")
            return 0

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
        fresh_eval = _re_evaluate(variant)
        fresh_calmar = fresh_eval["score"]
        edge = fresh_calmar - baseline_calmar

        _log({
            "event":           "shadow_evaluation",
            "date":            today.isoformat(),
            "variant_id":      vid,
            "fresh_calmar":    fresh_calmar,
            "fresh_sharpe":    fresh_eval["sharpe"],  # secondary/reported only
            "baseline_calmar": baseline_calmar,
            "edge":            round(edge, 4),
            "promoted":        edge >= MIN_EDGE_FOR_PROMOTION,
        })

        if edge >= MIN_EDGE_FOR_PROMOTION:
            _write_active_config(variant, fresh_eval)
            _archive(path, variant, f"promoted -- edge {edge:+.3f}")
            promoted += 1
            print(f"[ShadowPromoter] PROMOTED: {vid} edge={edge:+.3f}")
        else:
            _archive(path, variant, f"expired without edge (fresh={fresh_calmar:.3f}, baseline={baseline_calmar:.3f})")
            print(f"[ShadowPromoter] ARCHIVED: {vid} edge={edge:+.3f} < {MIN_EDGE_FOR_PROMOTION}")

    return promoted
