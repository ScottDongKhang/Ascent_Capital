"""
ascent/research/self_improve.py
Darwinian self-improving signal loop.

Runs weekly (via launchd or manual trigger).
Generates N=5 variant configs by perturbing alpha sleeve weights,
evaluates each on a heuristic OOS score, promotes winners to shadow.

V1 uses a lightweight heuristic evaluator. The real walk-forward
evaluation is Phase D (Mac Mini with compute time to spare).

Usage:
    python3 -m ascent.research.self_improve
"""

import json
import os
import copy
import random
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

LOG_PATH          = Path("logs/self_improve_log.jsonl")
SHADOW_DIR        = Path("data_cache/shadow_configs")
ACTIVE_CONFIG_PATH = Path("data_cache/active_alpha_config.json")

# Match current stack.py defaults exactly
DEFAULT_ALPHA_WEIGHTS = {
    "trend":      0.70,
    "meanrev":    0.05,
    "statarb":    0.15,
    "ml":         0.10,
    "volatility": 0.00,
}

PERTURB_RANGE  = 0.10   # max +/- 10% per sleeve per variant
MIN_SHARPE_EDGE = 0.10  # variant must beat live by this much to enter shadow
N_VARIANTS     = 5
OOS_WINDOW     = 63     # trading days (for future real eval)
# Baseline Sharpe from Phase 5.1 walk-forward (that mode was removed).
# Phase D TODO: replace with live forward PnL Sharpe from skill_tracker:
#   from ascent.monitoring.skill_tracker import get_current_sharpe
#   CURRENT_OOS_SHARPE = get_current_sharpe("us_equities") or 0.518
CURRENT_OOS_SHARPE = 0.518


# ── Config I/O ─────────────────────────────────────────────────────────────────

def _load_active_config() -> dict:
    if ACTIVE_CONFIG_PATH.exists():
        try:
            with open(ACTIVE_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"alpha_weights": DEFAULT_ALPHA_WEIGHTS.copy()}


def _save_active_config(config: dict):
    os.makedirs(ACTIVE_CONFIG_PATH.parent, exist_ok=True)
    with open(ACTIVE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SelfImprove] Active config saved to {ACTIVE_CONFIG_PATH}")


# ── Variant generation ─────────────────────────────────────────────────────────

def generate_variants(base_config: dict, n: int = N_VARIANTS) -> list:
    """
    Generate N variant configs by perturbing sleeve weights within safe bounds.
    Weights are renormalized to sum to 1 after perturbation.
    Volatility sleeve is excluded (currently disabled in stack.py).
    """
    base_weights = base_config.get("alpha_weights", DEFAULT_ALPHA_WEIGHTS)
    active_sleeves = {k: v for k, v in base_weights.items() if k != "volatility"}
    variants = []

    for i in range(n):
        variant = copy.deepcopy(active_sleeves)
        for sleeve in variant:
            delta = random.uniform(-PERTURB_RANGE, PERTURB_RANGE)
            variant[sleeve] = max(0.0, variant[sleeve] + delta)

        total = sum(variant.values())
        if total > 0:
            variant = {k: round(v / total, 4) for k, v in variant.items()}
        else:
            variant = copy.deepcopy(active_sleeves)

        # Keep volatility at 0 (disabled)
        variant["volatility"] = 0.0

        variants.append({
            "variant_id":    f"v{i+1}_{datetime.now().strftime('%Y%m%d')}",
            "alpha_weights": variant,
        })

    return variants


# ── Heuristic evaluator (V1) ───────────────────────────────────────────────────

def evaluate_variant(variant_config: dict) -> float:
    """
    V1 lightweight heuristic evaluation.

    Penalizes extreme deviation from defaults (overfit risk).
    Adds small random noise to simulate real OOS variability.
    Returns estimated Sharpe.

    Phase D TODO: Replace with real walk-forward call:
        from ascent.research.walk_forward_runner import walk_forward_pipeline
        result = walk_forward_pipeline(alpha_weights=variant_config["alpha_weights"], ...)
        return result["sharpe"]
    """
    weights   = variant_config.get("alpha_weights", DEFAULT_ALPHA_WEIGHTS)
    deviation = sum(
        abs(weights.get(k, 0) - DEFAULT_ALPHA_WEIGHTS.get(k, 0))
        for k in DEFAULT_ALPHA_WEIGHTS
    )

    # Moderate diversity is good; extreme changes risk overfitting
    if deviation < 0.05:
        diversity_bonus = -0.02   # too similar — no real change
    elif deviation < 0.25:
        diversity_bonus = 0.02    # healthy exploration range
    else:
        diversity_bonus = -0.05   # too extreme

    noise            = random.gauss(0, 0.12)
    estimated_sharpe = CURRENT_OOS_SHARPE + noise + diversity_bonus
    return round(estimated_sharpe, 4)


# ── Shadow promotion ───────────────────────────────────────────────────────────

def _promote_to_shadow(variant: dict, edge: float):
    """Save a winning variant to shadow for 30-day monitoring."""
    os.makedirs(SHADOW_DIR, exist_ok=True)
    variant["promoted_at"]       = datetime.now().isoformat()
    variant["edge_over_current"] = round(edge, 4)
    variant["shadow_expires"]    = (date.today() + timedelta(days=30)).isoformat()

    path = SHADOW_DIR / f"{variant['variant_id']}.json"
    with open(path, "w") as f:
        json.dump(variant, f, indent=2)
    print(f"[SelfImprove] Shadow config saved to {path}")
    print(f"[SelfImprove] Monitor for 30 days before promoting to live")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_self_improve():
    """Main entry point for the weekly self-improve loop."""
    print(f"\n{'='*60}")
    print(f"[SelfImprove] Darwinian signal optimization | {date.today()}")
    print(f"{'='*60}")

    active = _load_active_config()
    print(f"[SelfImprove] Active alpha weights: {active.get('alpha_weights', {})}")

    variants = generate_variants(active, n=N_VARIANTS)
    print(f"[SelfImprove] Generated {len(variants)} variants\n")

    results = []
    for v in variants:
        sharpe      = evaluate_variant(v)
        v["oos_sharpe"] = sharpe
        results.append(v)
        print(f"  {v['variant_id']}: trend={v['alpha_weights'].get('trend', 0):.2f} "
              f"statarb={v['alpha_weights'].get('statarb', 0):.2f} "
              f"ml={v['alpha_weights'].get('ml', 0):.2f} "
              f"| Sharpe={sharpe:.3f}")

    best          = max(results, key=lambda x: x["oos_sharpe"])
    current_sharpe = evaluate_variant(active)
    edge          = best["oos_sharpe"] - current_sharpe

    print(f"\n[SelfImprove] Current config Sharpe (estimated): {current_sharpe:.3f}")
    print(f"[SelfImprove] Best variant Sharpe:               {best['oos_sharpe']:.3f}")
    print(f"[SelfImprove] Edge:                              {edge:+.3f}")

    if edge > MIN_SHARPE_EDGE:
        print(f"\n[SelfImprove] PROMOTING {best['variant_id']} to shadow (edge +{edge:.3f})")
        _promote_to_shadow(best, edge)
    else:
        print(f"\n[SelfImprove] No variant beat current by >{MIN_SHARPE_EDGE:.2f}. Keeping current config.")

    # Log
    os.makedirs(LOG_PATH.parent, exist_ok=True)
    log_entry = {
        "date":               date.today().isoformat(),
        "timestamp":          datetime.now().isoformat(),
        "current_sharpe_est": current_sharpe,
        "best_variant":       best["variant_id"],
        "best_sharpe":        best["oos_sharpe"],
        "edge":               edge,
        "promoted":           edge > MIN_SHARPE_EDGE,
        "n_variants_tested":  len(results),
        "variants":           results,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"[SelfImprove] Logged to {LOG_PATH}")
    return results


if __name__ == "__main__":
    run_self_improve()
