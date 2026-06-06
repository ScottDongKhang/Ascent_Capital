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
import logging
import os
import copy
import random
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

# Hard gate on self-modification. Keep False until OOS Sharpe is positive
# for 30 consecutive trading days on a flat config. Set to True only
# after that condition is confirmed manually.
SELF_MODIFY_ENABLED = False

log = logging.getLogger(__name__)

LOG_PATH          = Path("logs/self_improve_log.jsonl")
SHADOW_DIR        = Path("data_cache/shadow_configs")
ACTIVE_CONFIG_PATH = Path("data_cache/active_alpha_config.json")

# Match current stack.py defaults exactly
DEFAULT_ALPHA_WEIGHTS = {
    "trend":           0.43,   # increased: absorbs fundamental(0.05) — only confirmed-positive sleeve
    "meanrev":         0.05,
    "statarb":         0.15,
    "ml":              0.10,
    "volatility":      0.05,
    "fundamental":     0.00,   # IC=-0.015, IC-t=-4.75 across 31 live days: anti-signal, disabled
    "llm_fundamental": 0.03,
    "earnings":        0.05,
    "analyst":         0.05,
    "options_flow":    0.02,
    "insider":         0.02,
    "short_interest":  0.02,
    "altdata":         0.00,   # zero until first source passes IC gate
    "narrative":       0.03,   # activate narrative alpha
}

PERTURB_RANGE  = 0.10   # max +/- 10% per sleeve per variant
MIN_SHARPE_EDGE = 0.05  # variant must beat live by this much to enter shadow
N_VARIANTS     = 5
OOS_WINDOW     = 63     # trading days (for future real eval)

# Minimum weight floor per sleeve.
# Sleeves with a floor can never be perturbed to zero — this prevents the
# self-improve loop from accidentally pruning intentional signals that were
# deliberately added (e.g. earnings) but have small initial weights that
# fall within the ±10% perturbation range.
# fundamental is excluded: IC-t=-4.75 makes it an anti-signal; the loop
# must be free to keep it at zero.
MIN_SLEEVE_WEIGHTS = {
    "trend":          0.10,   # core signal — never drop below 10%
    "earnings":       0.02,   # PEAD signal — floor at 2%
    "analyst":        0.02,   # analyst revision signal — floor at 2%
    "options_flow":   0.01,   # options sentiment — floor at 1%
    "insider":        0.01,   # insider flow — floor at 1%
    "short_interest": 0.01,   # short squeeze — floor at 1%
}


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

def generate_variants(base_config: dict, n: int = N_VARIANTS, regime: str = None) -> list:
    """
    Generate N variant configs by perturbing sleeve weights within safe bounds.

    When regime is provided, first attempts LLM-guided hypothesis generation via
    factor_proposer. Falls back to random perturbation if LLM is unavailable or
    produces fewer variants than requested.

    Weights are renormalized to sum to 1 after perturbation.
    """
    # Respect the kill switch -- no variants generated while gate is closed
    if not SELF_MODIFY_ENABLED:
        log.info("[SelfImprove] SELF_MODIFY_ENABLED=False -- generate_variants returning []")
        return []

    base_weights   = base_config.get("alpha_weights", DEFAULT_ALPHA_WEIGHTS)
    active_sleeves = dict(base_weights)

    # Try LLM-guided hypothesis generation when regime is known
    if regime:
        try:
            from ascent.research.factor_proposer import propose_hypotheses, generate_guided_variants
            hypotheses = propose_hypotheses(regime=regime, current_weights=active_sleeves, n=n)
            if hypotheses:
                guided = generate_guided_variants(active_sleeves, hypotheses, perturb_range=0.03)
                if len(guided) >= n:
                    return guided[:n]
                # Top up with random variants if guided produced fewer than n
                random_count = n - len(guided)
                return guided + _random_variants(active_sleeves, n=random_count)
        except Exception as exc:
            log.warning("[SelfImprove] LLM hypothesis generation failed (%s), using random perturbation", exc)

    return _random_variants(active_sleeves, n=n)


def _random_variants(active_sleeves: dict, n: int) -> list:
    """Generate n random weight perturbation variants (original behavior)."""
    variants = []

    for i in range(n):
        variant = copy.deepcopy(active_sleeves)
        for sleeve in variant:
            delta = random.uniform(-PERTURB_RANGE, PERTURB_RANGE)
            floor = MIN_SLEEVE_WEIGHTS.get(sleeve, 0.0)
            variant[sleeve] = max(floor, variant[sleeve] + delta)

        total = sum(variant.values())
        if total > 0:
            variant = {k: round(v / total, 4) for k, v in variant.items()}
        else:
            variant = copy.deepcopy(active_sleeves)

        variants.append({
            "variant_id":    f"v{i+1}_{datetime.now().strftime('%Y%m%d')}",
            "alpha_weights": variant,
        })

    return variants


# ── Baseline and return history ───────────────────────────────────────────────

def get_baseline_sharpe():
    """Return live forward PnL Sharpe for us_equities agent. None if unavailable."""
    try:
        from ascent.monitoring.skill_tracker import get_current_sharpe
        return get_current_sharpe("us_equities")
    except Exception:
        return None


def _load_recent_returns(agent_id="us_equities", window=63):
    """Load last `window` daily returns from agent's PnL log. Returns [] if unavailable."""
    try:
        from ascent.monitoring.forward_pnl_tracker import PNL_LOGS
        log_path = PNL_LOGS.get(agent_id)
        if not log_path or not Path(log_path).exists():
            return []
        records = []
        for line in Path(log_path).read_text().splitlines():
            try:
                e = json.loads(line)
                r = e.get("return")
                if r is not None:
                    records.append(float(r))
            except Exception:
                pass
        return records[-window:]
    except Exception:
        return []


# ── Deterministic evaluator ────────────────────────────────────────────────────

def evaluate_variant(variant_config: dict) -> float:
    """Evaluate variant using real lightweight OOS walk-forward. Deterministic."""
    try:
        from ascent.research.walk_forward_lightweight import run_lightweight_oos, TURNOVER_PENALTY
        result = run_lightweight_oos(variant_config, n_days=63)
        n_folds = result.get("n_folds", 0)
        if n_folds == 0:
            # Fall back to baseline Sharpe if OOS failed
            baseline = get_baseline_sharpe()
            if baseline is None:
                print("[SelfImprove] WARNING: no live Sharpe available — falling back to hardcoded 0.518")
                baseline = 0.518
            return round(float(baseline), 4)
        sharpe   = result["sharpe"]
        turnover = result["turnover"]
        return round(float(sharpe - TURNOVER_PENALTY * turnover), 4)
    except Exception as e:
        print(f"[SelfImprove] evaluate_variant failed: {e} — using baseline")
        baseline = get_baseline_sharpe()
        if baseline is None:
            print("[SelfImprove] WARNING: no live Sharpe available — falling back to hardcoded 0.518")
            baseline = 0.518
        return round(float(baseline), 4)


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


def _promote_regime_variant(weights: dict, regime: str, oos_sharpe: float, edge: float):
    """Write a regime-specific weight set into active_alpha_config.json by_regime section."""
    config = {}
    if ACTIVE_CONFIG_PATH.exists():
        try:
            config = json.loads(ACTIVE_CONFIG_PATH.read_text())
        except Exception:
            pass

    if "by_regime" not in config or not isinstance(config["by_regime"], dict):
        config["by_regime"] = {}

    config["by_regime"][str(regime).lower()] = {k: round(float(v), 4) for k, v in weights.items()}
    config["regime_updated_at"] = datetime.now().isoformat()
    config[f"regime_{regime}_edge"] = round(edge, 4)
    config[f"regime_{regime}_sharpe"] = round(oos_sharpe, 4)

    os.makedirs(ACTIVE_CONFIG_PATH.parent, exist_ok=True)
    with open(ACTIVE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SelfImprove] Per-regime weights written: {regime} -> {weights}")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_self_improve(current_regime: str = None):
    """Main entry point for the weekly self-improve loop."""
    if not SELF_MODIFY_ENABLED:
        log.warning("[SelfImprove] SELF_MODIFY_ENABLED=False — skipping self-modification run")
        return []

    print(f"\n{'='*60}")
    print(f"[SelfImprove] Darwinian signal optimization | {date.today()}")
    print(f"{'='*60}")

    active = _load_active_config()
    print(f"[SelfImprove] Active alpha weights: {active.get('alpha_weights', {})}")

    variants = generate_variants(active, n=N_VARIANTS, regime=current_regime)
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
    current_sharpe = get_baseline_sharpe() or evaluate_variant(active)
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

    # Per-regime promotion: if a regime is specified and best variant exceeds MIN_SHARPE_EDGE
    if current_regime and results:
        best_regime = max(results, key=lambda r: r.get("oos_sharpe", 0))
        live_sharpe = best_regime.get("oos_sharpe", 0)
        regime_edge = live_sharpe - current_sharpe  # reuse already-computed baseline
        if regime_edge > MIN_SHARPE_EDGE:
            regime_weights = best_regime.get("alpha_weights", {})
            _promote_regime_variant(regime_weights, current_regime, live_sharpe, regime_edge)
            print(f"[SelfImprove] Per-regime weights promoted: {current_regime}")

    return results


if __name__ == "__main__":
    run_self_improve()
