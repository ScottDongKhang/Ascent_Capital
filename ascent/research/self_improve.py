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
import math
import os
import copy
import random
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

from ascent.research.hypothesis_registry import was_previously_rejected, record_verdict
from ascent.research.evaluation import calmar_ratio

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
    "meanrev": 0.50,
    "statarb": 0.50,
}

PERTURB_RANGE  = 0.10   # max +/- 10% per sleeve per variant
# Promotion bar for the Calmar-based edge (2026-08-23 rework). Calmar and
# Sharpe live on different typical scales, so the pre-rework MIN_SHARPE_EDGE
# (0.05) cannot be reused as-is for a Calmar-based edge -- see CLAUDE.md /
# CURRENT_VERIFIED_NUMBERS.md, which cites the canonical walk-forward
# artifact's calmar_ratio=0.223 against a Sharpe of ~0.415-0.42 for the same
# book (docs/session_log_archive.md: "Sharpe 0.415 ... now"). That gives a
# Calmar/Sharpe ratio of roughly 0.223/0.415 ~= 0.54 for this strategy's
# return profile. Scaling the old bar by that ratio: 0.05 * 0.54 ~= 0.027.
# Rounded down slightly to stay conservative (promotion should be at least as
# hard to clear as before, not easier), MIN_CALMAR_EDGE = 0.03.
MIN_CALMAR_EDGE = 0.03  # variant must beat live Calmar by this much to enter shadow
N_VARIANTS     = 5
OOS_WINDOW     = 63     # trading days (for future real eval)

# Minimum weight floor per sleeve.
# Sleeves with a floor can never be perturbed to zero — this prevents the
# self-improve loop from accidentally pruning intentional signals that were
# deliberately added (e.g. earnings) but have small initial weights that
# fall within the ±10% perturbation range.
# fundamental is excluded: IC-t=-4.75 makes it an anti-signal; the loop
# must be free to keep it at zero.
# meanrev/statarb (the only surviving sleeves) had no floors defined pre-reduction,
# so there is nothing to carry forward here.
MIN_SLEEVE_WEIGHTS = {}


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
    """Return live forward PnL Sharpe for us_equities agent. None if unavailable.

    Deliberately left Sharpe-only: it reads ascent.monitoring.skill_tracker,
    which only exposes a Sharpe scalar (no underlying return series), and that
    module is out of scope for this change. Kept as a secondary/reported metric
    -- ranking and promotion decisions use get_baseline_calmar() below, not this.
    """
    try:
        from ascent.monitoring.skill_tracker import get_current_sharpe
        return get_current_sharpe("us_equities")
    except Exception:
        return None


def get_baseline_calmar():
    """Live Calmar baseline for the ranking/promotion decision.

    Computed from the same recent daily PnL returns get_baseline_sharpe's
    source cannot expose (skill_tracker only returns a Sharpe scalar), using
    the already-existing _load_recent_returns() helper below and the same
    OOS_WINDOW used to size variant evaluation. Returns None if too few
    observations are available, so callers can fall back to an OOS-evaluated
    baseline (see run_self_improve).
    """
    returns = _load_recent_returns(agent_id="us_equities", window=OOS_WINDOW)
    if len(returns) < 10:
        return None
    try:
        return round(float(calmar_ratio(pd.Series(returns))), 4)
    except Exception:
        return None


def _artifact_baseline_sharpe() -> float:
    """Fallback baseline Sharpe, read from the verified walk-forward artifact.

    Deliberately left Sharpe-only, same reasoning as get_baseline_sharpe():
    canonical_wf() (ascent.reporting.verified_numbers) only exposes wf.sharpe,
    not a Calmar equivalent, and that module is out of scope here. This is the
    last-resort fallback used only inside evaluate_variant's own failure path
    (OOS itself produced zero folds or raised) -- not the primary baseline
    used for the promotion edge, which is get_baseline_calmar().

    This used to be a hardcoded 0.518 that matched no artifact in the repo. A
    fabricated baseline is the worst possible value here: it silently sets the
    bar that every variant is promoted against. If the artifact cannot be read
    we raise, aborting the promotion run, because declining to promote is
    always safe and promoting against an invented number is not.
    """
    from ascent.reporting.verified_numbers import canonical_wf
    wf = canonical_wf()  # raises MissingArtifact if unreadable
    print(f"[SelfImprove] no live Sharpe available — baseline from "
          f"{wf.artifact}: {wf.sharpe:.4f}")
    return wf.sharpe


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
#
# Ranking objective (2026-08-23): Calmar (CAGR / |max drawdown|), not Sharpe.
# The owner's goal is steady, low-volatility monthly returns with capital
# preservation first -- "many small consistent gains over a big win followed
# by a big loss." Sharpe penalizes upside and downside volatility symmetrically,
# which doesn't match that preference. Calmar is return per unit of the actual
# drawdown suffered, which directly encodes "avoid the big loss." Sharpe is
# still computed and reported alongside for visibility -- it is no longer the
# ranking key.

def score_variant(sharpe: float, turnover: float, returns) -> dict:
    """Single source of truth for the Calmar-based variant scoring formula.

    Shared by _evaluate_variant_full (self_improve.py's shadow *entry* gate)
    and shadow_promoter._re_evaluate (the 30-day shadow *graduation*
    re-score) -- 2026-08-23 review found those two independently
    reimplementing the identical `calmar_ratio(returns) - TURNOVER_PENALTY *
    turnover` math, which is exactly what let their missing-data fallback
    behaviors and promotion thresholds drift apart. There must be exactly
    one implementation of this formula.

    Fail-closed contract: `returns` is the per-day OOS return series. When
    it is empty/falsy there is no drawdown signal to compute a real Calmar
    from, and this returns score=None, calmar=None rather than substituting
    `sharpe` for `calmar` -- Sharpe and Calmar live on different scales, so
    letting one silently stand in for the other is a unit-mismatch bug, not
    a harmless fallback. `sharpe` is always populated (it comes straight
    from the OOS result) and is a legitimate, reportable figure even when
    score/calmar are None. Callers must treat score=None as a hard loss
    (float('-inf')) in ranking/edge comparisons, never as 0 or as `sharpe`.

    Returns {"score": <calmar - TURNOVER_PENALTY * turnover, the ranking
                        key, or None>,
             "calmar": <raw Calmar, before the turnover penalty, or None>,
             "sharpe": <raw Sharpe, secondary/reported only, always a float>}.
    """
    from ascent.research.walk_forward_lightweight import TURNOVER_PENALTY

    sharpe = round(float(sharpe), 4)
    turnover = float(turnover)
    if not returns:
        return {"score": None, "calmar": None, "sharpe": sharpe}

    calmar = float(calmar_ratio(pd.Series(returns)))
    score = round(float(calmar - TURNOVER_PENALTY * turnover), 4)
    return {"score": score, "calmar": round(calmar, 4), "sharpe": sharpe}


def _evaluate_variant_full(variant_config: dict) -> dict:
    """Evaluate variant using real lightweight OOS walk-forward. Deterministic.

    Returns {"score": <Calmar-based fitness, the ranking key, or None if
                        unavailable -- see below>,
             "calmar": <raw Calmar, before the turnover penalty, or None>,
             "sharpe": <raw Sharpe, secondary/reported only>}.

    Bug fix (2026-08-23 review): the OOS-failure and exception fallback paths
    used to reuse ONE baseline Sharpe value for all three of score/calmar/
    sharpe. Since score/calmar feed direct numeric comparisons against real
    Calmar scores elsewhere (max() ranking, edge = best_calmar - baseline
    calmar, MIN_CALMAR_EDGE gating), silently substituting a Sharpe-scale
    number there is the same unit-mismatch bug as the `prior.get('oos_calmar',
    prior.get('oos_sharpe', ...))` fallback fixed above. When no real Calmar
    can be computed, score/calmar are now returned as None instead -- callers
    must not treat None as a real Calmar; they should score it as a loss
    (float('-inf')) rather than let a Sharpe-scale number pass for Calmar.
    `sharpe` still returns the baseline as a legitimate, reportable Sharpe
    figure -- only the Calmar-scale fields are affected.

    The actual scoring math (and its missing-returns fail-closed behavior)
    now lives in score_variant() above -- see that docstring for the
    consolidation rationale.
    """
    try:
        from ascent.research.walk_forward_lightweight import run_lightweight_oos
        result = run_lightweight_oos(variant_config, n_days=63)
        n_folds = result.get("n_folds", 0)
        if n_folds == 0:
            # OOS failed outright -- no return series exists in this branch,
            # so there is nothing to compute a real Calmar from. Report the
            # baseline Sharpe for visibility only; score/calmar stay None.
            baseline = get_baseline_sharpe()
            if baseline is None:
                baseline = _artifact_baseline_sharpe()
            baseline = round(float(baseline), 4)
            return {"score": None, "calmar": None, "sharpe": baseline}

        metrics = score_variant(result["sharpe"], result["turnover"], result.get("returns"))
        if metrics["score"] is None:
            # No per-day OOS return series available (e.g. an older/mocked
            # run_lightweight_oos in a test that doesn't return "returns").
            print("[SelfImprove] evaluate_variant: no 'returns' in OOS result "
                  "-- no Calmar computable for this evaluation")
        return metrics
    except Exception as e:
        print(f"[SelfImprove] evaluate_variant failed: {e} — using baseline Sharpe only")
        baseline = get_baseline_sharpe()
        if baseline is None:
            baseline = _artifact_baseline_sharpe()
        baseline = round(float(baseline), 4)
        return {"score": None, "calmar": None, "sharpe": baseline}


def evaluate_variant(variant_config: dict) -> float:
    """Backward-compatible scalar API.

    Returns the Calmar-based fitness score (calmar - TURNOVER_PENALTY *
    turnover) used for ranking -- kept as a plain float because
    tests/test_plan_c.py::test_evaluate_variant_is_deterministic and
    tests/test_self_improve_phase_d.py::test_evaluate_variant_uses_real_oos
    treat this as a scalar. Use _evaluate_variant_full() when the raw
    calmar/sharpe breakdown is also needed (run_self_improve does).

    Bug fix (2026-08-23 review, round 2): this used to fall back to
    `metrics["sharpe"]` when `metrics["score"]` was None -- silently
    returning a Sharpe-scale number from a function whose module-level
    docstring and every caller treat the return value as Calmar-scale. That
    reintroduces the exact Sharpe-as-Calmar unit-mismatch bug the rest of
    this rework eliminated, and this function is public (no leading
    underscore) with no guarantee a future caller is one of the two known
    tests. `run_self_improve` itself never called this wrapper (it reads
    the dict from _evaluate_variant_full directly and already treats a
    None score as a hard loss, float('-inf')) -- so match that same
    fail-closed convention here instead of inventing a second one. A caller
    that actually needs to distinguish "genuinely no Calmar available" from
    "a real, very bad Calmar" must use _evaluate_variant_full() directly.
    """
    metrics = _evaluate_variant_full(variant_config)
    return metrics["score"] if metrics["score"] is not None else float("-inf")


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


def _promote_regime_variant(weights: dict, regime: str, oos_sharpe: float, edge: float,
                             oos_sharpe_raw: float = None):
    """Write a regime-specific weight set into active_alpha_config.json by_regime section.

    NOTE on the `oos_sharpe` param name: since the 2026-08-23 Calmar rework,
    callers pass the Calmar-based ranking score here, not raw Sharpe. The
    parameter name is kept unchanged for backward compatibility with the
    existing call signature and
    tests/test_self_evolving_alpha.py::test_promote_regime_variant_writes_by_regime,
    which calls this with the keyword `oos_sharpe=`.

    Bug fix (2026-08-23 review): this used to write the Calmar-based ranking
    score into the `regime_{regime}_sharpe` field -- a name that promises a
    Sharpe ratio -- while the true Sharpe only landed under a separate,
    optional `..._sharpe_raw` key. Any consumer reading `regime_*_sharpe` at
    face value silently ingested a Calmar-scale number as Sharpe. Fixed so
    field names now match their contents: `regime_{regime}_calmar` holds the
    Calmar-based ranking score (the `oos_sharpe` argument, despite its name),
    and `regime_{regime}_sharpe` holds the true Sharpe (the `oos_sharpe_raw`
    argument). Both keys are always written -- `regime_{regime}_sharpe` is
    written as `null` when the caller has no true-Sharpe figure to pass,
    rather than being silently absent or backfilled with the Calmar value.
    grep of the repo (excluding wf_framework/.venv) found no other reader of
    `active_alpha_config.json`'s `regime_*_sharpe` field, so this rename is
    safe.
    """
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
    config[f"regime_{regime}_calmar"] = round(float(oos_sharpe), 4)
    config[f"regime_{regime}_sharpe"] = (
        round(float(oos_sharpe_raw), 4) if oos_sharpe_raw is not None else None
    )

    os.makedirs(ACTIVE_CONFIG_PATH.parent, exist_ok=True)
    with open(ACTIVE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SelfImprove] Per-regime weights written: {regime} -> {weights}")


def _json_safe(x):
    """Replace the fail-closed float('-inf') sentinel with None before it
    reaches json.dumps().

    current_calmar, edge, and per-variant oos_calmar/oos_sharpe can all be
    float('-inf') in-memory (the deliberate fail-closed value used when no
    real Calmar baseline/score is computable -- see score_variant() and
    run_self_improve() above). That's fine for the in-memory max()-ranking
    comparisons, but json.dumps(..., allow_nan=True) (the default) would
    serialize -inf as the bare token `-Infinity`, which is not valid JSON --
    any non-Python reader (live_dashboard.py's st.json(), a JS JSON.parse,
    `jq` without -c/--stream tricks) fails to parse that log line.

    None is the right JSON-safe stand-in rather than a large-but-finite
    negative number: the existing read-back paths (this file's
    `prior.get("oos_calmar")` / `prior.get("oos_sharpe")`, both just above,
    and hypothesis_registry.was_previously_rejected's downstream readers)
    already treat a missing/None value as "no real score on record" and
    re-derive float('-inf') for ranking on read. Writing None keeps that
    round-trip intact; only the on-disk representation changes here, never
    the in-memory fail-closed value used for comparisons during this run.
    """
    return None if x == float("-inf") else x


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
        prior = was_previously_rejected(v.get("alpha_weights", {}))
        if prior is not None:
            print(f"  {v['variant_id']}: skipping evaluation -- already tested and "
                  f"rejected on {prior.get('date', 'unknown date')} "
                  f"(prior variant_id={prior.get('variant_id')}, "
                  f"oos_sharpe={prior.get('oos_sharpe')})")
            # oos_calmar is only present on registry entries written after the
            # 2026-08-23 Calmar rework (hypothesis_registry.record_verdict now
            # persists it). Older entries -- or any entry where it was never
            # computed -- have no real Calmar on record. Falling back to
            # oos_sharpe here was the bug: Sharpe and Calmar live on different
            # scales, so a rejected variant could silently outrank a real
            # Calmar score just from unit mismatch. Fall straight to -inf
            # instead, so a variant with no real Calmar simply loses the max()
            # comparison rather than being scored on the wrong metric.
            v["oos_calmar"] = prior.get("oos_calmar")
            if v["oos_calmar"] is None:
                v["oos_calmar"] = float("-inf")
            # Same None-safe pattern as oos_calmar above (not `prior.get(...,
            # float("-inf"))`, which only applies its default when the key is
            # *absent* -- a registry entry written with oos_sharpe explicitly
            # null on disk, e.g. by the JSON-safety fix below, would slip
            # through as None and later crash a `:.3f` format elsewhere).
            v["oos_sharpe"] = prior.get("oos_sharpe")
            if v["oos_sharpe"] is None:
                v["oos_sharpe"] = float("-inf")
            results.append(v)
            continue
        metrics = _evaluate_variant_full(v)
        # metrics["score"] is None when no real Calmar was computable (see
        # _evaluate_variant_full docstring). Score it as a hard loss rather
        # than let a Sharpe-scale number stand in for Calmar in the max()
        # ranking below -- the same unit-mismatch fix as the rejected-variant
        # branch above.
        v["oos_calmar"] = metrics["score"] if metrics["score"] is not None else float("-inf")
        v["oos_sharpe"] = metrics["sharpe"]  # secondary, reported only
        results.append(v)
        calmar_str = f"{metrics['score']:.3f}" if metrics["score"] is not None else "N/A"
        # Print the sleeve keys actually present in this variant's alpha_weights
        # (meanrev/statarb per the current 2-sleeve set -- CLAUDE.md integrity
        # constraint #6) rather than stale hardcoded 'trend'/'ml' keys that no
        # longer exist and always printed 0.00.
        weights_str = " ".join(f"{k}={wt:.2f}" for k, wt in v["alpha_weights"].items())
        print(f"  {v['variant_id']}: {weights_str} "
              f"| Calmar={calmar_str} (Sharpe={metrics['sharpe']:.3f})")

    best = max(results, key=lambda x: x["oos_calmar"])

    # Primary baseline: live Calmar. Falls back to an OOS-evaluated Calmar on
    # the active config if too little live PnL history exists yet.
    current_calmar = get_baseline_calmar()
    # Secondary/reported only -- see get_baseline_sharpe()/_artifact_baseline_sharpe()
    # docstrings for why this stays Sharpe-only rather than feeding the promotion decision.
    current_sharpe = get_baseline_sharpe()

    # Bug fix (2026-08-23 review): computing the fallback separately for
    # Calmar and Sharpe could call _evaluate_variant_full(active) twice for
    # the identical `active` config in the same run (once per missing
    # baseline) -- doubling the cost of a real 63-day lightweight OOS
    # walk-forward for no new information. Compute once and reuse it for
    # both fallbacks.
    if current_calmar is None or current_sharpe is None:
        active_metrics = _evaluate_variant_full(active)
        if current_calmar is None:
            current_calmar = active_metrics["score"]
        if current_sharpe is None:
            current_sharpe = active_metrics["sharpe"]

    if current_calmar is None:
        # No live Calmar and no OOS-evaluable Calmar for the active config
        # either (_evaluate_variant_full's own OOS path also failed). There
        # is no safe Calmar-scale baseline to compare against -- falling
        # back to a Sharpe-scale number here would reintroduce the same
        # unit-mismatch bug this review fixed elsewhere. Fail closed: no
        # promotion this run.
        print("[SelfImprove] No Calmar baseline available (live or OOS) -- "
              "skipping promotion decision this run.")
        current_calmar = float("-inf")
        edge = float("-inf")
    else:
        edge = best["oos_calmar"] - current_calmar

    print(f"\n[SelfImprove] Current config Calmar (estimated): {current_calmar:.3f} "
          f"(Sharpe={current_sharpe:.3f})")
    print(f"[SelfImprove] Best variant Calmar:               {best['oos_calmar']:.3f} "
          f"(Sharpe={best['oos_sharpe']:.3f})")
    print(f"[SelfImprove] Edge (Calmar):                     {edge:+.3f}")

    if edge > MIN_CALMAR_EDGE:
        print(f"\n[SelfImprove] PROMOTING {best['variant_id']} to shadow (edge +{edge:.3f})")
        _promote_to_shadow(best, edge)
    else:
        print(f"\n[SelfImprove] No variant beat current by >{MIN_CALMAR_EDGE:.2f}. Keeping current config.")

    # Log
    os.makedirs(LOG_PATH.parent, exist_ok=True)
    # _json_safe(): current_calmar/edge/per-variant oos_calmar/oos_sharpe can
    # be float('-inf') (fail-closed sentinel); JSON has no -inf token, so
    # swap it for None only in what gets persisted -- the in-memory `results`
    # list (still used below, by the record_verdict loop and the per-regime
    # promotion block) keeps its real -inf values for comparisons.
    logged_variants = [
        {**v, "oos_calmar": _json_safe(v.get("oos_calmar")), "oos_sharpe": _json_safe(v.get("oos_sharpe"))}
        for v in results
    ]
    log_entry = {
        "date":               date.today().isoformat(),
        "timestamp":          datetime.now().isoformat(),
        "current_calmar_est": _json_safe(current_calmar),
        "current_sharpe_est": _json_safe(current_sharpe),
        "best_variant":       best["variant_id"],
        "best_calmar":        _json_safe(best["oos_calmar"]),
        "best_sharpe":        _json_safe(best["oos_sharpe"]),
        "edge":               _json_safe(edge),
        "promoted":           edge > MIN_CALMAR_EDGE,
        "n_variants_tested":  len(results),
        "variants":           logged_variants,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"[SelfImprove] Logged to {LOG_PATH}")

    # Rejected-hypothesis registry: one verdict per variant, reusing the
    # already-computed edge/promoted values -- no new evaluation logic.
    # hypothesis_registry.record_verdict's `oos_sharpe` param is passed the
    # raw Sharpe (v["oos_sharpe"]), matching its literal name/semantics --
    # the ranking decision itself (edge/promoted) is Calmar-based.
    for v in results:
        # record_verdict persists these via json.dumps too (hypothesis_registry.py)
        # -- same -inf-is-not-valid-JSON issue as log_entry above, so sanitize
        # here as well. The in-memory `edge` used for the `promoted` comparison
        # just below is untouched (still real -inf when applicable).
        record_verdict(
            variant_config=v.get("alpha_weights", {}),
            variant_id=v["variant_id"],
            oos_sharpe=_json_safe(v["oos_sharpe"]),
            oos_calmar=_json_safe(v["oos_calmar"]),
            edge=_json_safe(edge),
            promoted=edge > MIN_CALMAR_EDGE,
        )

    # Per-regime promotion: if a regime is specified and best variant exceeds MIN_CALMAR_EDGE.
    # Guard: current_calmar can be the float("-inf") fail-closed sentinel set above when no
    # valid baseline (live or OOS) was available. Without this guard,
    # `live_calmar - (-inf) == +inf` for ANY live_calmar, which always exceeds MIN_CALMAR_EDGE
    # and would promote a per-regime variant with zero valid baseline comparison -- exactly
    # the outcome the fail-closed sentinel above is meant to prevent. Skip the whole block
    # when there's no finite baseline, mirroring the main promotion path's fail-closed edge.
    if current_regime and results and math.isfinite(current_calmar):
        best_regime = max(results, key=lambda r: r.get("oos_calmar", float("-inf")))
        live_calmar = best_regime.get("oos_calmar", 0)
        regime_edge = live_calmar - current_calmar  # reuse already-computed baseline
        if regime_edge > MIN_CALMAR_EDGE:
            regime_weights = best_regime.get("alpha_weights", {})
            _promote_regime_variant(regime_weights, current_regime, live_calmar, regime_edge,
                                     oos_sharpe_raw=best_regime.get("oos_sharpe"))
            print(f"[SelfImprove] Per-regime weights promoted: {current_regime}")

    return results


if __name__ == "__main__":
    run_self_improve()
