"""
ascent/research/factor_discovery/discovery_runner.py

Orchestrates two-path autonomous factor discovery pipeline.

Path A — PySR symbolic regression (primary)
Path B — LLM template suggestions (secondary)

Acceptance gate: IC_mean > 0.015 AND IC_IR > 0.60 AND IC_min_regime > 0.01
Nothing auto-deploys — human reviews every accepted proposal.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ascent.research.factor_discovery.regime_cpcv_evaluator import (
    evaluate_factor_regime_ic,
    passes_harvey_threshold,
)

log = logging.getLogger(__name__)

PROPOSALS_DIR     = Path("outputs/factor_proposals")
DISCOVERY_LOG     = Path("logs/factor_discovery_log.jsonl")
IC_MEAN_THRESHOLD = 0.015
IC_IR_THRESHOLD   = 0.60
IC_MIN_REGIME     = 0.010
MIN_OBSERVATIONS  = 20

_DEPLOYED_FACTORS = [
    "trend", "meanrev", "volatility", "statarb", "ml",
    "fundamental", "earnings", "analyst", "options_flow",
    "insider", "short_interest", "llm_fundamental",
]


def _load_prices() -> pd.DataFrame:
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data("prices_live"):
            df = load_parquet("prices_live")
            if "close" in df.columns:
                return df.pivot(columns="symbol", values="close").sort_index()
            elif isinstance(df.columns, pd.MultiIndex):
                return df["Close"].sort_index()
            return df.sort_index()
    except Exception as exc:
        log.warning("[FactorDiscovery] Parquet load failed: %s", exc)
    try:
        import yfinance as yf
        from ascent.config.settings import get_config
        syms = list(getattr(get_config().universe, "symbols", []))[:60]
        raw  = yf.download(syms, period="3y", auto_adjust=True, progress=False)
        return (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(
            axis=1, how="all").sort_index()
    except Exception as exc:
        log.warning("[FactorDiscovery] yfinance fallback failed: %s", exc)
        return pd.DataFrame()


def _load_regime_labels(prices_index: pd.DatetimeIndex) -> Optional[pd.Series]:
    try:
        df = pd.read_csv("dashboard/regime_labels.csv", parse_dates=["date"])
        df = df.set_index("date").sort_index()
        return df["label"].reindex(prices_index, method="ffill")
    except Exception:
        return None


def _write_log(entry: dict) -> None:
    DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISCOVERY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_proposal(candidate: dict, ic_result: dict) -> Path:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    fname = f"{candidate['name']}_{today}.json"
    path  = PROPOSALS_DIR / fname
    payload = {
        "name":            candidate["name"],
        "source":          candidate.get("source", "unknown"),
        "description":     candidate.get("description", ""),
        "expression":      candidate.get("expression", ""),
        "template":        candidate.get("template", ""),
        "template_params": candidate.get("params", {}),
        "rationale":       candidate.get("rationale", ""),
        **{k: ic_result.get(k) for k in [
            "ic_mean", "ic_ir", "ic_p5", "n_observations",
            "ic_calm_bull", "ic_stressed", "ic_crisis", "ic_min_regime"
        ]},
        "proposed_at":        datetime.now().isoformat(),
        "regime_at_proposal": candidate.get("regime", "unknown"),
        "review_status":      "pending",
        "review_notes":       "",
        "how_to_deploy": (
            "1. Review expression/description for economic soundness.\n"
            "2. PySR: translate expression into a function in ascent/features/feature_defs.py.\n"
            "3. Template: call instantiate_template(template, params).compute(df) in feature_defs.py.\n"
            "4. Register in build_all_features() with appropriate lag.\n"
            "5. Add to stack.py DEFAULT_ALPHA_WEIGHTS at initial weight 0.02.\n"
            "6. Reduce another sleeve by 0.02 to keep sum at 1.0.\n"
            "7. Run full test suite before committing.\n"
            "8. Run system once in dry-run to confirm no pipeline errors."
        ),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _build_candidates(regime: str, prices_df: pd.DataFrame) -> List[Dict]:
    candidates = []

    # Path A: PySR
    try:
        from ascent.research.factor_discovery.pysr_engine import discover_via_pysr
        pysr_candidates = discover_via_pysr(prices_df, n_periods=5, n_iterations=40)
        for c in pysr_candidates:
            c["regime"] = regime
        candidates.extend(pysr_candidates)
        log.info("[FactorDiscovery] PySR produced %d candidates", len(pysr_candidates))
    except Exception as exc:
        log.warning("[FactorDiscovery] PySR path failed: %s", exc)

    # Path B: LLM template suggestion
    try:
        from ascent.research.factor_discovery.llm_suggester import suggest_template_params
        from ascent.research.factor_discovery.feature_templates import instantiate_template
        suggestion = suggest_template_params(
            regime=regime,
            ic_context={},
            existing_factor_names=_DEPLOYED_FACTORS,
        )
        if suggestion:
            tmpl = instantiate_template(suggestion["template"], suggestion["params"])

            def _make_fn(t):
                def _fn(df):
                    return t.compute(df)
                return _fn

            candidates.append({
                "name":        f"factor_llm_{suggestion['template'].lower()[:8]}",
                "description": suggestion.get("rationale", ""),
                "source":      "llm_template",
                "template":    suggestion["template"],
                "params":      suggestion["params"],
                "rationale":   suggestion.get("rationale", ""),
                "regime":      regime,
                "fn":          _make_fn(tmpl),
            })
            log.info("[FactorDiscovery] LLM template: %s params=%s",
                     suggestion["template"], suggestion["params"])
    except Exception as exc:
        log.warning("[FactorDiscovery] LLM template path failed: %s", exc)

    return candidates


def run_factor_discovery(
    n_candidates: int = 5,
    regime: Optional[str] = None,
) -> Dict:
    """
    Run one full factor discovery cycle (both paths).
    Returns dict: n_generated, n_valid, n_accepted, n_rejected, proposals.
    """
    regime = regime or "unknown"
    log.info("[FactorDiscovery] Starting cycle — regime=%s", regime)

    prices = _load_prices()
    if prices.empty:
        log.warning("[FactorDiscovery] No price data — aborting")
        return {"n_generated": 0, "n_valid": 0, "n_accepted": 0, "n_rejected": 0, "proposals": []}

    regime_labels = _load_regime_labels(prices.index)
    candidates    = _build_candidates(regime, prices)

    n_valid = n_accepted = n_rejected = 0
    proposals = []

    for candidate in candidates:
        name = candidate.get("name", "unknown")
        fn   = candidate.get("fn")
        if fn is None:
            continue

        log.info("[FactorDiscovery] Evaluating %s (source=%s)", name, candidate.get("source"))

        ic_result = evaluate_factor_regime_ic(
            factor_fn=fn,
            prices_df=prices,
            regime_labels=regime_labels,
            n_periods=5,
        )

        if "error" in ic_result:
            log.info("[FactorDiscovery] Eval error for %s: %s", name, ic_result["error"])
            _write_log({
                "date": date.today().isoformat(), "regime": regime,
                "name": name, "status": "evaluation_error", "error": ic_result["error"],
            })
            n_rejected += 1
            continue

        n_valid  += 1
        ic_mean   = ic_result["ic_mean"]
        ic_ir     = ic_result["ic_ir"]
        n_obs     = ic_result["n_observations"]
        ic_min_r  = ic_result.get("ic_min_regime", ic_mean)

        log.info("[FactorDiscovery] %s — IC=%.4f, IR=%.3f, IC_min_regime=%.4f, n=%d",
                 name, ic_mean, ic_ir, ic_min_r, n_obs)

        log_entry = {
            "date": date.today().isoformat(), "regime": regime, "name": name,
            "source": candidate.get("source"),
            "ic_mean": ic_mean, "ic_ir": ic_ir, "n_observations": n_obs,
            "ic_min_regime": ic_min_r,
        }

        if (passes_harvey_threshold(ic_mean, ic_ir)
                and ic_min_r > IC_MIN_REGIME
                and n_obs >= MIN_OBSERVATIONS):
            path = _write_proposal(candidate, ic_result)
            proposals.append(str(path))
            n_accepted += 1
            log_entry["status"] = "accepted"
            log_entry["proposal_path"] = str(path)
            log.info("[FactorDiscovery] ACCEPTED: %s → %s", name, path)
        else:
            n_rejected += 1
            reasons = []
            if not passes_harvey_threshold(ic_mean, ic_ir):
                reasons.append(
                    f"IC={ic_mean:.4f} or IR={ic_ir:.3f} below Harvey threshold "
                    f"({IC_MEAN_THRESHOLD}/{IC_IR_THRESHOLD})"
                )
            if ic_min_r <= IC_MIN_REGIME:
                reasons.append(f"IC_min_regime={ic_min_r:.4f} <= {IC_MIN_REGIME}")
            if n_obs < MIN_OBSERVATIONS:
                reasons.append(f"Only {n_obs} observations (need {MIN_OBSERVATIONS})")
            log_entry["status"] = "rejected"
            log_entry["reasons"] = reasons
            log.info("[FactorDiscovery] Rejected %s: %s", name, "; ".join(reasons))

        _write_log(log_entry)

    summary = {
        "n_generated": len(candidates),
        "n_valid":     n_valid,
        "n_accepted":  n_accepted,
        "n_rejected":  n_rejected,
        "proposals":   proposals,
        "regime":      regime,
        "date":        date.today().isoformat(),
    }
    log.info("[FactorDiscovery] Cycle complete: %d accepted / %d rejected",
             n_accepted, n_rejected)
    return summary
