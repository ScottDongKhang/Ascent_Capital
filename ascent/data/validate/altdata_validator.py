"""ascent/data/validate/altdata_validator.py

IC validation gate for alternative data sources.
Same thresholds as factor discovery (Harvey FDR correction):
  IC_mean ≥ 0.015, IC_IR ≥ 0.60, IC_min_regime > 0.010, n_obs ≥ 20.

Accepted sources written to outputs/altdata_proposals/ for human review.
Nothing auto-deploys.
"""
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

IC_MEAN_THRESHOLD   = 0.015
IC_IR_THRESHOLD     = 0.60
IC_MIN_REGIME       = 0.010
MIN_OBSERVATIONS    = 20

_PROPOSALS_DIR = Path("outputs/altdata_proposals")
_CONFIG_PATH   = Path("data_cache/active_alpha_config.json")


def _compute_ic(signal_panel: pd.DataFrame, prices_df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Compute cross-sectional Spearman IC between signal and forward returns.
    Returns a Series of per-date IC values.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        log.warning("[AltdataValidator] scipy not available")
        return pd.Series(dtype=float)

    common_syms = signal_panel.columns.intersection(prices_df.columns)
    if common_syms.empty:
        return pd.Series(dtype=float)

    sig = signal_panel[common_syms]
    px  = prices_df[common_syms]

    # Forward returns: horizon-day log return
    fwd_ret = np.log(px / px.shift(horizon)).shift(-horizon)

    ic_values = []
    ic_dates  = []
    for dt in sig.index:
        if dt not in fwd_ret.index:
            continue
        sig_row = sig.loc[dt].dropna()
        ret_row = fwd_ret.loc[dt].reindex(sig_row.index).dropna()
        common  = sig_row.index.intersection(ret_row.index)
        if len(common) < 5:
            continue
        ic, _ = spearmanr(sig_row[common], ret_row[common])
        if not np.isnan(ic):
            ic_values.append(ic)
            ic_dates.append(dt)

    return pd.Series(ic_values, index=ic_dates)


def validate_altdata_source(
    signal_panel: pd.DataFrame,
    prices_df: pd.DataFrame,
    regime_labels: pd.Series,
    source_name: str,
    horizon: int = 5,
) -> dict:
    """
    Run IC gate on a signal panel (dates × symbols).
    Returns {"source", "ic_mean", "ic_ir", "ic_min_regime", "n_observations", "status", "reasons"}.
    """
    reasons = []
    result = {
        "source": source_name,
        "ic_mean": None,
        "ic_ir": None,
        "ic_min_regime": None,
        "n_observations": 0,
        "status": "rejected",
        "reasons": reasons,
    }

    if signal_panel.empty:
        reasons.append("signal_panel is empty")
        return result

    ic_series = _compute_ic(signal_panel, prices_df, horizon=horizon)
    n_obs = len(ic_series)
    result["n_observations"] = n_obs

    if n_obs < MIN_OBSERVATIONS:
        reasons.append(f"n_obs={n_obs} < {MIN_OBSERVATIONS}")
        if n_obs > 0:
            result["ic_mean"] = float(ic_series.mean())
            result["ic_ir"]   = float(ic_series.mean() / (ic_series.std() + 1e-9))
        return result

    ic_mean = float(ic_series.mean())
    ic_std  = float(ic_series.std())
    ic_ir   = float(ic_mean / (ic_std + 1e-9))

    result["ic_mean"] = round(ic_mean, 6)
    result["ic_ir"]   = round(ic_ir,   6)

    if ic_mean < IC_MEAN_THRESHOLD:
        reasons.append(f"ic_mean={ic_mean:.4f} < {IC_MEAN_THRESHOLD}")

    if ic_ir < IC_IR_THRESHOLD:
        reasons.append(f"ic_ir={ic_ir:.4f} < {IC_IR_THRESHOLD}")

    # Per-regime IC check
    regime_labels_aligned = regime_labels.reindex(ic_series.index).ffill()
    regime_ic: dict[str, float] = {}
    ic_min_regime = None

    if not regime_labels_aligned.dropna().empty:
        for regime in regime_labels_aligned.dropna().unique():
            mask = regime_labels_aligned == regime
            subset = ic_series[mask]
            if len(subset) >= 3:
                regime_ic[str(regime)] = float(subset.mean())

        if regime_ic:
            ic_min_regime = min(regime_ic.values())
            result["ic_min_regime"] = round(ic_min_regime, 6)
            if ic_min_regime <= IC_MIN_REGIME:
                reasons.append(
                    f"ic_min_regime={ic_min_regime:.4f} ≤ {IC_MIN_REGIME} "
                    f"(worst regime: {min(regime_ic, key=regime_ic.get)})"
                )
    else:
        reasons.append("no regime labels available for per-regime check")

    if not reasons:
        result["status"] = "accepted"

    return result


def run_altdata_validation(
    sources_dict: dict[str, pd.DataFrame],
    prices_df: pd.DataFrame,
    regime_labels: pd.Series,
) -> list[dict]:
    """
    Validate all alternative data sources.
    Accepted results written to outputs/altdata_proposals/.
    Returns list of validation dicts.
    """
    results = []
    for source_name, signal_panel in sources_dict.items():
        log.info("[AltdataValidator] Validating %s ...", source_name)
        result = validate_altdata_source(signal_panel, prices_df, regime_labels, source_name)
        results.append(result)
        log.info(
            "[AltdataValidator] %s → %s (IC=%.4f, IR=%.4f, n=%d)",
            source_name, result["status"],
            result["ic_mean"] or 0, result["ic_ir"] or 0, result["n_observations"],
        )
        if result["status"] == "accepted":
            _write_proposal(result)

    return results


def _write_proposal(result: dict) -> None:
    _PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    fname = _PROPOSALS_DIR / f"{result['source']}_{date.today().isoformat()}.json"
    fname.write_text(json.dumps(result, indent=2))
    log.info("[AltdataValidator] Proposal written: %s", fname)


def register_altdata_source(source_name: str, initial_weight: float = 0.02) -> None:
    """
    Register a validated source in active_alpha_config.json.
    Only called after human review of the validation proposal.
    """
    config = {}
    if _CONFIG_PATH.exists():
        try:
            config = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass

    altdata_weights = config.get("altdata_weights", {})
    altdata_weights[source_name] = initial_weight
    config["altdata_weights"] = altdata_weights

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log.info("[AltdataValidator] Registered %s with weight=%.3f", source_name, initial_weight)
