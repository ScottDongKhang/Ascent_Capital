"""
Ascent Capital - Alpha Stack
Combines multiple alpha signals into a composite score.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ascent.alpha.trend import trend_alpha
from ascent.alpha.meanrev import meanrev_alpha
from ascent.alpha.statarb import statarb_alpha
from ascent.alpha.ml_sleeve import build_ml_alpha, build_ml_alpha_cpcv

log = logging.getLogger(__name__)

DEFAULT_ALPHA_WEIGHTS = {
    "trend":       0.50,
    "meanrev":     0.05,
    "volatility":  0.05,
    "statarb":     0.15,
    "ml":          0.10,
    "fundamental": 0.05,
    "earnings":    0.05,
    "analyst":     0.05,
}

def _load_active_alpha_weights(regime: str = None) -> dict:
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


def _load_sector_map():
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if not has_data("profiles"):
            return {}
        profiles = load_parquet("profiles")
        if "symbol" not in profiles.columns or "sector" not in profiles.columns:
            return {}
        return dict(zip(profiles["symbol"], profiles["sector"]))
    except Exception as exc:
        log.debug("profiles.parquet not available: %s", exc)
        return {}

def _cs_normalize(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)

def build_alpha_stack(features, alpha_weights=None, regime_signal=None, agent_id: str = "us_equities"):
    """
    Build composite alpha from all enabled sleeves.

    Args:
        features:       Dict of feature DataFrames
        alpha_weights:  Optional override for sleeve mixing weights
        regime_signal:  Optional regime signal for regime-aware weight adjustment
        agent_id:       Agent identifier passed to ML sleeve for model cache key.
                        Each specialist agent maintains its own trained XGBoost model.
                        Defaults to "us_equities" for the current single-agent setup.
    """
    if alpha_weights is None:
        regime_label = None
        if regime_signal is not None:
            try:
                regime_label = str(regime_signal.label.value).lower()
            except Exception:
                pass
        alpha_weights = _load_active_alpha_weights(regime=regime_label)
    if regime_signal is not None:
        try:
            from ascent.regime import regime_adjust_sleeve_weights
            alpha_weights = regime_adjust_sleeve_weights(
                base_sleeve_weights=alpha_weights, signal=regime_signal)
            log.info("alpha_stack: regime=%s", regime_signal.label.value)
        except Exception as exc:
            log.warning("regime adjustment failed: %s", exc)
    alphas = {}
    try:
        trend = trend_alpha(features)
        if not trend.empty:
            alphas["trend"] = trend
            log.info("trend alpha loaded shape=%s", trend.shape)
    except Exception as exc:
        log.error("trend alpha failed: %s", exc)
    try:
        mr = meanrev_alpha(features)
        if not mr.empty:
            alphas["meanrev"] = mr
            log.info("meanrev alpha loaded shape=%s", mr.shape)
    except Exception as exc:
        log.error("meanrev alpha failed: %s", exc)
    if "vol_of_vol_21d" in features and "vol_trend_10d" in features:
        try:
            # Vol-regime alpha: long names with declining vol AND stable vol-of-vol.
            # Signal = -(vol_trend) / (vol_of_vol + epsilon)
            # Positive when vol is falling stably; orthogonal to raw momentum.
            vov   = features["vol_of_vol_21d"].copy().replace(0, np.nan)
            vtrnd = features["vol_trend_10d"].copy()
            vol_alpha = _cs_normalize(-vtrnd / (vov + 1e-6))
            alphas["volatility"] = vol_alpha
            log.info("vol-regime alpha loaded shape=%s", vol_alpha.shape)
        except Exception as exc:
            log.error("vol-regime alpha failed: %s", exc)
    elif "vol_21d" in features:
        try:
            vol_alpha = -_cs_normalize(features["vol_21d"].copy())
            alphas["volatility"] = vol_alpha
            log.info("volatility alpha (low-vol fallback) loaded shape=%s", vol_alpha.shape)
        except Exception as exc:
            log.error("volatility alpha failed: %s", exc)
    try:
        sector_map = _load_sector_map()
        sa = statarb_alpha(features, sector_map=sector_map)
        if not sa.empty:
            alphas["statarb"] = sa
            log.info("statarb alpha loaded shape=%s", sa.shape)
    except Exception as exc:
        log.error("statarb alpha failed: %s", exc)
    try:
        if "targets" in features:
            targets_df = features["targets"]
            # Phase 3a: CPCV replaces the 80/20 split.
            # build_ml_alpha_cpcv uses C(6,2)=15 purged folds with 5-day embargo.
            # Returns empty DF (and logs reason) if reliability guards fail.
            ml = build_ml_alpha_cpcv(
                features=features,
                targets=targets_df,
                agent_id=agent_id,
            )
            if not ml.empty:
                alphas["ml"] = ml
                log.info("ML sleeve loaded shape=%s (CPCV OOS)", ml.shape)
            else:
                log.warning("ML sleeve returned empty (CPCV reliability guard or insufficient data)")
        else:
            log.warning("ML sleeve skipped - targets not in features dict")
    except Exception as exc:
        log.warning("ML sleeve failed: %s", exc)
    try:
        from ascent.alpha.fundamental import fundamental_alpha
        fund = fundamental_alpha(features)
        if fund is not None and not fund.empty:
            alphas["fundamental"] = fund
            log.info("fundamental alpha loaded shape=%s", fund.shape)
        else:
            log.warning("fundamental alpha returned empty")
    except Exception as exc:
        log.error("fundamental alpha failed: %s", exc)
    try:
        from ascent.alpha.earnings import earnings_alpha
        earn = earnings_alpha(features)
        if earn is not None and not earn.empty:
            alphas["earnings"] = earn
            log.info("earnings alpha (PEAD) loaded shape=%s", earn.shape)
        else:
            log.debug("earnings alpha returned empty — cache absent or no recent surprises")
    except Exception as exc:
        log.error("earnings alpha failed: %s", exc)
    try:
        from ascent.alpha.analyst import analyst_alpha
        anl = analyst_alpha(features)
        if anl is not None and not anl.empty:
            alphas["analyst"] = anl
            log.info("analyst revision alpha loaded shape=%s", anl.shape)
        else:
            log.debug("analyst alpha returned empty — cache absent or no revision data")
    except Exception as exc:
        log.error("analyst alpha failed: %s", exc)
    loaded = list(alphas.keys())
    skipped = [k for k in alpha_weights if k not in loaded]
    print(f"[alpha_stack] loaded={loaded}  skipped={skipped}")
    if not alphas:
        raise ValueError("No alpha signals could be computed")
    total_w = sum(alpha_weights.get(k, 0.0) for k in alphas)
    if total_w == 0:
        total_w = 1.0
    composite = None
    for name, alpha_df in alphas.items():
        w = alpha_weights.get(name, 0.0) / total_w
        # Normalize every sleeve to the same cross-sectional z-score scale before
        # blending. Without this, percentile-ranked sleeves (0-1) and z-scored
        # sleeves mix at different scales and the blend is not a true weighted average.
        normed = _cs_normalize(alpha_df)
        if composite is None:
            composite = normed * w
        else:
            union_idx  = composite.index.union(normed.index)
            union_cols = composite.columns.union(normed.columns)
            composite  = composite.reindex(index=union_idx, columns=union_cols).fillna(0.0)
            normed_r   = normed.reindex(index=union_idx, columns=union_cols).fillna(0.0)
            composite  = composite + normed_r * w
    # Distressed name filter: zero out alpha for names down >65% over the past year.
    # Prevents momentum from picking up recovery bounces in deeply impaired names.
    if composite is not None and "mom_252d" in features:
        annual_ret = features["mom_252d"].reindex(
            index=composite.index, columns=composite.columns
        )
        distressed = annual_ret < -0.65
        composite = composite.where(~distressed, 0.0)
        n_filtered = int(distressed.iloc[-1].sum()) if len(distressed) > 0 else 0
        if n_filtered > 0:
            names = list(distressed.columns[distressed.iloc[-1]])
            log.warning("distressed filter: zeroed %d names on latest date: %s", n_filtered, names)

    return composite

def alpha_to_ranks(alpha):
    return alpha.rank(axis=1, pct=True)
