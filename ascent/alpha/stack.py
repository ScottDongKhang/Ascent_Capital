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
    "meanrev": 0.50,
    "statarb": 0.50,
}

IC_GATE_THRESHOLD = -0.005  # tightened from -0.010: fundamental IC=-0.008 recently evaded the gate


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

    # Priority: config by_regime → meta-learner → config global → flat default
    # (DEFAULT_ALPHA_WEIGHTS_BY_REGIME was removed: with only meanrev/statarb live,
    # there is nothing left to regime-tilt between.)
    if config_regime_weights is not None:
        return config_regime_weights

    if regime:
        regime_key = str(regime).lower()
        try:
            from ascent.alpha.meta_learner import SleeveMetaLearner, log_weight_proposal
            ml = SleeveMetaLearner()
            ml_weights = ml.get_weights(regime_key, DEFAULT_ALPHA_WEIGHTS, ai_prior=ai_prior)
            if ml_weights is not None:
                log.info("_load_active_alpha_weights: meta-learner weights for regime=%s", regime_key)
                log_weight_proposal(regime_key, ml_weights, "meta_learner")
                return ml_weights
        except Exception as exc:
            log.warning("_load_active_alpha_weights: meta-learner error (%s) — using regime defaults", exc)

    if config_global_weights is not None:
        return config_global_weights
    return DEFAULT_ALPHA_WEIGHTS.copy()


def _get_gated_weights(
    alpha_weights: dict,
    ic_log_path: str = "logs/sleeve_ic_log.jsonl",
    window: int = 5,
) -> dict:
    """
    Read last `window` unique-date entries from sleeve_ic_log.jsonl.
    Zero out any sleeve whose rolling mean IC < IC_GATE_THRESHOLD.
    Freed weight is redistributed proportionally among the surviving
    (non-gated, non-zero) sleeves, by their existing weight share. If every
    sleeve ends up gated (no survivors), the original alpha_weights are
    returned unchanged rather than producing an all-zero dict.
    """
    import json as _json
    from collections import defaultdict as _defaultdict
    from pathlib import Path as _Path

    log_path = _Path(ic_log_path)
    if not log_path.exists():
        return alpha_weights

    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    seen, recent = set(), []
    for line in reversed(lines):
        try:
            entry = _json.loads(line)
            d = entry.get("date", "")
            if d and d not in seen:
                seen.add(d)
                recent.append(entry)
            if len(recent) >= window:
                break
        except Exception:
            continue

    if not recent:
        return alpha_weights

    sleeve_ics: dict = _defaultdict(list)
    for entry in recent:
        for sleeve, stats in entry.get("sleeves", {}).items():
            ic = stats.get("mean_ic")
            if ic is not None:
                sleeve_ics[sleeve].append(float(ic))

    gated = {
        sleeve: sum(ics) / len(ics)
        for sleeve, ics in sleeve_ics.items()
        if len(ics) >= window and sum(ics) / len(ics) < IC_GATE_THRESHOLD
    }

    if not gated:
        return alpha_weights

    result = dict(alpha_weights)
    freed = sum(result.get(s, 0.0) for s in gated)
    for sleeve, avg_ic in gated.items():
        log.warning(
            "[Stack] IC gate: zeroing %s (rolling mean_ic=%.4f < %.3f)",
            sleeve, avg_ic, IC_GATE_THRESHOLD,
        )
        result[sleeve] = 0.0

    survivors = {s: w for s, w in result.items() if s not in gated and w > 0}
    survivors_total = sum(survivors.values())

    if freed > 0 and survivors_total > 0:
        for sleeve, weight in survivors.items():
            result[sleeve] = round(weight + freed * (weight / survivors_total), 4)
    elif freed > 0 and survivors_total <= 0:
        # Every sleeve gated: fail safe, don't return an all-zero dict.
        log.warning(
            "[Stack] IC gate: all sleeves gated (%s) -- keeping original weights unchanged",
            sorted(gated),
        )
        return alpha_weights

    return result


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

def build_alpha_stack(features, alpha_weights=None, regime_signal=None, agent_id: str = "us_equities", ai_prior: dict = None, return_breakdown: bool = False):
    """
    Build composite alpha from all enabled sleeves.

    Args:
        features:          Dict of feature DataFrames
        alpha_weights:     Optional override for sleeve mixing weights
        regime_signal:     Unused (kept for call-site compatibility). Regime-conditional
                           weight adjustment was removed; regime_overlay scored CUT
                           (no proven value) in the proof audit.
        agent_id:          Agent identifier passed to ML sleeve for model cache key.
        ai_prior:          AI PM sleeve weight adjustments (global IC deltas).
        return_breakdown:  If True, return (composite, breakdown_dict) where
                           breakdown_dict = {
                               "per_sleeve": {sleeve: last_row_series},
                               "convergence": {symbol: float},   # 0-1, fraction of sleeves agreeing
                               "primary_sleeve": {symbol: str},  # sleeve with highest contribution
                           }
    """
    # regime_signal is accepted for call-site compatibility but no longer used to
    # adjust sleeve weights: regime_overlay scored CUT (p=0.35, no proven value) in
    # the proof audit, so build_alpha_stack() always uses the flat
    # DEFAULT_ALPHA_WEIGHTS (or an explicit override). See
    # ascent.regime.integration.apply_regime_to_portfolio for the separate,
    # in-scope regime overlay on the backtest path.
    if alpha_weights is None:
        alpha_weights = _load_active_alpha_weights(ai_prior=ai_prior)
        alpha_weights = _get_gated_weights(alpha_weights)
    alphas = {}
    if alpha_weights.get("trend", 0) > 0:
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
    if alpha_weights.get("volatility", 0) > 0:
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
    if alpha_weights.get("fundamental", 0) > 0:
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
    if alpha_weights.get("earnings", 0) > 0:
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
    # LLM Fundamental sleeve (Chicago Booth 6-step CoT via Haiku)
    if alpha_weights.get("llm_fundamental", 0) > 0:
        try:
            from ascent.alpha.llm_fundamental import llm_fundamental_alpha
            from ascent.data.store.parquet import has_data as _hd, load_parquet as _lp
            _fund_df = _lp("fundamentals") if _hd("fundamentals") else None
            _llm_fund = llm_fundamental_alpha(_fund_df)
            # llm_fundamental_alpha returns a cross-sectional (per-symbol) snapshot
            # Series, unlike the date×symbol DataFrames the blend/_cs_normalize expect.
            # Broadcast it across the price panel's dates so it integrates like the
            # other fundamentals-derived sleeves (cf. fundamental_alpha).
            if isinstance(_llm_fund, pd.Series):
                _close = features.get("close")
                if _close is not None and not _close.empty and not _llm_fund.empty:
                    _row = _llm_fund.reindex(_close.columns).to_numpy(dtype=float)
                    _llm_fund = pd.DataFrame(
                        np.tile(_row, (len(_close.index), 1)),
                        index=_close.index, columns=_close.columns,
                    )
                else:
                    _llm_fund = pd.DataFrame()
            if not _llm_fund.empty:
                alphas["llm_fundamental"] = _llm_fund
                log.info("[Stack] LLM fundamental sleeve: %d symbols", _llm_fund.shape[1])
            else:
                log.debug("[Stack] LLM fundamental sleeve returned empty — cache absent or API unavailable")
        except Exception as _e:
            log.warning("[Stack] LLM fundamental sleeve failed: %s", _e)
    if alpha_weights.get("analyst", 0) > 0:
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
    if alpha_weights.get("options_flow", 0) > 0:
        try:
            from ascent.alpha.options_flow import options_flow_alpha
            opt = options_flow_alpha(features)
            if opt is not None and not opt.empty:
                alphas["options_flow"] = opt
                log.info("options flow alpha loaded shape=%s", opt.shape)
            else:
                log.debug("options flow alpha returned empty — cache absent or insufficient data")
        except Exception as exc:
            log.error("options flow alpha failed: %s", exc)
    if alpha_weights.get("insider", 0) > 0:
        try:
            from ascent.alpha.insider import insider_alpha
            ins = insider_alpha(features)
            if ins is not None and not ins.empty:
                alphas["insider"] = ins
                log.info("insider alpha loaded shape=%s", ins.shape)
            else:
                log.debug("insider alpha returned empty — cache absent or no transactions")
        except Exception as exc:
            log.error("insider alpha failed: %s", exc)
    if alpha_weights.get("short_interest", 0) > 0:
        try:
            from ascent.alpha.short_interest import short_interest_alpha
            si = short_interest_alpha(features)
            if si is not None and not si.empty:
                alphas["short_interest"] = si
                log.info("short interest alpha loaded shape=%s", si.shape)
            else:
                log.debug("short interest alpha returned empty — cache absent or no data")
        except Exception as exc:
            log.error("short interest alpha failed: %s", exc)
    # Alternative data sleeve — active only when sources pass IC gate
    if alpha_weights.get("altdata", 0) > 0:
        try:
            from ascent.alpha.altdata_alpha import altdata_alpha
            alt = altdata_alpha(features=features)
            if alt is not None and not alt.empty:
                alphas["altdata"] = alt
                log.info("[Stack] altdata sleeve loaded shape=%s", alt.shape)
            else:
                log.debug("[Stack] altdata sleeve returned empty — no validated sources")
        except Exception as exc:
            log.warning("[Stack] altdata sleeve failed: %s", exc)
    # Earnings-tone alpha — offline panel built weekly; parquet read only in hot path
    if alpha_weights.get("earnings_tone", 0) > 0:
        try:
            from ascent.alpha.earnings_tone import earnings_tone_alpha
            et = earnings_tone_alpha(features)
            if et is not None and not et.empty:
                alphas["earnings_tone"] = et
                log.info("[Stack] earnings_tone sleeve loaded shape=%s", et.shape)
            else:
                log.debug("[Stack] earnings_tone sleeve returned empty — cache absent or no data")
        except Exception as exc:
            log.warning("[Stack] earnings_tone sleeve failed: %s", exc)
    # Narrative alpha — 0% weight until narrative cache has ≥30 days history
    if alpha_weights.get("narrative", 0) > 0:
        try:
            from ascent.alpha.narrative_alpha import build_narrative_alpha
            _symbols = list(features.get("mom_252d", pd.DataFrame()).columns) if "mom_252d" in features else []
            if _symbols:
                narr = build_narrative_alpha(_symbols)
                if narr is not None and not narr.empty:
                    # Convert Series to DataFrame matching features shape if needed
                    if isinstance(narr, pd.Series):
                        # Stack needs a DataFrame indexed by date×symbol; create a single-row DF
                        if "mom_252d" in features:
                            _idx = features["mom_252d"].index
                            narr_df = pd.DataFrame(
                                {sym: narr.get(sym, 0.0) for sym in narr.index},
                                index=_idx,
                            )
                        else:
                            narr_df = narr.to_frame().T
                        alphas["narrative"] = narr_df
                    else:
                        alphas["narrative"] = narr
                    log.info("[Stack] narrative alpha sleeve: %d symbols", len(narr))
            else:
                log.debug("[Stack] narrative alpha skipped — no symbols available")
        except Exception as exc:
            log.warning("[Stack] narrative_alpha failed: %s", exc)
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

    if not return_breakdown:
        return composite

    # ── Build per-sleeve breakdown for AI PM two-way feedback ──
    # Only computed when explicitly requested (zero cost on normal runs).
    breakdown: dict = {"per_sleeve": {}, "convergence": {}, "primary_sleeve": {}}
    if composite is not None and len(composite) > 0 and alphas:
        last_composite = composite.iloc[-1]
        normed_sleeves: dict = {}
        for name, alpha_df in alphas.items():
            try:
                normed = _cs_normalize(alpha_df)
                normed_sleeves[name] = normed.iloc[-1] if len(normed) > 0 else pd.Series(dtype=float)
            except Exception:
                pass

        # Per-sleeve last-row scores
        breakdown["per_sleeve"] = {
            name: row.to_dict() for name, row in normed_sleeves.items()
        }

        # Convergence: fraction of sleeves that agree with the composite sign per symbol
        all_syms = list(last_composite.index)
        for sym in all_syms:
            composite_sign = np.sign(last_composite.get(sym, 0.0))
            if composite_sign == 0:
                breakdown["convergence"][sym] = 0.5
                continue
            agreeing = sum(
                1 for row in normed_sleeves.values()
                if np.sign(row.get(sym, 0.0)) == composite_sign
            )
            breakdown["convergence"][sym] = round(agreeing / max(len(normed_sleeves), 1), 3)

        # Primary sleeve: highest absolute contribution per symbol
        for sym in all_syms:
            best_sleeve, best_val = "trend", 0.0
            for name, row in normed_sleeves.items():
                val = abs(row.get(sym, 0.0)) * alpha_weights.get(name, 0.0)
                if val > best_val:
                    best_val, best_sleeve = val, name
            breakdown["primary_sleeve"][sym] = best_sleeve

    return composite, breakdown

def alpha_to_ranks(alpha):
    return alpha.rank(axis=1, pct=True)
