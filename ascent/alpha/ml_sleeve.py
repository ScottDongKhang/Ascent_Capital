"""
ML alpha sleeve - XGBoost cross-sectional return predictor.
Caches the trained model to disk and retrains only every 21 trading days.

Phase 3a: CPCV replaces the 80/20 in-sample split.
  build_ml_alpha_cpcv() uses CPCVSplitter(n_splits=6, n_test_splits=2, purge=5, embargo=5)
  → C(6,2)=15 OOS folds, each date appears in C(5,1)=5 test folds.
  Two guards:
    1. n_converged < 10 → ML sleeve disabled (biased estimate)
    2. p5 Sharpe < 0    → ML sleeve disabled (unreliable OOS)
"""
import os
import pickle
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

ML_RETRAIN_INTERVAL_DAYS = 21

ML_FEATURES = [
    "mom_21d", "mom_63d", "mom_126d",
    "vol_21d", "vol_63d", "vol_ratio_10_63",
    "zscore_20d", "rsi_14", "macd_hist",
    "dollar_vol_rank_21d",
    "high_52w_pct",
    "gross_profitability",
    "accruals",
    "asset_growth",
]


# ── Model cache helpers ───────────────────────────────────────────────────────

def _ml_cache_path(agent_id: str = "us_equities") -> Path:
    """Return path to the cached ML model file."""
    return Path("data_cache") / f"ml_model_{agent_id}.pkl"


def _load_cached_model(agent_id: str = "us_equities"):
    """
    Load cached XGBoost model if it exists.
    Returns (model, train_date) or (None, None).
    """
    cache_path = _ml_cache_path(agent_id)
    if not cache_path.exists():
        return None, None
    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        model = cached["model"]
        train_date = cached["train_date"]
        return model, train_date
    except Exception as e:
        print(f"[ML] Cache load failed ({e}), will retrain")
        return None, None


def _save_cached_model(model, train_date, agent_id: str = "us_equities"):
    """Persist trained model with its training date."""
    cache_path = _ml_cache_path(agent_id)
    os.makedirs(cache_path.parent, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"model": model, "train_date": train_date}, f)
    print(f"[ML] Saved model cache to {cache_path} (trained on {train_date})")


def _is_cache_fresh(cached_train_date, current_date) -> bool:
    """
    Return True if the cached model is within ML_RETRAIN_INTERVAL_DAYS business days.
    Both dates can be str (YYYY-MM-DD) or date/Timestamp.
    """
    try:
        t0 = pd.Timestamp(cached_train_date)
        t1 = pd.Timestamp(current_date)
        if t1 < t0:
            return False
        days_since_train = len(pd.bdate_range(t0, t1)) - 1  # inclusive → exclusive end
        return days_since_train <= ML_RETRAIN_INTERVAL_DAYS
    except Exception:
        return False


# ── Cross-sectional normalizer ────────────────────────────────────────────────

def _cs_normalize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, 1), axis=0)


# ── Main alpha function ───────────────────────────────────────────────────────

def build_ml_alpha(
    features,
    targets,
    train_end,
    predict_start,
    predict_end,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=20,
    agent_id: str = "us_equities",
):
    """
    Build ML alpha using XGBoost.

    Retrains only when the cached model is more than ML_RETRAIN_INTERVAL_DAYS
    business days old. Otherwise loads the cached model and skips training.

    Args:
        features:       Dict of feature DataFrames (wide: date × symbol)
        targets:        DataFrame of forward returns (wide: date × symbol)
        train_end:      Last date available for training (str YYYY-MM-DD)
        predict_start:  First date to predict (str YYYY-MM-DD)
        predict_end:    Last date to predict (str YYYY-MM-DD)
        n_estimators:   XGBoost trees (default 200, pinned for v1.7.6 compat)
        max_depth:      XGBoost max depth
        learning_rate:  XGBoost learning rate
        min_child_weight: XGBoost min child weight
        agent_id:       Cache key — one model file per agent

    Returns:
        pd.DataFrame of cross-sectional alpha scores (date × symbol), or empty DF.
    """
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed")
        return pd.DataFrame()

    available = [f for f in ML_FEATURES if f in features]
    if len(available) < 4:
        log.warning("ML sleeve: only %d features available, skipping", len(available))
        return pd.DataFrame()

    # Stack all features into long format — one row per (date, symbol)
    long_frames = []
    for feat_name in available:
        stacked = features[feat_name].stack().rename(feat_name)
        long_frames.append(stacked)
    X_all = pd.concat(long_frames, axis=1).dropna()

    # Stack targets — NaN where forward return is not yet known
    y_all = targets.stack().rename("fwd_ret")

    # Training rows: must have both features AND a known target, and be <= train_end
    train_aligned = X_all.join(y_all).dropna()
    train_mask = train_aligned.index.get_level_values(0) <= train_end
    train_data = train_aligned[train_mask]

    # Prediction rows: features only — NaN targets do NOT disqualify a row
    pred_mask = (
        (X_all.index.get_level_values(0) >= predict_start) &
        (X_all.index.get_level_values(0) <= predict_end)
    )
    pred_data = X_all[pred_mask]

    if len(train_data) < 500:
        log.warning("ML sleeve: only %d training rows, skipping", len(train_data))
        return pd.DataFrame()

    if len(pred_data) == 0:
        log.warning("ML sleeve: no prediction rows in [%s, %s]", predict_start, predict_end)
        return pd.DataFrame()

    X_pred = pred_data[available].values

    # ── Cache check: skip training if model is fresh enough ──────────────────
    cached_model, cached_train_date = _load_cached_model(agent_id)

    if cached_model is not None and cached_train_date is not None:
        if _is_cache_fresh(cached_train_date, predict_end):
            days_since = len(pd.bdate_range(pd.Timestamp(cached_train_date), pd.Timestamp(predict_end))) - 1
            print(f"[ML] Using cached model (trained {days_since} bdays ago on {cached_train_date}, agent={agent_id})")
            model = cached_model
        else:
            days_since = len(pd.bdate_range(pd.Timestamp(cached_train_date), pd.Timestamp(predict_end))) - 1
            print(f"[ML] Cache stale ({days_since} bdays old), retraining (agent={agent_id})")
            model = None
    else:
        print(f"[ML] No cache found, training fresh model (agent={agent_id})")
        model = None

    # ── Train if needed ───────────────────────────────────────────────────────
    if model is None:
        X_train = train_data[available].values
        y_train = train_data["fwd_ret"].values

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_train, y_train)
        _save_cached_model(model, train_end, agent_id)

        log.info(
            "ML sleeve: trained on %d rows, predicting %d dates, %d symbols",
            len(train_data), len(pred_data.index.get_level_values(0).unique()),
            len(pred_data.index.get_level_values(1).unique()),
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    preds = model.predict(X_pred)

    pred_series = pd.Series(preds, index=pred_data.index, name="ml_alpha")
    score_wide  = pred_series.unstack(level=1)
    score_norm  = _cs_normalize(score_wide).clip(-3, 3)

    log.info(
        "ML sleeve: predicted %d dates, %d symbols",
        score_wide.shape[0], score_wide.shape[1],
    )
    return score_norm


# ── CPCV helpers ──────────────────────────────────────────────────────────────

def _stack_features(features: dict, available: list) -> pd.DataFrame:
    """Stack feature dict into long-form (date, symbol) MultiIndex DataFrame."""
    long_frames = []
    for feat_name in available:
        stacked = features[feat_name].stack().rename(feat_name)
        long_frames.append(stacked)
    return pd.concat(long_frames, axis=1).dropna()


def _train_xgboost(features: dict, targets: pd.DataFrame, train_dates: pd.DatetimeIndex):
    """
    Train an XGBoost model on the given training dates.
    Returns a fitted XGBRegressor.
    """
    import xgboost as xgb
    available = [f for f in ML_FEATURES if f in features]
    X_all = _stack_features(features, available)
    y_all = targets.stack().rename("fwd_ret")
    train_aligned = X_all.join(y_all).dropna()
    # Filter to training dates
    train_mask = train_aligned.index.get_level_values(0).isin(train_dates)
    train_data = train_aligned[train_mask]
    if len(train_data) < 100:
        raise ValueError(f"Only {len(train_data)} training rows — too few to train")
    X_train = train_data[available].values
    y_train = train_data["fwd_ret"].values
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        verbosity=0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _compute_fold_ic(model, features: dict, targets: pd.DataFrame,
                     test_dates: pd.DatetimeIndex) -> float:
    """
    Compute fold Information Coefficient: rank correlation between predictions
    and realized forward returns on test_dates.
    Returns float IC (approximately annualised Sharpe proxy).
    """
    available = [f for f in ML_FEATURES if f in features]
    X_all = _stack_features(features, available)
    y_all = targets.stack().rename("fwd_ret")
    test_mask = X_all.index.get_level_values(0).isin(test_dates)
    X_test = X_all[test_mask]
    if X_test.empty:
        return 0.0
    preds = model.predict(X_test[available].values)
    pred_series = pd.Series(preds, index=X_test.index)
    target_series = y_all.reindex(X_test.index).dropna()
    if target_series.empty:
        return 0.0
    aligned = pred_series.reindex(target_series.index).dropna()
    if len(aligned) < 10:
        return 0.0
    from scipy.stats import spearmanr
    ic, _ = spearmanr(aligned.values, target_series.reindex(aligned.index).values)
    return float(ic) if not np.isnan(ic) else 0.0


# ── CPCV main function ────────────────────────────────────────────────────────

def build_ml_alpha_cpcv(
    features: dict,
    targets: pd.DataFrame,
    agent_id: str = "us_equities",
    splitter=None,
) -> pd.DataFrame:
    """
    Build ML alpha using Combinatorial Purged Cross-Validation (CPCV).

    Replaces the 80/20 in-sample split. Uses C(6,2)=15 train/test folds.
    Each date's prediction = mean of the C(5,1)=5 models that covered it as OOS.

    Two reliability guards:
      1. n_converged < 10  → ML sleeve disabled (estimate too noisy)
      2. p5 Sharpe < 0    → ML sleeve disabled (unreliable OOS)

    Final production model is trained on 80% of dates for live prediction
    (same as before, but only after OOS reliability is confirmed).

    Returns
    -------
    pd.DataFrame of cross-sectional alpha scores (dates × symbols), or empty DF.
    """
    try:
        import xgboost  # noqa: F401
    except ImportError:
        log.error("[ML-CPCV] xgboost not installed")
        return pd.DataFrame()

    if splitter is None:
        from ascent.research.cpcv import CPCVSplitter
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)

    available = [f for f in ML_FEATURES if f in features]
    if len(available) < 4:
        log.warning("[ML-CPCV] only %d features available (need ≥4), skipping", len(available))
        return pd.DataFrame()

    if targets.empty:
        log.warning("[ML-CPCV] targets is empty, skipping")
        return pd.DataFrame()

    dates = pd.DatetimeIndex(sorted(targets.index.unique()))
    if len(dates) < 100:
        log.warning("[ML-CPCV] only %d dates (need ≥100), skipping", len(dates))
        return pd.DataFrame()

    all_predictions: dict = {}   # date → list of per-symbol prediction arrays
    fold_ics: list[float] = []
    n_converged = 0
    n_total = splitter.n_splits_total()

    for fold_idx, (train_dates, test_dates) in enumerate(splitter.split(dates)):
        if len(train_dates) < 50 or len(test_dates) < 5:
            log.debug("[ML-CPCV] fold %d skipped — too few dates", fold_idx)
            continue
        try:
            model = _train_xgboost(features, targets, train_dates)
            ic = _compute_fold_ic(model, features, targets, test_dates)
            fold_ics.append(ic)
            n_converged += 1

            # Store predictions for each test date
            available_feats = [f for f in ML_FEATURES if f in features]
            X_all = _stack_features(features, available_feats)
            test_mask = X_all.index.get_level_values(0).isin(test_dates)
            X_test = X_all[test_mask]
            if not X_test.empty:
                preds = model.predict(X_test[available_feats].values)
                pred_series = pd.Series(preds, index=X_test.index)
                for date, grp in pred_series.groupby(level=0):
                    all_predictions.setdefault(date, []).append(grp)

        except Exception as exc:
            log.warning("[ML-CPCV] fold %d failed: %s", fold_idx, exc)
            continue

    # Guard: too many failed folds
    if n_converged < 10:
        log.warning(
            "[ML-CPCV] only %d/%d folds converged — ML sleeve disabled "
            "(biased estimate from partial CPCV)", n_converged, n_total
        )
        return pd.DataFrame()

    # Guard: p5 Sharpe < 0 = unreliable OOS
    sharpe_p5 = float(np.percentile(fold_ics, 5))
    sharpe_p50 = float(np.percentile(fold_ics, 50))
    log.info(
        "[ML-CPCV] %d/%d folds converged — IC Sharpe p5=%.3f p50=%.3f",
        n_converged, n_total, sharpe_p5, sharpe_p50
    )

    if sharpe_p5 < 0:
        log.warning(
            "[ML-CPCV] p5 IC=%.3f < 0 — ML sleeve disabled (unreliable OOS)",
            sharpe_p5
        )
        return pd.DataFrame()

    # Average predictions across folds that covered each date
    avg_preds: dict = {}
    for date, pred_list in all_predictions.items():
        # Stack all fold predictions for this date and mean them
        combined = pd.concat(pred_list)
        avg_preds[date] = combined.groupby(level=1).mean()

    if not avg_preds:
        log.warning("[ML-CPCV] no OOS predictions produced — ML sleeve disabled")
        return pd.DataFrame()

    # Build wide prediction DataFrame (dates × symbols)
    pred_df = pd.DataFrame(avg_preds).T
    pred_df.index = pd.DatetimeIndex(pred_df.index)
    pred_df = pred_df.sort_index()

    # Normalize cross-sectionally and cache final production model
    score_norm = _cs_normalize(pred_df).clip(-3, 3)

    # Train final production model on most recent 80% for live prediction
    n_final_train = int(len(dates) * 0.80)
    final_train_dates = dates[:n_final_train]
    try:
        final_model = _train_xgboost(features, targets, final_train_dates)
        cache_path = _ml_cache_path(agent_id)
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "model": final_model,
                "train_date": str(dates[-1])[:10],
                "cpcv_meta": {
                    "sharpe_p5": sharpe_p5,
                    "sharpe_p50": sharpe_p50,
                    "n_converged": n_converged,
                    "n_splits": splitter.n_splits,
                    "n_test_splits": splitter.n_test_splits,
                    "purge_days": splitter.purge_days,
                    "embargo_days": splitter.embargo_days,
                },
            }, f)
        log.info("[ML-CPCV] cached final production model (trained on %d dates)", len(final_train_dates))
    except Exception as exc:
        log.warning("[ML-CPCV] final model cache failed: %s", exc)

    log.info(
        "[ML-CPCV] returning OOS alpha: %d dates, %d symbols",
        score_norm.shape[0], score_norm.shape[1]
    )
    return score_norm
