"""
ascent/regime/engine.py
------------------------
Top-level RegimeEngine that orchestrates all subsystem layers.

This is the single object that main.py / stack.py imports.
It handles:
  • Feature building
  • Walk-forward model selection (on first use)
  • Rolling refits on schedule
  • Probability generation
  • Hysteresis decision layer
  • Signal caching

Usage pattern (from main.py or walk-forward runner):

    engine = RegimeEngine(config)
    engine.fit(spy_prices, universe_prices, vix_prices, macro_df)

    # At each rebalance date:
    signal = engine.get_signal(as_of_date)

    # Apply to portfolio:
    weights, meta = apply_regime_to_portfolio(raw_weights, signal)

    # For walk-forward: call engine.fit() on training slice, then
    # engine.get_signal() on each test date.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import RegimeFeatureBuilder
from .model import RegimeModel, walk_forward_model_select
from .breaks import BreakDetector
from .decision import RegimeDecisionEngine
from .integration import get_signal_for_date, apply_regime_to_portfolio
from .types import (
    RegimeSignal, RegimeLabel, RegimeScorecard, REGIME_CONFIG_DEFAULTS
)

log = logging.getLogger(__name__)

EMERGENCY_SPY_RETURN_THRESHOLD = -0.03    # -3% 1-day return
EMERGENCY_VIX_THRESHOLD = 30.0           # VIX above this with crash = trigger
EMERGENCY_CORR_THRESHOLD = 0.30          # SPY/TLT 5-day corr > this = liquidity crisis
EMERGENCY_BREAK_ZSCORE_THRESHOLD = 3.5   # BreakDetector z-score threshold
EMERGENCY_REFIT_LOG = "logs/regime_emergency_refit.jsonl"


def check_emergency_refit_triggers(
    spy: pd.Series,
    tlt: pd.Series,
    vix: pd.Series,
    breaks_detector,
) -> tuple[bool, str]:
    """
    Check if any emergency refit condition is met.

    Returns (should_refit: bool, reason: str).

    Triggers (any one fires):
      1. SPY 1-day return < -3% AND VIX > 30
      2. SPY crosses below its 200-day MA from above
      3. SPY/TLT 5-day return correlation > +0.30 (both selling = liquidity crisis)
      4. Structural break z-score from breaks_detector > 3.5σ

    Parameters
    ----------
    spy : pd.Series  SPY close prices (chronological)
    tlt : pd.Series  TLT close prices (chronological)
    vix : pd.Series  VIX levels (chronological)
    breaks_detector  BreakDetector instance with latest_zscore() method
    """
    if len(spy) < 201:
        return False, ""

    # Trigger 1: crash + fear
    spy_ret_1d = spy.pct_change().iloc[-1]
    vix_current = float(vix.iloc[-1])
    if spy_ret_1d < EMERGENCY_SPY_RETURN_THRESHOLD and vix_current > EMERGENCY_VIX_THRESHOLD:
        return True, f"SPY {spy_ret_1d:.1%} + VIX {vix_current:.0f}"

    # Trigger 2: bear market entry (SPY crosses below 200-day MA)
    ma200 = spy.rolling(200).mean()
    if len(ma200.dropna()) >= 2:
        prev_above = spy.iloc[-2] > ma200.iloc[-2]
        now_below = spy.iloc[-1] < ma200.iloc[-1]
        if prev_above and now_below:
            return True, (
                f"SPY crossed below 200MA "
                f"({spy.iloc[-1]:.2f} < {ma200.iloc[-1]:.2f})"
            )

    # Trigger 3: correlation flip — both SPY and TLT falling = liquidity crisis
    # Compute pct_change on full series first, then align last 5 days
    spy_rets = spy.pct_change().dropna()
    tlt_rets = tlt.pct_change().dropna()
    shared_idx = spy_rets.index.intersection(tlt_rets.index)[-5:]
    if len(shared_idx) >= 5:
        spy_5d = spy_rets.reindex(shared_idx)
        tlt_5d = tlt_rets.reindex(shared_idx)
        corr = float(spy_5d.corr(tlt_5d))
        if not np.isnan(corr) and corr > EMERGENCY_CORR_THRESHOLD:
            return True, f"SPY/TLT 5-day correlation={corr:.2f} (liquidity crisis signal)"

    # Trigger 4: statistical structural break
    try:
        if hasattr(breaks_detector, "latest_zscore"):
            # Pass feature panel if we can, otherwise use None (will return 0.0)
            break_zscore = breaks_detector.latest_zscore(None) if breaks_detector._feature_panel is None else breaks_detector.latest_zscore(breaks_detector._feature_panel)
        else:
            break_zscore = 0.0
    except Exception:
        break_zscore = 0.0

    if break_zscore > EMERGENCY_BREAK_ZSCORE_THRESHOLD:
        return True, f"Structural break z-score={break_zscore:.1f}σ"

    return False, ""


class RegimeEngine:
    """
    Production regime engine for Ascent Capital.

    Parameters
    ----------
    config : dict, optional
        Override dict for any REGIME_CONFIG_DEFAULTS keys.
    """

    def __init__(self, config: Optional[Dict] = None):
        self._cfg = dict(REGIME_CONFIG_DEFAULTS)
        if config:
            self._cfg.update(config)

        self._model: Optional[RegimeModel] = None
        self._decision_engine: Optional[RegimeDecisionEngine] = None
        self._break_detector = BreakDetector(
            min_size=self._cfg["regime_break_min_size"],
            penalty_multiplier=self._cfg["regime_break_penalty_multiplier"],
        )
        self._signal_cache: Optional[pd.DataFrame] = None
        self._feature_panel: Optional[pd.DataFrame] = None
        self._best_k: int = 3
        self._scorecard: Optional[RegimeScorecard] = None
        self._last_fit_date: Optional[pd.Timestamp] = None
        self._fitted = False
        # Phase 3b: particle filter for daily online updates between batch refits
        self._particle_filter = None
        self._spy_prices: Optional[pd.Series] = None
        self._tlt_prices: Optional[pd.Series] = None
        self._vix_prices: Optional[pd.Series] = None

    # ── public fit API ────────────────────────────────────────────────────

    def fit(
        self,
        spy_prices: pd.Series,
        universe_prices: Optional[pd.DataFrame] = None,
        vix_prices: Optional[pd.Series] = None,
        macro_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.DataFrame] = None,
        fwd_returns: Optional[pd.DataFrame] = None,
        run_model_selection: bool = True,
    ) -> None:
        """
        Build features and fit the regime model.

        Parameters
        ----------
        spy_prices : pd.Series
            SPY (or benchmark) close prices.
        universe_prices : pd.DataFrame, optional
            Cross-sectional universe prices.
        vix_prices : pd.Series, optional
            VIX levels — optional, graceful fallback.
        macro_df : pd.DataFrame, optional
            FRED macro data.
        alpha_scores : pd.DataFrame, optional
            For strategy IC features.
        fwd_returns : pd.DataFrame, optional
            Forward return targets aligned to alpha_scores.
        run_model_selection : bool
            If True, run walk-forward K selection before final fit.
        """
        log.info("regime.engine: starting fit")

        # 1. Build feature panel
        builder = RegimeFeatureBuilder(
            spy_prices=spy_prices,
            universe_prices=universe_prices,
            vix_prices=vix_prices,
            macro_df=macro_df,
            alpha_scores=alpha_scores,
            fwd_returns=fwd_returns,
            lookbacks=tuple(self._cfg["regime_feature_lookbacks"]),
        )
        self._feature_panel = builder.build()

        # 2. Walk-forward model selection
        print("REGIME DEBUG panel_len=", len(self._feature_panel), "min_days=", self._cfg.get("regime_training_min_days"))
        if run_model_selection and len(self._feature_panel) >= self._cfg["regime_training_min_days"]:
            spy_ret = spy_prices.pct_change().reindex(self._feature_panel.index)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_k, scorecard = walk_forward_model_select(
                    feature_panel=self._feature_panel,
                    spy_returns=spy_ret,
                    candidate_k=self._cfg["regime_n_candidates"],
                    train_days=self._cfg["regime_wf_train_days"],
                    test_days=self._cfg["regime_wf_test_days"],
                    step_days=self._cfg["regime_wf_step_days"],
                    min_train=self._cfg["regime_training_min_days"],
                )
            self._best_k = best_k
            self._scorecard = scorecard
            log.info(
                f"regime.engine: model selection chose K={best_k}, "
                f"composite_score={scorecard.composite_score:.4f}"
            )
        else:
            self._best_k = self._cfg["regime_n_candidates"][len(self._cfg["regime_n_candidates"]) // 2]
            log.info(f"regime.engine: skipping model selection, using K={self._best_k} default")

        # 3. Fit production model
        self._model = RegimeModel(k_regimes=self._best_k)
        success = self._model.fit(self._feature_panel)
        if not success:
            log.warning("regime.engine: model fit failed — using uniform probs")

        # 4. Compute probability series and run decision layer
        self._decision_engine = RegimeDecisionEngine(
            state_labels=self._model.state_labels if self._model else {},
            enter_threshold=self._cfg["regime_enter_threshold"],
            exit_threshold=self._cfg["regime_exit_threshold"],
            min_dwell_days=self._cfg["regime_min_dwell_days"],
            entropy_uncertain_threshold=self._cfg["regime_entropy_uncertain_threshold"],
            risk_multipliers=self._cfg["regime_risk_multiplier"],
            sleeve_adjustments=self._cfg["regime_sleeve_adjustments"],
        )

        if self._model and self._model._fitted:
            prob_df = self._model.predict_probs(self._feature_panel)
        else:
            k = self._best_k
            prob_df = pd.DataFrame(
                np.full((len(self._feature_panel), k), 1.0 / k),
                index=self._feature_panel.index,
                columns=list(range(k)),
            )

        self._signal_cache = self._decision_engine.process_to_frame(prob_df)
        self._last_fit_date = self._feature_panel.index[-1]
        self._fitted = True

        # Phase 3b: reinitialize particle filter from new HMM model
        self._reinit_particle_filter()

        # Store raw price series for emergency trigger checks
        self._spy_prices = spy_prices
        self._tlt_prices = None   # populated if caller passes it
        self._vix_prices = vix_prices

        # DIAGNOSTIC: regime occupancy
        if self._signal_cache is not None and 'label' in self._signal_cache.columns:
            occ = self._signal_cache['label'].value_counts().to_dict()
            total = len(self._signal_cache)
            pct = {k: f'{v} ({100*v/total:.1f}%)' for k, v in sorted(occ.items())}
            log.info(
                f'REGIME ENGINE COMPLETE candidates={self._cfg["regime_n_candidates"]} '
                f'best_k={self._best_k} '
                f'backend={self._model.active_backend if self._model else "none"} '
                f'dates={total} occupancy={pct}'
            )

        log.info(
            f"regime.engine: fit complete. {len(self._signal_cache)} signal dates, "
            f"K={self._best_k}, backend={self._model.active_backend if self._model else 'none'}"
        )

    # ── public signal API ─────────────────────────────────────────────────

    def get_signal(self, as_of_date: pd.Timestamp) -> Optional[RegimeSignal]:
        """
        Return the regime signal for as_of_date.
        Uses only data available up to that date (causal).
        """
        if not self._fitted or self._signal_cache is None:
            log.warning("regime.engine: get_signal called before fit — returning None")
            return None
        return get_signal_for_date(self._signal_cache, as_of_date)

    def get_signal_series(self) -> pd.DataFrame:
        """Return the full signal DataFrame (for backtesting / diagnostics)."""
        return self._signal_cache.copy() if self._signal_cache is not None else pd.DataFrame()

    def get_feature_panel(self) -> pd.DataFrame:
        return self._feature_panel.copy() if self._feature_panel is not None else pd.DataFrame()

    # ── particle filter helpers ───────────────────────────────────────────────

    def _reinit_particle_filter(self) -> None:
        """Initialize (or reinitialize) particle filter from current HMM model."""
        if self._model is None or not self._model._fitted:
            self._particle_filter = None
            return
        try:
            from .particle_filter import RegimeParticleFilter
            # Extract the underlying hmmlearn GaussianHMM if available
            hmm_backend = getattr(self._model, "_hmm", None)
            if hmm_backend is not None and hmm_backend._fitted and hmm_backend._model is not None:
                self._particle_filter = RegimeParticleFilter(
                    hmm_model=hmm_backend._model, n_particles=500
                )
                log.info("[Engine] Particle filter initialized (K=%d)", self._best_k)
            else:
                # Markov backend or unfitted HMM — particle filter not supported
                self._particle_filter = None
                log.debug("[Engine] Particle filter skipped (non-HMM backend)")
        except Exception as exc:
            log.warning("[Engine] Particle filter init failed: %s", exc)
            self._particle_filter = None

    def update_particle_filter(
        self, today_observation: np.ndarray
    ) -> Optional["RegimeSignal"]:
        """
        Update regime posterior with today's observation.
        Call once per day after new prices are available.

        Returns updated RegimeSignal, or None if particle filter is unavailable.
        """
        if self._particle_filter is None:
            return None
        state_labels = {
            k: v.value if hasattr(v, "value") else str(v)
            for k, v in (self._model.state_labels.items() if self._model else {}).items()
        }
        return self._particle_filter.update(today_observation, state_labels=state_labels)

    def check_and_run_emergency_refit(
        self,
        spy_prices: pd.Series,
        tlt_prices: Optional[pd.Series] = None,
        vix_prices: Optional[pd.Series] = None,
        universe_prices: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> bool:
        """
        Check emergency refit triggers and run refit if any fire.

        Call once per day after price data is available (before portfolio construction).

        Returns True if emergency refit was executed, False otherwise.
        Logs trigger to logs/regime_emergency_refit.jsonl.
        """
        import json
        import os
        from datetime import datetime

        if not self._fitted:
            return False

        tlt = tlt_prices if tlt_prices is not None else spy_prices * 0.25
        vix = vix_prices if vix_prices is not None else pd.Series(
            [20.0] * len(spy_prices), index=spy_prices.index
        )

        triggered, reason = check_emergency_refit_triggers(
            spy=spy_prices,
            tlt=tlt,
            vix=vix,
            breaks_detector=self._break_detector,
        )

        if not triggered:
            return False

        log.warning("[Engine] Emergency refit triggered: %s", reason)

        try:
            os.makedirs("logs", exist_ok=True)
            with open(EMERGENCY_REFIT_LOG, "a") as f:
                f.write(json.dumps({
                    "date": str(spy_prices.index[-1].date()),
                    "trigger": reason,
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")
        except Exception as exc:
            log.warning("[Engine] Failed to write emergency refit log: %s", exc)

        try:
            self.fit(
                spy_prices=spy_prices,
                universe_prices=universe_prices,
                vix_prices=vix_prices,
                macro_df=macro_df,
                run_model_selection=False,  # fast refit: reuse best_k, skip K selection
            )
            log.info("[Engine] Emergency refit complete: %s", reason)
        except Exception as exc:
            log.error("[Engine] Emergency refit failed: %s", exc)

        return True

    # ── refit trigger ─────────────────────────────────────────────────────

    def should_refit(self, as_of_date: pd.Timestamp) -> bool:
        """
        Returns True if a model refit is recommended based on:
          1. Scheduled quarterly refit interval, OR
          2. Structural break detector recommendation.
        """
        if not self._fitted or self._last_fit_date is None:
            return True

        days_since_fit = (as_of_date - self._last_fit_date).days
        if days_since_fit >= self._cfg["regime_refit_every_days"]:
            log.info(
                f"regime.engine: scheduled refit triggered ({days_since_fit} days since last fit)"
            )
            return True

        if self._feature_panel is not None:
            return self._break_detector.should_refit(
                self._feature_panel,
                as_of_date=as_of_date,
            )

        return False

    # ── break detection ───────────────────────────────────────────────────

    def detect_breaks(
        self,
        target_columns: Optional[List[str]] = None,
    ) -> Dict:
        """Run structural break detection on the feature panel (diagnostic)."""
        if self._feature_panel is None:
            return {}
        return self._break_detector.detect(self._feature_panel, target_columns)

    def break_intensity_series(self) -> pd.Series:
        """Time series of aggregate break intensity (diagnostic)."""
        if self._feature_panel is None:
            return pd.Series(dtype=float)
        return self._break_detector.detect_aggregate(self._feature_panel)

    # ── scorecard access ──────────────────────────────────────────────────

    @property
    def scorecard(self) -> Optional[RegimeScorecard]:
        return self._scorecard

    @property
    def best_k(self) -> int:
        return self._best_k

    @property
    def state_labels(self) -> Dict[int, RegimeLabel]:
        if self._model:
            return self._model.state_labels
        return {}

    # ── serialization ─────────────────────────────────────────────────────

    def save_for_intel(self, dashboard_dir: str = "dashboard") -> None:
        """
        Write regime_labels.csv in the exact schema run_20in20.py expects.
        Call after engine.fit(), regardless of run mode.

        Writes: dashboard/regime_labels.csv with columns [date, label, ...]
        """
        import os
        if self._signal_cache is None or self._signal_cache.empty:
            log.warning("regime.engine: save_for_intel called but signal_cache is empty")
            return
        os.makedirs(dashboard_dir, exist_ok=True)
        out_path = Path(dashboard_dir) / "regime_labels.csv"
        df = self._signal_cache.copy()
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(out_path, index=False)
        print(f"[Regime] Intel export -> {out_path}  ({len(df)} rows, latest={df['label'].iloc[-1]})")

    def save(self, cache_dir: str = "data_cache") -> None:
        path = Path(cache_dir) / "regime_engine.pkl"
        if self._model:
            self._model.save(path)
        if self._signal_cache is not None:
            self._signal_cache.to_parquet(
                Path(cache_dir) / "regime_signals.parquet"
            )
        log.info(f"regime.engine: saved to {cache_dir}")

    @classmethod
    def load(cls, cache_dir: str = "data_cache", config: Optional[Dict] = None) -> "RegimeEngine":
        engine = cls(config=config)
        model_path = Path(cache_dir) / "regime_engine.pkl"
        signal_path = Path(cache_dir) / "regime_signals.parquet"
        if model_path.exists():
            engine._model = RegimeModel.load(model_path)
            engine._best_k = engine._model.k_regimes
        if signal_path.exists():
            engine._signal_cache = pd.read_parquet(signal_path)
        if engine._model or engine._signal_cache is not None:
            engine._fitted = True
        return engine