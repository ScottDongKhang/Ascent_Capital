# tests/test_phase3_hardening.py
"""
Phase 3 hardening — Fix 3a: CPCV for ML sleeve, Fix 3b: Regime particle filter
+ emergency refit.

RED phase: all tests should fail before implementation exists.
"""
from __future__ import annotations

import math
from itertools import combinations
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Phase 3a — CPCV Splitter
# ============================================================

class TestCPCVSplitter:
    """Tests for ascent/research/cpcv.py — CPCVSplitter."""

    def _make_dates(self, n: int = 600) -> pd.DatetimeIndex:
        return pd.bdate_range("2020-01-02", periods=n)

    def test_produces_c_n_k_splits(self):
        """C(6,2) = 15 splits for default n_splits=6, n_test_splits=2."""
        from ascent.research.cpcv import CPCVSplitter
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)
        dates = self._make_dates(600)
        splits = list(splitter.split(dates))
        assert len(splits) == math.comb(6, 2), f"Expected 15 splits, got {len(splits)}"

    def test_train_test_no_overlap(self):
        """Train and test date sets must not overlap."""
        from ascent.research.cpcv import CPCVSplitter
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)
        dates = self._make_dates(600)
        for train_dates, test_dates in splitter.split(dates):
            overlap = set(train_dates) & set(test_dates)
            assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} dates"

    def test_purge_gap_enforced(self):
        """No train date within purge_days business days before the start of each test fold segment."""
        from ascent.research.cpcv import CPCVSplitter
        purge = 5
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=purge, embargo_days=0)
        dates = self._make_dates(600)
        all_dates_sorted = sorted(dates)
        for train_dates, test_dates in splitter.split(dates):
            if len(test_dates) == 0 or len(train_dates) == 0:
                continue
            # Find start of each contiguous test fold segment
            test_dt_sorted = sorted(test_dates)
            fold_starts = [test_dt_sorted[0]]
            for i in range(1, len(test_dt_sorted)):
                gap_days = (test_dt_sorted[i] - test_dt_sorted[i - 1]).days
                if gap_days > 5:  # non-consecutive → new test fold segment
                    fold_starts.append(test_dt_sorted[i])
            train_set = set(train_dates)
            for fold_start in fold_starts:
                # The `purge` positions immediately before fold_start in the full date index
                fold_start_idx = all_dates_sorted.index(fold_start)
                purge_zone = set(
                    all_dates_sorted[i]
                    for i in range(max(0, fold_start_idx - purge), fold_start_idx)
                )
                overlap = purge_zone & train_set
                assert len(overlap) == 0, (
                    f"Train contains {len(overlap)} dates in purge zone before {fold_start}: {sorted(overlap)[:3]}"
                )

    def test_each_date_in_correct_number_of_test_folds(self):
        """Each date appears in exactly C(N-1, k-1) = C(5,1) = 5 test folds."""
        from ascent.research.cpcv import CPCVSplitter
        n, k = 6, 2
        expected_appearances = math.comb(n - 1, k - 1)  # C(5,1) = 5
        splitter = CPCVSplitter(n_splits=n, n_test_splits=k, purge_days=0, embargo_days=0)
        dates = self._make_dates(600)
        date_test_count: dict = {}
        for _, test_dates in splitter.split(dates):
            for d in test_dates:
                date_test_count[d] = date_test_count.get(d, 0) + 1
        # Check interior dates (edge folds may be skipped due to data limits)
        interior = sorted(date_test_count.keys())[30:-30]
        for d in interior:
            assert date_test_count[d] == expected_appearances, (
                f"Date {d} appeared in {date_test_count[d]} test folds, "
                f"expected {expected_appearances}"
            )

    def test_test_dates_cover_full_range(self):
        """The union of all test sets covers (approximately) the full date range."""
        from ascent.research.cpcv import CPCVSplitter
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=0, embargo_days=0)
        dates = self._make_dates(600)
        all_test_dates: set = set()
        for _, test_dates in splitter.split(dates):
            all_test_dates.update(test_dates)
        # Every date should appear in some test fold
        assert len(all_test_dates) >= int(0.95 * len(dates)), (
            f"Only {len(all_test_dates)} of {len(dates)} dates covered by test folds"
        )

    def test_n_test_splits_parametric(self):
        """Works with different n_splits/n_test_splits combinations."""
        from ascent.research.cpcv import CPCVSplitter
        splitter = CPCVSplitter(n_splits=4, n_test_splits=1, purge_days=0, embargo_days=0)
        dates = self._make_dates(400)
        splits = list(splitter.split(dates))
        assert len(splits) == math.comb(4, 1)  # C(4,1) = 4


# ============================================================
# Phase 3a — build_ml_alpha_cpcv guards
# ============================================================

class TestBuildMlAlphaCPCV:
    """Tests for build_ml_alpha_cpcv() in ascent/alpha/ml_sleeve.py."""

    def _make_features(self, n_dates=300, n_symbols=20) -> dict:
        dates = pd.bdate_range("2020-01-02", periods=n_dates)
        symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
        rng = np.random.default_rng(42)
        feat_dict = {}
        for feat in ["mom_skip1m", "zscore_20d", "high_52w_pct", "mom_126d", "vol_63d"]:
            feat_dict[feat] = pd.DataFrame(
                rng.standard_normal((n_dates, n_symbols)),
                index=dates, columns=symbols,
            )
        feat_dict["targets"] = pd.DataFrame(
            rng.standard_normal((n_dates, n_symbols)),
            index=dates, columns=symbols,
        )
        return feat_dict

    def test_returns_empty_df_when_too_few_folds_converge(self):
        """Returns empty DataFrame when fewer than 10/15 splits converge."""
        from ascent.alpha.ml_sleeve import build_ml_alpha_cpcv
        from ascent.research.cpcv import CPCVSplitter
        features = self._make_features()
        targets = features.pop("targets")
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)
        # Mock _train_xgboost to raise for most folds
        call_count = {"n": 0}
        def _failing_train(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 12:
                raise RuntimeError("simulated fold failure")
            return MagicMock()
        with patch("ascent.alpha.ml_sleeve._train_xgboost", side_effect=_failing_train):
            result = build_ml_alpha_cpcv(features, targets, agent_id="test", splitter=splitter)
        assert result.empty, "Expected empty DF when < 10 folds converge"

    def test_returns_empty_df_when_p5_sharpe_negative(self):
        """Returns empty DataFrame when 5th-percentile fold Sharpe < 0."""
        from ascent.alpha.ml_sleeve import build_ml_alpha_cpcv
        from ascent.research.cpcv import CPCVSplitter
        features = self._make_features()
        targets = features.pop("targets")
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)
        mock_model = MagicMock()
        # Force all fold ICs to be strongly negative → p5 < 0
        mock_model.predict.return_value = np.ones(20) * -10.0
        with patch("ascent.alpha.ml_sleeve._train_xgboost", return_value=mock_model):
            with patch("ascent.alpha.ml_sleeve._compute_fold_ic", return_value=-0.5):
                result = build_ml_alpha_cpcv(features, targets, agent_id="test", splitter=splitter)
        assert result.empty, "Expected empty DF when p5 Sharpe < 0"

    def test_logs_sharpe_p5_and_p50(self, caplog):
        """Logs p5 and p50 Sharpe from CPCV folds."""
        import logging
        from ascent.alpha.ml_sleeve import build_ml_alpha_cpcv
        from ascent.research.cpcv import CPCVSplitter
        features = self._make_features()
        targets = features.pop("targets")
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.standard_normal(20)
        with caplog.at_level(logging.INFO, logger="ascent.alpha.ml_sleeve"):
            with patch("ascent.alpha.ml_sleeve._train_xgboost", return_value=mock_model):
                with patch("ascent.alpha.ml_sleeve._compute_fold_ic", return_value=0.2):
                    build_ml_alpha_cpcv(features, targets, agent_id="test", splitter=splitter)
        assert any("p5" in r.message and "p50" in r.message for r in caplog.records), (
            "Expected log message with p5 and p50 Sharpe"
        )


# ============================================================
# Phase 3b — Emergency refit triggers
# ============================================================

class TestEmergencyRefitTriggers:
    """Tests for check_emergency_refit_triggers() in ascent/regime/engine.py."""

    def _make_spy(self, n=300, base=400.0) -> pd.Series:
        dates = pd.bdate_range("2023-01-02", periods=n)
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0003, 0.01, n)
        prices = base * np.cumprod(1 + returns)
        return pd.Series(prices, index=dates, name="SPY")

    def _make_tlt(self, n=300, base=95.0) -> pd.Series:
        dates = pd.bdate_range("2023-01-02", periods=n)
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0001, 0.008, n)
        prices = base * np.cumprod(1 + returns)
        return pd.Series(prices, index=dates, name="TLT")

    def _make_vix(self, n=300, level=18.0) -> pd.Series:
        dates = pd.bdate_range("2023-01-02", periods=n)
        return pd.Series([level] * n, index=dates, name="VIX")

    def test_no_trigger_in_calm_market(self):
        """Returns False when no triggers are active."""
        from ascent.regime.engine import check_emergency_refit_triggers
        spy = self._make_spy()
        tlt = self._make_tlt()
        vix = self._make_vix(level=15.0)
        mock_detector = MagicMock()
        mock_detector.latest_zscore.return_value = 1.0
        triggered, reason = check_emergency_refit_triggers(spy, tlt, vix, mock_detector)
        assert not triggered
        assert reason == ""

    def test_trigger_on_spy_crash_plus_high_vix(self):
        """Trigger 1: SPY 1-day return < -3% AND VIX > 30."""
        from ascent.regime.engine import check_emergency_refit_triggers
        spy = self._make_spy()
        # Force last day to be a -4% crash
        spy.iloc[-1] = spy.iloc[-2] * 0.96
        tlt = self._make_tlt()
        vix = self._make_vix(level=35.0)
        mock_detector = MagicMock()
        mock_detector.latest_zscore.return_value = 1.0
        triggered, reason = check_emergency_refit_triggers(spy, tlt, vix, mock_detector)
        assert triggered, "Expected trigger: SPY -4% + VIX 35"
        assert "SPY" in reason

    def test_trigger_on_200ma_cross_below(self):
        """Trigger 2: SPY crosses below its 200-day MA."""
        from ascent.regime.engine import check_emergency_refit_triggers
        # Build a series that was above MA then falls below
        n = 250
        dates = pd.bdate_range("2022-01-03", periods=n)
        # First 248 days above a declining MA, last 2 days cross below
        prices = np.linspace(500, 450, n)      # declining prices
        ma200 = np.full(n, 460.0)              # MA stays at 460
        # Day -2: above MA (price > 460), day -1: below MA (price < 460)
        prices[-2] = 465.0   # above
        prices[-1] = 455.0   # below
        spy = pd.Series(prices, index=dates)
        tlt = self._make_tlt(n=n)
        vix = self._make_vix(n=n, level=20.0)
        mock_detector = MagicMock()
        mock_detector.latest_zscore.return_value = 1.0
        # Patch rolling().mean() to return the fixed MA
        with patch.object(spy, "rolling", return_value=MagicMock(
            return_value=MagicMock(
                mean=lambda: pd.Series(ma200, index=dates)
            )
        )):
            # Build a spy with explicit MA cross
            spy2 = pd.Series(prices, index=dates)
            # We need the function to use real rolling so let's just build the data properly
            pass
        # Direct approach: just use a long series where spy was above MA, now below
        long_n = 210
        long_dates = pd.bdate_range("2022-01-03", periods=long_n)
        # Rising then crashing prices — build so last point crosses below rolling 200
        p = np.ones(long_n) * 500.0
        # Force a sharp drop at the end
        p[-1] = 300.0   # Crashes below any reasonable MA of ~500
        p[-2] = 505.0   # Was above
        spy3 = pd.Series(p, index=long_dates)
        tlt3 = self._make_tlt(n=long_n)
        vix3 = self._make_vix(n=long_n, level=22.0)
        triggered, reason = check_emergency_refit_triggers(spy3, tlt3, vix3, mock_detector)
        assert triggered, "Expected trigger: SPY crossed below 200MA"
        assert "200" in reason or "MA" in reason

    def test_trigger_on_spy_tlt_correlation_flip(self):
        """Trigger 3: SPY/TLT 5-day correlation > +0.30 (both selling = liquidity crisis)."""
        from ascent.regime.engine import check_emergency_refit_triggers
        n = 210   # need ≥201 for 200MA trigger to be inactive
        dates = pd.bdate_range("2018-01-02", periods=n)
        rng = np.random.default_rng(0)
        # Build SPY with a clear uptrend so 200MA trigger doesn't fire
        spy_prices = np.cumprod(1 + rng.normal(0.0008, 0.005, n)) * 400
        tlt_prices = np.cumprod(1 + rng.normal(0.0002, 0.003, n)) * 95
        spy = pd.Series(spy_prices, index=dates)
        tlt = pd.Series(tlt_prices, index=dates)
        # Liquidity crisis: last 5 returns are driven by a common negative shock
        # Both SPY and TLT sell off together (positive correlation)
        common_shock = np.array([-0.020, -0.015, -0.018, -0.022, -0.016])
        spy_rets = common_shock + rng.normal(0, 0.001, 5)   # SPY: ~2% daily losses
        tlt_rets = common_shock * 0.6 + rng.normal(0, 0.001, 5)  # TLT follows same shock
        # Rebuild last 5 prices from returns
        for j, i in enumerate(range(n - 5, n)):
            spy.iloc[i] = spy.iloc[i - 1] * (1 + spy_rets[j])
            tlt.iloc[i] = tlt.iloc[i - 1] * (1 + tlt_rets[j])
        vix = self._make_vix(n=n, level=22.0)
        mock_detector = MagicMock()
        mock_detector.latest_zscore.return_value = 1.0
        triggered, reason = check_emergency_refit_triggers(spy, tlt, vix, mock_detector)
        assert triggered, (
            f"Expected trigger: SPY/TLT both declining with shared shock. "
            f"triggered={triggered}, reason={reason!r}"
        )
        assert "corr" in reason.lower() or "correlation" in reason.lower()

    def test_trigger_on_structural_break_zscore(self):
        """Trigger 4: structural break z-score from BreakDetector > 3.5."""
        from ascent.regime.engine import check_emergency_refit_triggers
        spy = self._make_spy()
        tlt = self._make_tlt()
        vix = self._make_vix(level=18.0)
        mock_detector = MagicMock()
        mock_detector.latest_zscore.return_value = 4.0  # > 3.5 threshold
        triggered, reason = check_emergency_refit_triggers(spy, tlt, vix, mock_detector)
        assert triggered, "Expected trigger: structural break z-score > 3.5"
        assert "break" in reason.lower() or "z-score" in reason.lower() or "zscore" in reason.lower()


# ============================================================
# Phase 3b — BreakDetector.latest_zscore()
# ============================================================

class TestBreakDetectorLatestZscore:
    """Tests for BreakDetector.latest_zscore() added to ascent/regime/breaks.py."""

    def _make_panel(self, n=200) -> pd.DataFrame:
        dates = pd.bdate_range("2020-01-02", periods=n)
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            rng.standard_normal((n, 4)),
            index=dates,
            columns=["spy_ret", "vix_level", "yield_spread", "vol_ratio"],
        )

    def test_latest_zscore_returns_float(self):
        """latest_zscore() returns a float >= 0."""
        from ascent.regime.breaks import BreakDetector
        detector = BreakDetector()
        panel = self._make_panel()
        detector.detect(panel)  # prime the detector
        z = detector.latest_zscore(panel)
        assert isinstance(z, float), f"Expected float, got {type(z)}"
        assert z >= 0.0, f"Z-score should be non-negative, got {z}"

    def test_latest_zscore_higher_after_structural_break(self):
        """Z-score is elevated when recent observations diverge from historical."""
        from ascent.regime.breaks import BreakDetector
        n = 300
        dates = pd.bdate_range("2020-01-02", periods=n)
        rng = np.random.default_rng(99)
        # Normal regime for first 280 days, then sudden shift
        values = np.concatenate([
            rng.standard_normal(280),
            rng.standard_normal(20) + 5.0,  # mean shift of 5σ
        ])
        panel = pd.DataFrame({"spy_ret": values}, index=dates)
        detector = BreakDetector()
        z = detector.latest_zscore(panel)
        assert z > 0.0, "Expected non-zero z-score after structural break"

    def test_latest_zscore_without_ruptures_returns_zero(self):
        """Returns 0.0 gracefully when ruptures is not installed."""
        from ascent.regime.breaks import BreakDetector
        import ascent.regime.breaks as breaks_module
        detector = BreakDetector()
        panel = self._make_panel()
        with patch.object(breaks_module, "_HAS_RUPTURES", False):
            z = detector.latest_zscore(panel)
        assert z == 0.0


# ============================================================
# Phase 3b — RegimeParticleFilter
# ============================================================

class TestRegimeParticleFilter:
    """Tests for ascent/regime/particle_filter.py — RegimeParticleFilter."""

    def _make_mock_hmm(self, k: int = 3, n_features: int = 3):
        """Create a minimal mock GaussianHMM-like object."""
        rng = np.random.default_rng(0)
        mock = MagicMock()
        mock.n_components = k
        # Uniform start probs
        mock.startprob_ = np.ones(k) / k
        # Slightly sticky transition matrix
        transmat = np.full((k, k), 0.05 / (k - 1))
        np.fill_diagonal(transmat, 0.95)
        mock.transmat_ = transmat
        # Gaussian emissions
        mock.means_ = rng.standard_normal((k, n_features))
        mock.covars_ = np.array([np.eye(n_features) * 0.5 for _ in range(k)])
        mock.covariance_type = "full"
        return mock

    def test_init_creates_n_particles(self):
        """Particle filter initializes with exactly n_particles particles."""
        from ascent.regime.particle_filter import RegimeParticleFilter
        mock_hmm = self._make_mock_hmm(k=3)
        pf = RegimeParticleFilter(hmm_model=mock_hmm, n_particles=500)
        assert len(pf.particles) == 500
        assert len(pf.weights) == 500
        assert abs(pf.weights.sum() - 1.0) < 1e-9

    def test_update_returns_regime_signal(self):
        """update() returns a RegimeSignal with valid label and risk_multiplier."""
        from ascent.regime.particle_filter import RegimeParticleFilter
        from ascent.regime.types import RegimeSignal
        mock_hmm = self._make_mock_hmm(k=3)
        pf = RegimeParticleFilter(hmm_model=mock_hmm, n_particles=200)
        obs = np.zeros(3)  # 3-feature observation vector
        state_labels = {0: "calm_bull", 1: "stressed", 2: "crisis"}
        signal = pf.update(obs, state_labels=state_labels)
        assert isinstance(signal, RegimeSignal)
        assert signal.label in ("calm_bull", "stressed", "crisis", "uncertain")
        assert 0.0 <= signal.risk_multiplier <= 2.0

    def test_weights_sum_to_one_after_update(self):
        """Weights remain normalized to 1.0 after each update step."""
        from ascent.regime.particle_filter import RegimeParticleFilter
        mock_hmm = self._make_mock_hmm(k=3)
        pf = RegimeParticleFilter(hmm_model=mock_hmm, n_particles=300)
        state_labels = {0: "calm_bull", 1: "stressed", 2: "crisis"}
        rng = np.random.default_rng(5)
        for _ in range(10):
            obs = rng.standard_normal(3)
            pf.update(obs, state_labels=state_labels)
            assert abs(pf.weights.sum() - 1.0) < 1e-6, (
                f"Weights sum to {pf.weights.sum():.6f} after update"
            )

    def test_reinitialize_resets_particles(self):
        """reinitialize() fully resets particles to new model's stationary distribution."""
        from ascent.regime.particle_filter import RegimeParticleFilter
        mock_hmm = self._make_mock_hmm(k=3)
        pf = RegimeParticleFilter(hmm_model=mock_hmm, n_particles=300)
        state_labels = {0: "calm_bull", 1: "stressed", 2: "crisis"}
        # Run some updates to move particles from init state
        rng = np.random.default_rng(3)
        for _ in range(5):
            pf.update(rng.standard_normal(3), state_labels=state_labels)
        weights_before = pf.weights.copy()
        # Reinitialize with a new HMM (K=4)
        new_hmm = self._make_mock_hmm(k=4, n_features=3)
        pf.reinitialize(new_hmm)
        assert len(pf.particles) == 300, "Particle count unchanged after reinit"
        assert len(pf.weights) == 300
        assert abs(pf.weights.sum() - 1.0) < 1e-9
        # After reinit, particles should cover the new model's states (0..3)
        assert set(pf.particles).issubset({0, 1, 2, 3}), (
            f"Particles have unexpected states: {set(pf.particles)}"
        )

    def test_resampling_fires_when_effective_n_low(self):
        """SIR resampling triggers when effective N < n_particles/2."""
        from ascent.regime.particle_filter import RegimeParticleFilter
        mock_hmm = self._make_mock_hmm(k=3)
        pf = RegimeParticleFilter(hmm_model=mock_hmm, n_particles=200)
        # Manually degenerate weights (one particle has all weight)
        pf.weights = np.zeros(200)
        pf.weights[0] = 1.0
        eff_n = pf._effective_n()
        assert eff_n < 100, f"Effective N should be very low (1), got {eff_n}"
        state_labels = {0: "calm_bull", 1: "stressed", 2: "crisis"}
        # After update, resampling should fire and restore diversity
        pf.update(np.zeros(3), state_labels=state_labels)
        eff_n_after = pf._effective_n()
        assert eff_n_after > 50, (
            f"Effective N should improve after resampling, got {eff_n_after}"
        )


# ============================================================
# Phase 3b — refit_every_days default changed to 5
# ============================================================

class TestRegimeConfigDefaults:
    """Verify regime config default for refit interval."""

    def test_refit_every_days_is_5(self):
        """regime_refit_every_days should be 5 (was 63 in Phase 2)."""
        from ascent.regime.types import REGIME_CONFIG_DEFAULTS
        assert REGIME_CONFIG_DEFAULTS["regime_refit_every_days"] == 5, (
            f"Expected 5, got {REGIME_CONFIG_DEFAULTS['regime_refit_every_days']}"
        )
