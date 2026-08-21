# tests/portfolio/test_risk_construction.py
"""
Risk-aware construction: inverse-vol tilt + correlation-cluster cap.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.optimizer import (
    sector_constrained_weighted,
    enforce_cluster_cap,
    _apply_inverse_vol_tilt,
)


def _alpha_panel(symbols, n_dates=3, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    # Descending alpha so ranking is deterministic-ish but nonzero spread
    base = np.linspace(2.0, 0.5, len(symbols))
    data = base + 0.01 * rng.standard_normal((n_dates, len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


class TestInverseVolTilt:
    def test_high_vol_name_downweighted(self):
        scores = pd.Series({"CALM": 1.0, "WILD": 1.0})
        vol_row = pd.Series({"CALM": 0.15, "WILD": 0.60})
        tilted = _apply_inverse_vol_tilt(scores, vol_row)
        assert tilted["CALM"] > tilted["WILD"]

    def test_half_strength_and_clip(self):
        scores = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
        vol_row = pd.Series({"A": 0.20, "B": 0.20, "C": 10.0})  # extreme vol
        tilted = _apply_inverse_vol_tilt(scores, vol_row)
        # Ratio clipped at 0.5 then sqrt → minimum tilt is sqrt(0.5) ≈ 0.707
        assert tilted["C"] == pytest.approx(np.sqrt(0.5))

    def test_none_vol_is_noop(self):
        scores = pd.Series({"A": 1.0, "B": 2.0})
        pd.testing.assert_series_equal(_apply_inverse_vol_tilt(scores, None), scores)

    def test_missing_vol_gets_neutral_tilt(self):
        scores = pd.Series({"A": 1.0, "B": 1.0})
        vol_row = pd.Series({"A": 0.20})  # B missing
        tilted = _apply_inverse_vol_tilt(scores, vol_row)
        assert tilted["B"] == pytest.approx(1.0)

    def test_panel_integration_postconditions(self):
        symbols = [f"S{i}" for i in range(12)]
        alpha = _alpha_panel(symbols)
        vol = pd.DataFrame(0.20, index=alpha.index, columns=symbols)
        vol["S0"] = 0.80  # top-alpha name is also the wildest
        w_tilt = sector_constrained_weighted(
            alpha, n=8, max_weight=0.20, sector_map={}, vol_panel=vol)
        w_flat = sector_constrained_weighted(
            alpha, n=8, max_weight=0.20, sector_map={})
        for dt in alpha.index:
            row = w_tilt.loc[dt]
            assert row.sum() == pytest.approx(1.0, abs=1e-6)
            # NOTE: 1e-3 tolerance matches pre-existing _water_fill_cap precision
            # (clamp-renorm loop exits at ~cap+1e-4 even without the tilt)
            assert row.max() <= 0.20 + 1e-3
            # The wild top name carries less weight than without the tilt
            assert row["S0"] <= w_flat.loc[dt, "S0"] + 1e-12


class _FakeRegimeLabel:
    value = "crisis"


class _FakeRegimeSignal:
    label = _FakeRegimeLabel()


class TestRegimeSignalNoOp:
    """
    regime_signal is accepted by sector_constrained_weighted but does not
    tighten max_weight (see the function's docstring for why: the caller
    that would matter, walk_forward_runner.py, already treats regime_signal
    as fold-local, and wiring in regime_max_weight() would silently change
    the canonical walk-forward numbers rather than just fixing a doc lie).
    This test locks in that current behavior: passing a regime_signal must
    never raise, and must produce identical output to passing None, so a
    future change to this call site is a deliberate decision, not a
    regression.
    """

    def test_regime_signal_does_not_raise_and_is_a_noop(self):
        symbols = [f"S{i}" for i in range(6)]
        alpha = _alpha_panel(symbols)
        w_with_signal = sector_constrained_weighted(
            alpha, n=4, max_weight=0.15, sector_map={},
            regime_signal=_FakeRegimeSignal(),
        )
        w_without_signal = sector_constrained_weighted(
            alpha, n=4, max_weight=0.15, sector_map={}, regime_signal=None,
        )
        pd.testing.assert_frame_equal(w_with_signal, w_without_signal)


class TestClusterCap:
    def _returns(self, n=63, seed=11):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2026-01-02", periods=n)
        common = rng.standard_normal(n) * 0.02
        return pd.DataFrame({
            # EM-style cluster: three names driven by one factor
            "EWY": common + 0.001 * rng.standard_normal(n),
            "EWT": common + 0.001 * rng.standard_normal(n),
            "EEM": common + 0.001 * rng.standard_normal(n),
            # Independent names
            "WMT": rng.standard_normal(n) * 0.01,
            "MRK": rng.standard_normal(n) * 0.01,
            "NEE": rng.standard_normal(n) * 0.01,
        }, index=dates)

    def test_cluster_capped_and_sum_preserved(self):
        w = pd.Series({"EWY": 0.12, "EWT": 0.10, "EEM": 0.08,
                       "WMT": 0.25, "MRK": 0.25, "NEE": 0.20})
        out = enforce_cluster_cap(w, self._returns(), max_cluster_weight=0.20)
        assert out[["EWY", "EWT", "EEM"]].sum() == pytest.approx(0.20, abs=1e-9)
        assert out.sum() == pytest.approx(w.sum(), abs=1e-9)
        # Freed weight went to the independent names
        assert out["WMT"] > w["WMT"]

    def test_uncorrelated_unchanged(self):
        w = pd.Series({"WMT": 0.4, "MRK": 0.3, "NEE": 0.3})
        out = enforce_cluster_cap(w, self._returns(), max_cluster_weight=0.20)
        pd.testing.assert_series_equal(out, w.astype(float))

    def test_cluster_below_cap_unchanged(self):
        w = pd.Series({"EWY": 0.06, "EWT": 0.06, "EEM": 0.05,
                       "WMT": 0.30, "MRK": 0.30, "NEE": 0.23})
        out = enforce_cluster_cap(w, self._returns(), max_cluster_weight=0.20)
        pd.testing.assert_series_equal(out, w.astype(float))

    def test_max_weight_reenforced_after_redistribution(self):
        w = pd.Series({"EWY": 0.20, "EWT": 0.20, "EEM": 0.20,
                       "WMT": 0.14, "MRK": 0.13, "NEE": 0.13})
        out = enforce_cluster_cap(
            w, self._returns(), max_cluster_weight=0.20, max_weight=0.20)
        assert out.max() <= 0.20 + 1e-6
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_single_name_and_empty_passthrough(self):
        rets = self._returns()
        one = pd.Series({"WMT": 1.0})
        pd.testing.assert_series_equal(enforce_cluster_cap(one, rets), one)
        empty = pd.Series(dtype=float)
        assert enforce_cluster_cap(empty, rets).empty

    def test_insufficient_return_history_passthrough(self):
        w = pd.Series({"EWY": 0.5, "EWT": 0.5})
        short_rets = self._returns(n=5)
        pd.testing.assert_series_equal(enforce_cluster_cap(w, short_rets), w)
