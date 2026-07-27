# tests/portfolio/test_risk_budget_cap.py
"""
W1 (Sizing) — per-name risk-budget cap.

No name may contribute more than `budget` annualized of w_i * sigma_i.
Excess is redistributed pro-rata across uncapped names, iterated to a fixed
point, preserving sum-to-1. Mirrors _water_fill_cap / enforce_cluster_cap.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.optimizer import enforce_risk_budget_cap


class TestRiskBudgetCap:
    def test_high_vol_name_capped_and_excess_redistributed(self):
        # BRBR-style wild name at 10% alongside calmer names.
        w = pd.Series({"BRBR": 0.10, "ALGM": 0.10, "SGOV": 0.10, "DLR": 0.10})
        vols = pd.Series({"BRBR": 0.758, "ALGM": 0.40, "SGOV": 0.02, "DLR": 0.20})
        out = enforce_risk_budget_cap(w, vols, budget=0.012)

        # BRBR risk contribution capped at budget
        assert out["BRBR"] * vols["BRBR"] <= 0.012 + 1e-6
        assert out["BRBR"] < w["BRBR"]
        # Excess flowed to the uncapped, low-vol names
        assert out["SGOV"] > w["SGOV"]
        assert out.sum() == pytest.approx(w.sum(), abs=1e-9)

    def test_no_violation_is_noop(self):
        w = pd.Series({"A": 0.10, "B": 0.10})
        vols = pd.Series({"A": 0.05, "B": 0.05})  # 0.10*0.05=0.005 < 0.012
        out = enforce_risk_budget_cap(w, vols, budget=0.012)
        pd.testing.assert_series_equal(out, w.astype(float))

    def test_sum_to_one_preserved(self):
        w = pd.Series({"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.20})
        vols = pd.Series({"A": 0.80, "B": 0.70, "C": 0.10, "D": 0.10})
        out = enforce_risk_budget_cap(w, vols, budget=0.012)
        assert out.sum() == pytest.approx(1.0, abs=1e-9)

    def test_missing_vol_data_skips_name(self, caplog):
        w = pd.Series({"A": 0.10, "B": 0.10})
        vols = pd.Series({"A": 0.80})  # B missing entirely
        with caplog.at_level("WARNING"):
            out = enforce_risk_budget_cap(w, vols, budget=0.012)
        # A gets capped, B (no vol data) is not touched by the cap logic —
        # it absorbs redistributed excess instead, and total is preserved.
        assert out["A"] * vols["A"] <= 0.012 + 1e-6
        assert out.sum() == pytest.approx(w.sum(), abs=1e-9)
        assert any("vol" in rec.message.lower() for rec in caplog.records)

    def test_all_names_capped_renormalizes(self):
        # Budget so tight relative to vols that every name hits its cap and
        # caps alone can't sum to 1 — must renormalize rather than silently
        # under-allocate (analogous to _water_fill_cap's infeasible branch).
        w = pd.Series({"A": 0.5, "B": 0.5})
        vols = pd.Series({"A": 1.0, "B": 1.0})  # cap_i = 0.012/1.0 = 0.012 each
        out = enforce_risk_budget_cap(w, vols, budget=0.012)
        assert out.sum() == pytest.approx(w.sum(), abs=1e-9)
        # Both should get equal share of the total (equal caps -> equal renorm)
        assert out["A"] == pytest.approx(out["B"], abs=1e-9)

    def test_empty_input_passthrough(self):
        w = pd.Series(dtype=float)
        vols = pd.Series(dtype=float)
        out = enforce_risk_budget_cap(w, vols, budget=0.012)
        assert out.empty

    def test_zero_or_negative_vol_treated_as_missing(self):
        w = pd.Series({"A": 0.10, "B": 0.10})
        vols = pd.Series({"A": 0.0, "B": 0.20})
        out = enforce_risk_budget_cap(w, vols, budget=0.012)
        # B is capped (0.10*0.20=0.02 > 0.012); A has invalid vol so isn't capped.
        assert out["B"] * vols["B"] <= 0.012 + 1e-6
        assert out.sum() == pytest.approx(w.sum(), abs=1e-9)

    def test_iterates_to_fixed_point_multiple_violators(self):
        w = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        vols = pd.Series({"A": 0.90, "B": 0.85, "C": 0.02, "D": 0.02})
        out = enforce_risk_budget_cap(w, vols, budget=0.012)
        assert (out * vols <= 0.012 + 1e-6).all()
        assert out.sum() == pytest.approx(1.0, abs=1e-9)
