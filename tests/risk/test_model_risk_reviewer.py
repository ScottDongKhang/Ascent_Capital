"""Unit tests for ascent.risk.irm.model_risk_reviewer.

Standalone module, not yet wired into the live pipeline. These tests
exercise `check()` directly with synthetic inputs.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ascent.risk.irm.model_risk_reviewer import (
    CACHE_STALE_FAIL_DAYS,
    NAN_RATE_EVAL_WINDOW,
    NAN_RATE_FAIL_THRESHOLD,
    ModelRiskVerdict,
    check,
)

SPARSE_EXEMPT = {
    "earnings_surprise",
    "analyst_revision",
    "iv_skew",
    "insider_net_score",
    "short_pct_float",
}


def _panel(nan_frac: float, shape=(20, 10)) -> pd.DataFrame:
    """Build a DataFrame with approximately `nan_frac` of its cells NaN."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape)
    df = pd.DataFrame(data)
    n_cells = shape[0] * shape[1]
    n_nan = int(round(nan_frac * n_cells))
    flat_idx = rng.choice(n_cells, size=n_nan, replace=False)
    values = df.values.copy()
    values.ravel()[flat_idx] = np.nan
    return pd.DataFrame(values)


class TestCleanPass:
    def test_fresh_cache_low_nan_panels_pass(self):
        panels = {
            "momentum": _panel(0.0),
            "value": _panel(0.01),
        }
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels=panels,
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert isinstance(verdict, ModelRiskVerdict)
        assert verdict.passed is True
        assert verdict.reason == "all checks passed"
        assert verdict.checks["cache_freshness"]["ok"] is True
        assert verdict.checks["nan_rate::momentum"]["ok"] is True
        assert verdict.checks["nan_rate::value"]["ok"] is True

    def test_none_reason_treated_as_fresh(self):
        # validate_cache returns "" on success; a caller passing None for an
        # unpopulated reason should behave the same way, not be treated as a
        # failure string.
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason=None,
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is True


class TestCacheStaleFail:
    def test_nonempty_reason_fails_even_without_word_stale(self):
        # validate_cache's real failure strings never contain the literal
        # word "stale" -- e.g. "cache latest date 2026-08-10 is 9 calendar
        # days old (limit=1)". This must still be detected as a failure.
        reason = "cache latest date 2026-08-10 is 9 calendar days old (limit=1)"
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason=reason,
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is False
        assert "cache_freshness" in verdict.reason
        assert reason in verdict.reason

    def test_fallback_cache_name_fails_even_with_empty_reason(self):
        # A simulated-fallback cache is degraded data regardless of whether
        # validate_cache itself reported a problem.
        verdict = check(
            price_cache_name="prices_live_fallback_simulated",
            price_cache_reason="",
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is False
        assert verdict.checks["cache_freshness"]["ok"] is False

    def test_direct_date_staleness_fail(self):
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
            cache_latest_date=date(2026, 8, 10),
        )
        assert verdict.passed is False
        detail = verdict.checks["cache_freshness::direct_date"]
        assert detail["ok"] is False
        assert f"limit={CACHE_STALE_FAIL_DAYS}" in detail["detail"]

    def test_direct_date_within_limit_passes(self):
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
            cache_latest_date=date(2026, 8, 18),
        )
        assert verdict.checks["cache_freshness::direct_date"]["ok"] is True


class TestNanRateSparseExemption:
    def test_high_nan_in_sparse_exempt_column_still_passes(self):
        # earnings_surprise is deliberately sparse -- it must be exempted
        # from the NaN-rate check entirely, not merely given a lenient
        # threshold.
        panels = {
            "earnings_surprise": _panel(0.9),  # far above NAN_RATE_FAIL_THRESHOLD
            "momentum": _panel(0.0),
        }
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels=panels,
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is True
        # The exempt panel must not even appear as a check -- it was skipped.
        assert "nan_rate::earnings_surprise" not in verdict.checks
        assert verdict.checks["nan_rate::momentum"]["ok"] is True

    def test_high_nan_in_non_exempt_column_fails(self):
        panels = {
            "momentum": _panel(0.9),  # far above NAN_RATE_FAIL_THRESHOLD, NOT exempt
        }
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels=panels,
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is False
        assert verdict.checks["nan_rate::momentum"]["ok"] is False
        assert "nan_rate::momentum" in verdict.reason

    def test_threshold_boundary(self):
        # Exactly at the threshold should pass (<=), just above should fail.
        # Built deterministically (not via the random `_panel` helper) so the
        # NaN rate is exactly NAN_RATE_FAIL_THRESHOLD, not an approximation.
        shape = (100, 100)
        n_cells = shape[0] * shape[1]
        n_nan = int(round(NAN_RATE_FAIL_THRESHOLD * n_cells))
        flat = np.zeros(n_cells)
        flat[:n_nan] = np.nan
        at_threshold = pd.DataFrame(flat.reshape(shape))
        assert float(at_threshold.isna().mean().mean()) == pytest.approx(NAN_RATE_FAIL_THRESHOLD)
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"panel": at_threshold},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.checks["nan_rate::panel"]["ok"] is True

        above_threshold = _panel(NAN_RATE_FAIL_THRESHOLD + 0.10, shape=(100, 100))
        verdict2 = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"panel": above_threshold},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict2.checks["nan_rate::panel"]["ok"] is False


class TestNanRateWarmupWindow:
    """Regression tests for the whole-branch review finding: measuring NaN
    rate over the entire panel history (instead of the trailing evaluation
    window that actually feeds the alpha stage) structurally fails on every
    real feature panel because rolling-window warm-up (e.g. a 252d momentum
    feature is NaN for its first ~252 rows by construction) dominates the
    average. Empirically measured against prices_live: 7 of 27 non-exempt
    panels exceeded the 5% threshold purely from warm-up."""

    def _warmup_panel(self, n_rows: int, warmup_rows: int, shape_cols: int = 10) -> pd.DataFrame:
        """A panel shaped like a real rolling-window feature: NaN for the
        first `warmup_rows` rows (warm-up), then fully populated (clean) for
        the remainder -- mirroring e.g. mom_252d on a long price history."""
        data = np.zeros((n_rows, shape_cols))
        data[:warmup_rows, :] = np.nan
        return pd.DataFrame(data)

    def test_long_warmup_panel_with_clean_trailing_window_passes(self):
        # 300 rows of history, first 252 NaN (mom_252d-style warm-up), last
        # 48 rows fully populated. Full-panel NaN rate is 252/300 = 84% --
        # would fail under the old whole-history measurement. The trailing
        # NAN_RATE_EVAL_WINDOW (21) rows are entirely within the clean tail,
        # so the check must pass.
        panel = self._warmup_panel(n_rows=300, warmup_rows=252)
        full_history_nan_rate = float(panel.isna().mean().mean())
        assert full_history_nan_rate > NAN_RATE_FAIL_THRESHOLD, (
            "test setup sanity check: full-history NaN rate should exceed "
            "the threshold to prove this scenario would have failed under "
            "the old (whole-panel) measurement"
        )
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"mom_252d": panel},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is True
        assert verdict.checks["nan_rate::mom_252d"]["ok"] is True
        assert f"trailing {NAN_RATE_EVAL_WINDOW}d" in verdict.checks["nan_rate::mom_252d"]["detail"]

    def test_nan_within_trailing_window_still_fails(self):
        # Warm-up extends INTO the trailing evaluation window (e.g. a very
        # short history, or a much longer-lookback feature) -- this must
        # still fail, proving the fix narrows the measurement window rather
        # than disabling the check.
        panel = self._warmup_panel(n_rows=30, warmup_rows=25)
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"mom_252d": panel},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is False
        assert verdict.checks["nan_rate::mom_252d"]["ok"] is False

    def test_panel_shorter_than_window_uses_whole_panel(self):
        # A panel with fewer rows than NAN_RATE_EVAL_WINDOW: tail() degrades
        # gracefully to the whole panel, not an error.
        panel = self._warmup_panel(n_rows=10, warmup_rows=10)
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"mom_252d": panel},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.checks["nan_rate::mom_252d"]["ok"] is False


class TestRegimeStalenessInformational:
    def test_stale_regime_labels_do_not_fail_verdict(self):
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"momentum": _panel(0.0)},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
            regime_label_date=date(2026, 8, 1),
            regime_assessment_date=date(2026, 8, 15),
        )
        assert verdict.passed is True
        assert "regime_label_staleness" in verdict.checks
        assert "WARNING" in verdict.checks["regime_label_staleness"]["detail"]

    def test_fresh_regime_labels_no_warning(self):
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
            regime_label_date=date(2026, 8, 18),
            regime_assessment_date=date(2026, 8, 19),
        )
        assert verdict.passed is True
        assert "WARNING" not in verdict.checks["regime_label_staleness"]["detail"]

    def test_omitted_regime_dates_produce_no_check(self):
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert "regime_label_staleness" not in verdict.checks


class TestReasonStringIsHumanReadable:
    def test_multiple_failures_all_cited_in_reason(self):
        panels = {
            "momentum": _panel(0.9),   # fails
            "value": _panel(0.0),      # passes
        }
        verdict = check(
            price_cache_name="prices_simulated",
            price_cache_reason="cache file missing: /data_cache/prices_live.parquet",
            feature_panels=panels,
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.passed is False
        # Both distinct failing checks must be individually named in the reason.
        assert "cache_freshness:" in verdict.reason
        assert "nan_rate::momentum:" in verdict.reason
        # The passing check must not be cited.
        assert "nan_rate::value" not in verdict.reason

    def test_all_pass_reason_is_exact_sentinel(self):
        verdict = check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"momentum": _panel(0.0)},
            sparse_exempt=SPARSE_EXEMPT,
            as_of_date=date(2026, 8, 19),
        )
        assert verdict.reason == "all checks passed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
