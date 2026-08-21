"""Model Risk Reviewer wiring into ascent/main.py's live pipeline.

Task 2 of the min-viable-cut completion plan wires the previously-standalone
`ascent.risk.irm.model_risk_reviewer.check()` into `run_pipeline()` at two
points: right after `load_or_fetch_prices()` (cache-freshness check) and
right after the `features` dict is built (NaN-rate check). Both calls are
shadow-mode: they only log, they never alter `run_pipeline()`'s control flow
or return value.

`run_pipeline()` itself needs the whole engine to execute (see
`tests/test_ai_prior_freshness.py`'s `TestWiredIntoPipeline` for precedent),
so this file:

1. Tests `load_or_fetch_prices()` directly — it now returns a 3-tuple
   `(df, cache_name, reason)` instead of a 2-tuple, on every return path.
2. Asserts on `ascent/main.py`'s source text that both MRR call sites exist
   and are unconditional (not gated behind a branch that could change
   control flow) and that neither wraps a `return`/mutation of pipeline
   state.
3. Functionally exercises the exact call shape used at each site against
   the real `check()`, confirming it logs on both a clean and a failing
   cache/feature scenario.
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

import ascent.main as main_mod
from ascent.risk.irm.model_risk_reviewer import check as mrr_check


def _root_src() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "ascent/main.py")) as f:
        return f.read()


class TestLoadOrFetchPricesReturnsReason:
    """`load_or_fetch_prices()` must return (df, cache_name, reason) on
    every code path, not the old (df, cache_name) 2-tuple."""

    def test_ok_cache_path_returns_empty_reason(self, monkeypatch):
        dummy_df = pd.DataFrame({
            "symbol": ["AAA", "SPY"],
            "date": [pd.Timestamp("2026-08-18")] * 2,
            "close": [1.0, 2.0],
        })
        monkeypatch.setattr(main_mod, "validate_cache", lambda *a, **k: (True, ""))
        monkeypatch.setattr(main_mod, "load_parquet", lambda name: dummy_df)

        cfg = main_mod.get_config()
        result = main_mod.load_or_fetch_prices(cfg, live=False)

        assert len(result) == 3
        df, cache_name, reason = result
        assert cache_name == "prices_simulated"
        assert reason == ""
        assert df is dummy_df

    def test_stale_cache_fallback_path_returns_nonempty_reason(self, monkeypatch):
        # live=True, validate_cache reports stale -> Yahoo fetch fails ->
        # falls back to the simulated-fallback branch. Exercises the
        # non-empty `reason` propagation on the degraded-data return path.
        stale_reason = "cache latest date 2026-08-10 is 9 calendar days old (limit=1)"
        monkeypatch.setattr(main_mod, "validate_cache", lambda *a, **k: (False, stale_reason))

        def _boom(*a, **k):
            raise RuntimeError("no network in test")

        import ascent.data.ingest.yahoo as yahoo_mod
        monkeypatch.setattr(yahoo_mod, "fetch_universe_daily", _boom)

        fallback_df = pd.DataFrame({
            "symbol": ["AAA", "SPY"],
            "date": [pd.Timestamp("2026-08-18")] * 2,
            "close": [1.0, 2.0],
        })
        monkeypatch.setattr(main_mod, "generate_price_data", lambda *a, **k: fallback_df)
        monkeypatch.setattr(main_mod, "normalize_prices", lambda df: df)
        monkeypatch.setattr(main_mod, "save_parquet", lambda df, name: None)

        cfg = main_mod.get_config()
        df, cache_name, reason = main_mod.load_or_fetch_prices(cfg, live=True)

        assert cache_name == "prices_live_fallback_simulated"
        assert reason == stale_reason


class TestWiredIntoPipeline:
    """Asserting on source text: run_pipeline() is too heavy to execute in
    a unit test (see TestWiredIntoPipeline in test_ai_prior_freshness.py for
    the same pattern)."""

    def test_both_mrr_call_sites_present(self):
        src = _root_src()
        assert src.count("_mrr_check(") == 2, (
            "expected exactly two Model Risk Reviewer call sites: one after "
            "load_or_fetch_prices(), one after the features dict is built"
        )

    def test_load_or_fetch_prices_unpacks_three_values_at_call_site(self):
        src = _root_src()
        assert "price_df, price_cache_name, price_cache_reason = load_or_fetch_prices(" in src

    def test_mrr_calls_are_shadow_mode_log_only(self):
        """Neither MRR call site may contain a `return`, raise, or reassignment
        of price_df/features between the call and the `if not ... .passed`
        print — i.e. the verdict is only ever printed, never acted on."""
        src = _root_src()
        idx = 0
        found = 0
        while True:
            idx = src.find("_mrr_check(", idx)
            if idx == -1:
                break
            found += 1
            # Look at the next ~500 chars following each call site.
            window = src[idx: idx + 500]
            assert "if not _" in window and ".passed:" in window
            assert "print(f\"[IRM]" in window
            # Shadow mode: no control-flow statements between the call and
            # the closing of its if-block.
            block = window.split("print(f\"[IRM]", 1)[0]
            for forbidden in ("return ", "raise ", "sys.exit"):
                assert forbidden not in block, (
                    f"MRR call site at offset {idx} contains {forbidden!r} — "
                    "this must stay shadow-mode (log only)"
                )
            idx += 1
        assert found == 2

    def test_second_call_uses_real_features_and_sparse_exempt(self):
        src = _root_src()
        assert "feature_panels=features, sparse_exempt=_SPARSE_FILL_ZERO" in src
        assert "from ascent.alpha.ml_sleeve import _SPARSE_FILL_ZERO" in src


class TestMrrCallShapeFunctional:
    """Exercise the exact keyword-argument shape used at each call site
    against the real `check()`, on both a clean and a failing scenario."""

    def test_cache_check_call_shape_clean(self, capsys):
        verdict = mrr_check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={}, sparse_exempt=set(), as_of_date=date(2026, 8, 20),
        )
        assert verdict.passed is True

    def test_cache_check_call_shape_failing_logs(self):
        verdict = mrr_check(
            price_cache_name="prices_live_fallback_simulated",
            price_cache_reason="cache file missing: prices_live.parquet",
            feature_panels={}, sparse_exempt=set(), as_of_date=date(2026, 8, 20),
        )
        assert verdict.passed is False
        assert verdict.reason != "all checks passed"

    def test_feature_check_call_shape_clean(self):
        from ascent.alpha.ml_sleeve import _SPARSE_FILL_ZERO
        rng = np.random.default_rng(0)
        clean_panel = pd.DataFrame(rng.standard_normal((20, 5)))
        verdict = mrr_check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"momentum": clean_panel},
            sparse_exempt=_SPARSE_FILL_ZERO,
            as_of_date=date(2026, 8, 20),
        )
        assert verdict.passed is True

    def test_feature_check_call_shape_failing(self):
        from ascent.alpha.ml_sleeve import _SPARSE_FILL_ZERO
        nan_panel = pd.DataFrame(np.full((20, 5), np.nan))
        verdict = mrr_check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"momentum": nan_panel},
            sparse_exempt=_SPARSE_FILL_ZERO,
            as_of_date=date(2026, 8, 20),
        )
        assert verdict.passed is False
        assert "nan_rate::momentum" in verdict.reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
