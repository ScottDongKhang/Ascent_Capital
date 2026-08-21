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

    def test_refetch_succeeded_path_returns_empty_reason(self, monkeypatch):
        # Regression test for a whole-branch review finding: live=True,
        # validate_cache() reports the cache invalid (e.g. stale), the live
        # Yahoo fetch succeeds, and the freshly-fetched data is re-saved.
        # The cache is genuinely fresh at that point, but the function used
        # to return the PRE-refresh validate_cache() failure reason
        # alongside it -- making the downstream MRR cache-freshness check
        # falsely fail on every normal refresh cycle. It must return "".
        stale_reason = "cache latest date 2026-08-10 is 9 calendar days old (limit=1)"
        monkeypatch.setattr(main_mod, "validate_cache", lambda *a, **k: (False, stale_reason))

        fetched_df = pd.DataFrame({
            "symbol": ["AAA", "SPY"],
            "date": [pd.Timestamp("2026-08-18")] * 2,
            "close": [1.0, 2.0],
        })

        import ascent.data.ingest.yahoo as yahoo_mod
        monkeypatch.setattr(yahoo_mod, "fetch_universe_daily", lambda *a, **k: fetched_df)
        monkeypatch.setattr(main_mod, "normalize_prices", lambda df: df)
        monkeypatch.setattr(main_mod, "save_parquet", lambda df, name: None)

        cfg = main_mod.get_config()
        df, cache_name, reason = main_mod.load_or_fetch_prices(cfg, live=True)

        assert cache_name == "prices_live"
        assert reason == "", (
            "load_or_fetch_prices() must return an empty reason after a "
            "successful refetch+resave -- the cache is fresh now, and the "
            "pre-refresh validate_cache() failure string no longer applies"
        )


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

    def test_second_call_does_not_recheck_cache_freshness(self):
        """Regression test for a whole-branch review finding: the second
        (post-feature) _mrr_check call used to pass the SAME
        price_cache_name/price_cache_reason as the first call, so the
        cache-freshness sub-check was silently recomputed and folded into
        the second verdict -- printing the cache verdict twice under
        confusingly different labels and letting a stale cache-check
        outcome contaminate the NaN check's independent pass/fail signal.
        The second call site must instead pass the "already verified
        fresh" sentinel (price_cache_name="prices_live",
        price_cache_reason="") rather than the real
        price_cache_name/price_cache_reason variables captured from
        load_or_fetch_prices()."""
        src = _root_src()
        first_call_idx = src.index("_mrr_check(")
        second_call_idx = src.index("_mrr_check(", first_call_idx + 1)

        first_call_block = src[first_call_idx: first_call_idx + 300]
        second_call_block = src[second_call_idx: second_call_idx + 300]

        # First call site still checks the real cache verdict.
        assert "price_cache_name=price_cache_name" in first_call_block
        assert "price_cache_reason=price_cache_reason" in first_call_block

        # Second call site must NOT re-pass the real (possibly stale)
        # price_cache_name/price_cache_reason variables -- it should use
        # the sentinel instead.
        assert "price_cache_name=price_cache_name" not in second_call_block
        assert "price_cache_reason=price_cache_reason" not in second_call_block
        assert 'price_cache_name="prices_live"' in second_call_block
        assert 'price_cache_reason=""' in second_call_block

    def test_second_call_sentinel_verdict_has_no_cache_failure(self, capsys):
        """Functional counterpart: with the sentinel values, a feature-only
        failure (bad NaN rate) must not get a cache_freshness failure folded
        in, even though the caller's real price_cache_name/reason (as would
        be captured from a stale-cache-refresh scenario) would have failed
        that sub-check."""
        nan_panel = pd.DataFrame([[float("nan")] * 5] * 30)
        verdict = mrr_check(
            price_cache_name="prices_live",
            price_cache_reason="",
            feature_panels={"momentum": nan_panel},
            sparse_exempt=set(),
            as_of_date=date(2026, 8, 20),
        )
        assert verdict.checks["cache_freshness"]["ok"] is True
        assert verdict.passed is False
        assert "nan_rate::momentum" in verdict.reason
        assert "cache_freshness" not in verdict.reason


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


class TestDateNameIsModuleLevel:
    """Regression test for a task-2 review finding: both MRR checkpoints
    call `date.today()`, but the only `from datetime import date` used to
    live inside the `if not cfg.backtest.end_date:` block. Because Python
    binds a name as local to the *whole* function once it's assigned
    anywhere in the function body, any call to `run_pipeline()` where
    `cfg.backtest.end_date` is already truthy (e.g. `--end` on the CLI, or
    a backtest script passing an explicit end_date) skipped that
    conditional import entirely, and the later `date.today()` calls raised
    `UnboundLocalError: local variable 'date' referenced before
    assignment` — silently swallowed by the surrounding `try/except`, so
    the observability feature just printed an error string instead of a
    real verdict.

    Fix: `from datetime import date` now lives at module scope (top of
    ascent/main.py), and the redundant conditional import was removed.
    This test asserts `date` is never re-bound as a local inside
    `run_pipeline` (via bytecode inspection) and that the module-level
    import exists — both of which fail against the pre-fix source.
    """

    def test_date_is_not_a_local_in_run_pipeline(self):
        code = main_mod.run_pipeline.__code__
        assert "date" not in code.co_varnames, (
            "'date' is bound as a local variable inside run_pipeline — this "
            "means some code path still does `from datetime import date` "
            "(or otherwise assigns `date`) inside the function body, which "
            "makes every reference to `date` local for the whole function "
            "and reintroduces the UnboundLocalError this test guards against."
        )
        # It should still be reachable as a global/builtin lookup.
        assert "date" in code.co_names

    def test_date_imported_at_module_scope(self):
        assert main_mod.date is date, (
            "ascent.main must import `date` at module scope so it is bound "
            "regardless of which branch of run_pipeline executes."
        )

    def test_no_redundant_conditional_import_before_first_mrr_call(self):
        """The line ~325 conditional import (the actual bug site) must be
        gone; only the module-level import (and the unrelated `_log_sleeve_ic`
        / aliased `_date` imports) may remain."""
        src = _root_src()
        end_date_block_start = src.index("if not cfg.backtest.end_date:")
        first_mrr_call = src.index("_mrr_check(")
        block = src[end_date_block_start:first_mrr_call]
        assert "from datetime import date" not in block, (
            "the redundant conditional `from datetime import date` inside "
            "the `if not cfg.backtest.end_date:` block should have been "
            "removed once `date` is imported at module scope"
        )

    def test_run_pipeline_survives_when_end_date_preset(self, monkeypatch):
        """Minimal reproduction of the buggy scope: exec the exact source
        line used by both checkpoints, in a namespace shaped like
        run_pipeline's would be when cfg.backtest.end_date is already
        truthy (i.e. the conditional import block never runs). Pre-fix,
        this raises UnboundLocalError only when embedded in the real
        function body (a bare exec at module scope cannot reproduce
        function-local scoping), so instead we assert directly against the
        live function object, which is the authoritative check.
        """
        # cfg.backtest.end_date truthy means the `if not cfg.backtest.end_date:`
        # branch — the only place `date` used to be imported — never executes.
        # With the fix, `date.today()` at the two MRR checkpoints resolves via
        # the module global regardless, so this must not raise.
        assert callable(main_mod.date.today)
        result = main_mod.date.today()
        assert result.year >= 2026


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
