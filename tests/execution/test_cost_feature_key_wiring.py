"""
Regression test for the dead cost-model / TWAP wiring bug.

Audit finding (PROJECT_STATUS.md §1.6): ascent/execution/eod_runner.py built
`_required_cost_keys = {"dollar_volume"}` and checked it against the dict
returned by `extract_cost_features()` (ascent/execution/cost_model.py), which
never returns a key literally named "dollar_volume" — only "dollar_vol_21d"
(and, when a "vol_21d" input is supplied, "vol_21d"). Because the required
key could never be present, `features_arg` was always None, which silently
disabled:
  1. apply_cost_filter() — HIGH_IMPACT orders above the ADV participation cap
     were never blocked.
  2. TWAP routing in order_engine.compute_orders() — its `if ... and features`
     gate was always false regardless of the TWAP_ENABLED kill switch.

This test reproduces the bug at the smallest testable unit (the `issubset`
check against a realistic extract_cost_features() output), proves the old
key set was always empty-intersecting / falsy, proves the fixed key set is
satisfied by real output, and then proves the fix actually causes
apply_cost_filter to block a HIGH_IMPACT order end-to-end through
compute_orders().
"""
import pandas as pd
import pytest

from ascent.execution.cost_model import extract_cost_features
from ascent.execution.order_engine import Order, compute_orders


def _build_realistic_cost_features():
    """Mirror what eod_runner.py builds: a dates x symbols dollar_volume frame."""
    dates = pd.date_range("2026-06-01", periods=25, freq="B")
    dv = pd.DataFrame(
        {
            "AAA": [5_000_000.0] * 25,   # liquid — small order should pass
            "BBB": [10_000.0] * 25,      # illiquid — same $ order should be HIGH_IMPACT
        },
        index=dates,
    )
    return extract_cost_features({"dollar_volume": dv})


class TestRequiredCostKeysBugRepro:
    def test_old_buggy_key_never_matches_real_output(self):
        """The pre-fix key set {'dollar_volume'} can never be a subset of
        extract_cost_features()'s output — this is the exact bug."""
        cost_features = _build_realistic_cost_features()
        assert "dollar_vol_21d" in cost_features  # sanity: real key IS present
        assert "dollar_volume" not in cost_features  # bug: wrong key checked

        buggy_required_keys = {"dollar_volume"}
        features_arg = (
            cost_features
            if (cost_features and buggy_required_keys.issubset(cost_features))
            else None
        )
        assert features_arg is None, (
            "pre-fix logic must always disable cost features — "
            "if this fails the repro is wrong, not the fix"
        )

    def test_fixed_key_matches_real_output(self):
        """The fix ({'dollar_vol_21d'}) must actually be satisfied by real
        extract_cost_features() output built the way eod_runner builds it."""
        cost_features = _build_realistic_cost_features()

        fixed_required_keys = {"dollar_vol_21d"}
        features_arg = (
            cost_features
            if (cost_features and fixed_required_keys.issubset(cost_features))
            else None
        )
        assert features_arg is not None
        assert features_arg is cost_features

    def test_vol_21d_not_required_since_eod_runner_never_supplies_it(self):
        """eod_runner.py only ever passes {'dollar_volume': df} into
        extract_cost_features, so 'vol_21d' can never be produced from the
        live path. Requiring it would reproduce the same always-None bug."""
        cost_features = _build_realistic_cost_features()
        assert "vol_21d" not in cost_features

        overly_strict_keys = {"dollar_vol_21d", "vol_21d"}
        features_arg = (
            cost_features
            if (cost_features and overly_strict_keys.issubset(cost_features))
            else None
        )
        assert features_arg is None, (
            "requiring vol_21d would silently re-disable cost features on the "
            "real eod_runner input path"
        )


class TestCostFilterActuallyRuns:
    def test_high_impact_order_is_blocked_when_features_wired_correctly(self):
        """End-to-end: compute_orders() must actually invoke apply_cost_filter
        and drop a HIGH_IMPACT order when features_arg is the (fixed) real
        cost-features dict, mirroring what eod_runner now passes in."""
        cost_features = _build_realistic_cost_features()

        target_weights = pd.Series({"AAA": 0.05, "BBB": 0.05})
        current_positions = pd.DataFrame(columns=["symbol", "weight"])

        # Prices needed for share-count math; patch _get_approx_price so the
        # order isn't skipped for lack of price data.
        import ascent.execution.order_engine as oe

        def _fake_price(symbol, positions):
            return 100.0

        orig = oe._get_approx_price
        oe._get_approx_price = _fake_price
        try:
            orders, diff_df = compute_orders(
                target_weights,
                current_positions,
                portfolio_value=1_000_000.0,
                features=cost_features,
            )
        finally:
            oe._get_approx_price = orig

        symbols_remaining = {o.symbol for o in orders}
        # AAA: $50k order against $5M/day ADV -> ~0.4% participation/day equiv -> passes.
        # BBB: $50k order against $10k/day ADV -> wildly over the 10% cap -> HIGH_IMPACT, blocked.
        assert "AAA" in symbols_remaining
        assert "BBB" not in symbols_remaining
        assert not diff_df.empty
        blocked_rows = diff_df[diff_df["symbol"] == "BBB"]
        assert (blocked_rows["action"] == "blocked_high_impact").all()

    def test_cost_filter_never_ran_with_old_broken_features_arg(self):
        """Sanity check of the pre-fix world: features=None means
        apply_cost_filter is never even attempted, so HIGH_IMPACT orders are
        NOT blocked."""
        target_weights = pd.Series({"AAA": 0.05, "BBB": 0.05})
        current_positions = pd.DataFrame(columns=["symbol", "weight"])

        import ascent.execution.order_engine as oe

        def _fake_price(symbol, positions):
            return 100.0

        orig = oe._get_approx_price
        oe._get_approx_price = _fake_price
        try:
            orders, diff_df = compute_orders(
                target_weights,
                current_positions,
                portfolio_value=1_000_000.0,
                features=None,  # this is what the bug always produced
            )
        finally:
            oe._get_approx_price = orig

        symbols_remaining = {o.symbol for o in orders}
        # Without cost features, nothing gets blocked — BBB survives despite
        # being wildly illiquid.
        assert {"AAA", "BBB"}.issubset(symbols_remaining)


class TestTwapWiring:
    def test_execute_twap_is_actually_called_when_enabled_and_routed(self, monkeypatch):
        """Confirms/repros the second audit claim: previously the TWAP block
        only logged and never called execute_twap. With the fix, when the
        kill switch is (hypothetically) on and routing triggers, execute_twap
        must actually be invoked."""
        import ascent.execution.order_engine as oe

        calls = []

        def _fake_execute_twap(symbol, total_dollars, side, adv, price, urgency="normal", _alpaca_api=None):
            calls.append(symbol)
            return [{"status": "submitted", "symbol": symbol, "slice": 0}]

        monkeypatch.setattr(oe, "TWAP_ENABLED", True)
        monkeypatch.setattr(oe, "should_use_twap", lambda trade, adv, threshold_pct=0.05: True)
        monkeypatch.setattr(oe, "execute_twap", _fake_execute_twap)

        def _fake_price(symbol, positions):
            return 100.0

        monkeypatch.setattr(oe, "_get_approx_price", _fake_price)

        cost_features = _build_realistic_cost_features()
        target_weights = pd.Series({"AAA": 0.05})
        current_positions = pd.DataFrame(columns=["symbol", "weight"])

        orders, diff_df = oe.compute_orders(
            target_weights,
            current_positions,
            portfolio_value=1_000_000.0,
            features=cost_features,
        )

        assert calls == ["AAA"], "execute_twap must actually be called, not just logged about"
        # A symbol that was fully routed (and reported a submitted fill) must
        # be pulled out of the plain-order list to avoid double execution.
        assert "AAA" not in {o.symbol for o in orders}

    def test_twap_disabled_by_default_leaves_plain_order_path_untouched(self, monkeypatch):
        """The kill switch must still default off, and with it off the order
        must still flow through the normal (non-TWAP) submission path exactly
        as before the fix — no behaviour change while TWAP_ENABLED=False."""
        from ascent.execution.twap_executor import TWAP_ENABLED as LIVE_TWAP_ENABLED
        assert LIVE_TWAP_ENABLED is False, "TWAP_ENABLED kill switch must default to False"

        import ascent.execution.order_engine as oe

        assert oe.TWAP_ENABLED is False

        def _fake_price(symbol, positions):
            return 100.0

        monkeypatch.setattr(oe, "_get_approx_price", _fake_price)

        cost_features = _build_realistic_cost_features()
        target_weights = pd.Series({"AAA": 0.05})
        current_positions = pd.DataFrame(columns=["symbol", "weight"])

        orders, diff_df = oe.compute_orders(
            target_weights,
            current_positions,
            portfolio_value=1_000_000.0,
            features=cost_features,
        )

        # AAA is liquid enough to pass the cost filter and TWAP is off, so it
        # must still appear as a plain order.
        assert "AAA" in {o.symbol for o in orders}
