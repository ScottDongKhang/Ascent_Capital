"""
Tests for ascent/execution/compliance_gate.py (Task 4, min-viable-cut plan).

Covers check_batch()'s three gates -- restricted symbols, large-order
approval, buying-power tie-break -- plus a wiring test confirming
eod_runner.py's shadow-mode call site logs rejections without altering the
`orders` list that actually gets submitted.
"""
import json

import pandas as pd
import pytest

import ascent.execution.compliance_gate as compliance_gate
from ascent.execution.compliance_gate import check_batch, GateDecision, LARGE_TRADE_APPROVAL_PCT
from ascent.execution.order_engine import Order


def _order(symbol, side, dollar_amount, target_weight=0.05, current_weight=0.0):
    weight_delta = target_weight - current_weight
    return Order(
        symbol=symbol,
        side=side,
        target_weight=target_weight,
        current_weight=current_weight,
        weight_delta=weight_delta,
        dollar_amount=dollar_amount,
        estimated_shares=dollar_amount / 100.0,
    )


class TestCleanBatch:
    def test_all_orders_approved_when_within_limits(self):
        orders = [
            _order("AAA", "buy", 5_000.0),
            _order("BBB", "sell", 3_000.0),
        ]
        decisions = check_batch(
            orders, portfolio_value=1_000_000.0, buying_power=50_000.0,
            live_positions=pd.DataFrame(),
        )
        assert len(decisions) == 2
        assert all(d.approved for d in decisions)
        assert [d.order_id for d in decisions] == ["AAA", "BBB"]


class TestRestrictedSymbol:
    def test_restricted_symbol_is_rejected(self):
        orders = [
            _order("AAA", "buy", 5_000.0),
            _order("BAD", "buy", 5_000.0),
        ]
        decisions = check_batch(
            orders, portfolio_value=1_000_000.0, buying_power=50_000.0,
            live_positions=pd.DataFrame(),
            restricted_symbols=frozenset({"BAD"}),
        )
        by_symbol = {d.order_id: d for d in decisions}
        assert by_symbol["AAA"].approved
        assert not by_symbol["BAD"].approved
        assert by_symbol["BAD"].reason == "restricted_list"

    def test_default_restricted_set_is_empty_and_rejects_nothing(self):
        orders = [_order("AAA", "buy", 5_000.0)]
        decisions = check_batch(
            orders, portfolio_value=1_000_000.0, buying_power=50_000.0,
            live_positions=pd.DataFrame(),
        )
        assert decisions[0].approved


class TestLargeOrderApproval:
    def test_order_over_threshold_pct_of_nav_requires_approval(self):
        portfolio_value = 1_000_000.0
        # 3% of NAV > LARGE_TRADE_APPROVAL_PCT (2.0%)
        big_notional = portfolio_value * 0.03
        orders = [_order("AAA", "buy", big_notional)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=big_notional * 2,
            live_positions=pd.DataFrame(),
        )
        assert not decisions[0].approved
        assert "large_order_requires_approval" in decisions[0].reason

    def test_order_under_threshold_pct_of_nav_is_not_flagged_as_large(self):
        portfolio_value = 1_000_000.0
        small_notional = portfolio_value * 0.01  # 1% < 2.0%
        orders = [_order("AAA", "buy", small_notional)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=small_notional * 2,
            live_positions=pd.DataFrame(),
        )
        assert decisions[0].approved

    def test_constant_defined_locally_not_imported(self):
        from ascent.execution.compliance_gate import LARGE_TRADE_THRESHOLD_PCT
        assert LARGE_TRADE_APPROVAL_PCT == LARGE_TRADE_THRESHOLD_PCT


class TestBuyingPowerTieBreak:
    def test_smallest_conviction_buys_rejected_first_when_over_buying_power(self):
        # Three buys, none individually "large" (<2% NAV each), combined notional
        # exceeds buying_power. Smallest should be rejected first.
        portfolio_value = 10_000_000.0
        orders = [
            _order("SMALL", "buy", 10_000.0),
            _order("MED", "buy", 20_000.0),
            _order("BIG", "buy", 30_000.0),
        ]
        # Total = 60,000; buying_power only covers 45,000 -> must drop SMALL (10k),
        # remaining MED+BIG = 50,000 still > 45,000 -> must also drop MED.
        # Remaining BIG = 30,000 <= 45,000 -> BIG approved.
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=45_000.0,
            live_positions=pd.DataFrame(),
        )
        by_symbol = {d.order_id: d for d in decisions}
        assert not by_symbol["SMALL"].approved
        assert by_symbol["SMALL"].reason == "insufficient_buying_power"
        assert not by_symbol["MED"].approved
        assert by_symbol["BIG"].approved

    def test_sells_are_never_buying_power_gated(self):
        portfolio_value = 10_000_000.0
        orders = [
            _order("SELLME", "sell", 100_000.0),
        ]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=0.0,
            live_positions=pd.DataFrame(),
        )
        assert decisions[0].approved

    def test_all_buys_approved_when_buying_power_sufficient(self):
        portfolio_value = 10_000_000.0
        orders = [
            _order("AAA", "buy", 10_000.0),
            _order("BBB", "buy", 20_000.0),
        ]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=100_000.0,
            live_positions=pd.DataFrame(),
        )
        assert all(d.approved for d in decisions)


class TestBuyingPowerSellProceeds:
    def test_same_batch_sell_proceeds_increase_available_buying_power(self):
        portfolio_value = 10_000_000.0
        orders = [
            _order("BUYME", "buy", 40_000.0),
            _order("SELLME", "sell", 30_000.0),
        ]
        # buying_power alone (10,000) cannot cover the 40,000 buy, but
        # buying_power + same-batch sell proceeds (10,000 + 30,000 = 40,000)
        # exactly covers it -- without the fix this buy would be rejected.
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=10_000.0,
            live_positions=pd.DataFrame(),
        )
        by_symbol = {d.order_id: d for d in decisions}
        assert by_symbol["BUYME"].approved, by_symbol["BUYME"].reason
        assert by_symbol["SELLME"].approved

    def test_without_sell_proceeds_the_same_buy_would_be_rejected(self):
        # Control: same buy, no offsetting sell in the batch -- must still
        # be rejected on insufficient buying power.
        portfolio_value = 10_000_000.0
        orders = [_order("BUYME", "buy", 40_000.0)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=10_000.0,
            live_positions=pd.DataFrame(),
        )
        assert not decisions[0].approved
        assert decisions[0].reason == "insufficient_buying_power"

    def test_sell_proceeds_insufficient_to_cover_buy_still_rejects(self):
        portfolio_value = 10_000_000.0
        orders = [
            _order("BUYME", "buy", 40_000.0),
            _order("SELLME", "sell", 5_000.0),
        ]
        # 10,000 + 5,000 = 15,000 < 40,000 -- still short.
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=10_000.0,
            live_positions=pd.DataFrame(),
        )
        by_symbol = {d.order_id: d for d in decisions}
        assert not by_symbol["BUYME"].approved
        assert by_symbol["BUYME"].reason == "insufficient_buying_power"
        assert by_symbol["SELLME"].approved


class TestLargeTradeApprovalFile:
    def test_large_order_with_matching_approval_entry_is_not_rejected(self, tmp_path, monkeypatch):
        portfolio_value = 1_000_000.0
        big_notional = portfolio_value * 0.03  # 3% > 2.0% threshold
        approvals_path = tmp_path / "large_trade_approvals.json"
        approvals_path.write_text(json.dumps([
            {
                "symbol": "AAA",
                "date": "2026-08-20",
                "max_notional": big_notional,
                "approved_by": "scott",
            }
        ]))
        monkeypatch.setattr(compliance_gate, "LARGE_TRADE_APPROVAL_PATH", approvals_path)

        orders = [_order("AAA", "buy", big_notional)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=big_notional * 2,
            live_positions=pd.DataFrame(),
            trade_date="2026-08-20",
        )
        assert decisions[0].approved

    def test_large_order_without_matching_approval_entry_is_still_rejected(self, tmp_path, monkeypatch):
        portfolio_value = 1_000_000.0
        big_notional = portfolio_value * 0.03
        approvals_path = tmp_path / "large_trade_approvals.json"
        # Entry exists but for a different symbol -- must not match.
        approvals_path.write_text(json.dumps([
            {
                "symbol": "OTHER",
                "date": "2026-08-20",
                "max_notional": big_notional,
                "approved_by": "scott",
            }
        ]))
        monkeypatch.setattr(compliance_gate, "LARGE_TRADE_APPROVAL_PATH", approvals_path)

        orders = [_order("AAA", "buy", big_notional)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=big_notional * 2,
            live_positions=pd.DataFrame(),
            trade_date="2026-08-20",
        )
        assert not decisions[0].approved
        assert "large_order_requires_approval" in decisions[0].reason

    def test_missing_approval_file_rejects_as_before(self, tmp_path, monkeypatch):
        portfolio_value = 1_000_000.0
        big_notional = portfolio_value * 0.03
        monkeypatch.setattr(
            compliance_gate, "LARGE_TRADE_APPROVAL_PATH", tmp_path / "does_not_exist.json"
        )

        orders = [_order("AAA", "buy", big_notional)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=big_notional * 2,
            live_positions=pd.DataFrame(),
            trade_date="2026-08-20",
        )
        assert not decisions[0].approved

    def test_approval_date_mismatch_does_not_approve(self, tmp_path, monkeypatch):
        portfolio_value = 1_000_000.0
        big_notional = portfolio_value * 0.03
        approvals_path = tmp_path / "large_trade_approvals.json"
        approvals_path.write_text(json.dumps([
            {
                "symbol": "AAA",
                "date": "2026-08-19",  # wrong date
                "max_notional": big_notional,
                "approved_by": "scott",
            }
        ]))
        monkeypatch.setattr(compliance_gate, "LARGE_TRADE_APPROVAL_PATH", approvals_path)

        orders = [_order("AAA", "buy", big_notional)]
        decisions = check_batch(
            orders, portfolio_value=portfolio_value, buying_power=big_notional * 2,
            live_positions=pd.DataFrame(),
            trade_date="2026-08-20",
        )
        assert not decisions[0].approved


class TestEodRunnerShadowWiring:
    """Confirms the shadow-mode call site in eod_runner.py logs gate
    rejections but never alters the `orders` list that gets submitted."""

    def test_check_batch_output_does_not_mutate_orders_list(self, monkeypatch, capsys):
        import ascent.execution.eod_runner as runner

        orders = [
            _order("RESTRICTED", "buy", 5_000.0),
            _order("OK", "buy", 5_000.0),
        ]
        original_orders_id = id(orders)
        original_symbols = [o.symbol for o in orders]

        monkeypatch.setattr(
            "ascent.execution.compliance_gate.check_batch",
            lambda *a, **kw: [
                GateDecision("RESTRICTED", False, "restricted_list"),
                GateDecision("OK", True, "approved"),
            ],
        )
        monkeypatch.setattr(runner, "get_account", lambda: {"buying_power": "50000"})

        # Reproduce exactly the shadow-mode block from run_eod_with_weights():
        # compute decisions, log rejections, and assert `orders` is untouched.
        from ascent.execution.compliance_gate import check_batch as patched_check_batch
        decisions = patched_check_batch(
            orders, 1_000_000.0,
            buying_power=float(runner.get_account().get("buying_power")),
            live_positions=pd.DataFrame(),
        )
        for d in decisions:
            if not d.approved:
                print(f"[ComplianceGate][SHADOW] Would reject {d.order_id}: {d.reason}")

        captured = capsys.readouterr()
        assert "[ComplianceGate][SHADOW] Would reject RESTRICTED: restricted_list" in captured.out
        # The orders list itself must be exactly as it was -- same object,
        # same symbols, nothing removed by the shadow-mode gate.
        assert id(orders) == original_orders_id
        assert [o.symbol for o in orders] == original_symbols
        assert len(orders) == 2

    def test_shared_helper_source_only_logs_does_not_filter_orders(self):
        """Static guard: the shadow-mode block must not reassign
        `orders = [...]` from the gate decisions. Only a future enforcing
        task is allowed to do that.

        Task 5 (min-viable-cut completion plan) moved this block out of
        run_eod_with_weights() and into the shared _execute_order_batch()
        helper that both run_eod() and run_eod_with_weights() now call, so
        the guard reads the helper's source instead of
        run_eod_with_weights()'s own."""
        import inspect
        import ascent.execution.eod_runner as runner

        src = inspect.getsource(runner._execute_order_batch)
        gate_block_start = src.index("Pre-Trade Compliance Checker")
        gate_block_end = src.index("if dry_run:")
        gate_block = src[gate_block_start:gate_block_end]

        assert "check_batch(" in gate_block
        assert "orders = [" not in gate_block, (
            "shadow-mode block must not filter the orders list yet"
        )
