"""
Tests for ascent/execution/eod_runner.py's _execute_order_batch() -- the
shared kill-switch-check -> Task 4 compliance-gate(shadow) ->
cancel-all-orders -> submit-loop -> per-order-audit helper extracted in
Task 5 (min-viable-cut completion plan) from run_eod() and
run_eod_with_weights(), which independently reimplemented this sequence
with subtly different behavior.

Covers every Step-1 divergence from task-5-report.md as its own test, plus
functional tests confirming both run_eod() and run_eod_with_weights() call
the shared helper (so Task 4's compliance-gate call fires from both).
"""
import pandas as pd
import pytest

import ascent.execution.eod_runner as runner
from ascent.execution import kill_switch
from ascent.execution.order_engine import Order


def _order(symbol="AAA", side="buy", target_weight=0.05, dollar_amount=5_000.0):
    return Order(
        symbol=symbol,
        side=side,
        target_weight=target_weight,
        current_weight=0.0,
        weight_delta=target_weight,
        dollar_amount=dollar_amount,
        estimated_shares=dollar_amount / 100.0,
    )


@pytest.fixture(autouse=True)
def _quiet_compliance_gate(monkeypatch):
    """Every test here goes through the compliance-gate shadow block. Make it
    a harmless no-op (approve everything) unless a test overrides it, and
    stub get_account so the buying-power lookup inside the gate doesn't hit
    the network."""
    monkeypatch.setattr(runner, "get_account", lambda: {"buying_power": "1000000"})
    monkeypatch.setattr(
        "ascent.execution.compliance_gate.check_batch",
        lambda orders, portfolio_value, buying_power, live_positions, **kw: [],
    )


@pytest.fixture(autouse=True)
def _no_real_cancel_or_submit(monkeypatch):
    """Never let a test accidentally hit the real Alpaca broker."""
    monkeypatch.setattr(runner, "cancel_all_orders", lambda: None)


class TestDivergence1_KillSwitchExceptionPropagation:
    """Step-1 divergence #1 (the known one): run_eod_with_weights() used to
    swallow any non-KillSwitchTriggered exception from kill_switch.check()
    and keep trading; run_eod() always let it propagate. Resolved to
    run_eod()'s stricter behavior in the shared helper."""

    def test_non_kill_switch_exception_propagates_and_orders_are_not_submitted(self, monkeypatch):
        def _boom(current_nav=None):
            raise ValueError("some unrelated kill-switch plumbing failure")

        monkeypatch.setattr(kill_switch, "check", _boom)
        submitted = []
        monkeypatch.setattr(runner, "submit_order", lambda **kw: submitted.append(kw) or {"id": "x"})

        with pytest.raises(ValueError, match="unrelated kill-switch"):
            runner._execute_order_batch(
                [_order()], pd.DataFrame(), 100_000.0, "2026-08-20",
            )

        assert submitted == [], (
            "a non-KillSwitchTriggered exception must abort before any order "
            "is submitted, not be swallowed and continued past"
        )


class TestDivergence2_KillSwitchTriggerAlwaysAudited:
    """Step-1 divergence #2: only run_eod() called _audit('kill_switch_triggered',
    ...); run_eod_with_weights() skipped the audit trail entirely on a trip.
    Unified to always audit."""

    def test_kill_switch_trigger_is_audited_and_reraised(self, monkeypatch):
        def _trip(current_nav=None):
            raise kill_switch.KillSwitchTriggered("drawdown exceeded")

        monkeypatch.setattr(kill_switch, "check", _trip)
        audit_calls = []
        monkeypatch.setattr(runner, "_audit", lambda event_type, payload: audit_calls.append((event_type, payload)))

        with pytest.raises(kill_switch.KillSwitchTriggered):
            runner._execute_order_batch(
                [_order()], pd.DataFrame(), 100_000.0, "2026-08-20",
            )

        assert len(audit_calls) == 1
        event_type, payload = audit_calls[0]
        assert event_type == "kill_switch_triggered"
        assert payload["nav"] == 100_000.0
        assert payload["date"] == "2026-08-20"


class TestDivergence3_CloseVsSubmitForFullLiquidation:
    """Step-1 divergence #3: run_eod_with_weights() used close_position() for
    full-liquidation sells (target_weight == 0.0); run_eod() always used
    submit_order(). This changes what gets submitted to the broker, so it
    stays a per-caller flag rather than being unified."""

    def test_flag_true_uses_close_position_for_full_liquidation_sell(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})
        monkeypatch.setattr(
            "ascent.execution.order_engine._get_approx_price",
            lambda symbol, positions: 100.0,
        )
        closed, submitted = [], []
        # close_position is imported locally inside the helper -- patch the
        # source module it's imported from.
        monkeypatch.setattr(
            "ascent.execution.alpaca_broker.close_position",
            lambda symbol: closed.append(symbol) or {"id": "c1"},
        )
        monkeypatch.setattr(runner, "submit_order", lambda **kw: submitted.append(kw) or {"id": "s1"})

        full_exit = _order(symbol="EXIT", side="sell", target_weight=0.0, dollar_amount=1_000.0)
        executed, skipped = runner._execute_order_batch(
            [full_exit], pd.DataFrame(), 100_000.0, "2026-08-20",
            use_close_position_for_full_liquidation=True,
        )

        assert closed == ["EXIT"]
        assert submitted == []
        assert executed[0]["symbol"] == "EXIT"

    def test_flag_false_uses_submit_order_for_the_same_full_liquidation_sell(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})
        monkeypatch.setattr(
            "ascent.execution.order_engine._get_approx_price",
            lambda symbol, positions: 100.0,
        )
        closed, submitted = [], []
        monkeypatch.setattr(
            "ascent.execution.alpaca_broker.close_position",
            lambda symbol: closed.append(symbol) or {"id": "c1"},
        )
        monkeypatch.setattr(runner, "submit_order", lambda **kw: submitted.append(kw) or {"id": "s1"})

        full_exit = _order(symbol="EXIT", side="sell", target_weight=0.0, dollar_amount=1_000.0)
        executed, skipped = runner._execute_order_batch(
            [full_exit], pd.DataFrame(), 100_000.0, "2026-08-20",
            use_close_position_for_full_liquidation=False,
        )

        assert closed == []
        assert len(submitted) == 1 and submitted[0]["symbol"] == "EXIT"
        assert executed[0]["symbol"] == "EXIT"


class TestDivergence4_OrderSubmittedAlwaysAudited:
    """Step-1 divergence #4: run_eod() audited every successful submission
    via _audit('order_submitted', ...); run_eod_with_weights() never did.
    Unified to always audit."""

    def test_successful_submission_is_audited(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})
        monkeypatch.setattr(
            "ascent.execution.order_engine._get_approx_price",
            lambda symbol, positions: 100.0,
        )
        monkeypatch.setattr(runner, "submit_order", lambda **kw: {"id": "abc123"})
        audit_calls = []
        monkeypatch.setattr(runner, "_audit", lambda event_type, payload: audit_calls.append((event_type, payload)))

        executed, skipped = runner._execute_order_batch(
            [_order(symbol="AAA")], pd.DataFrame(), 100_000.0, "2026-08-20",
        )

        order_audits = [c for c in audit_calls if c[0] == "order_submitted"]
        assert len(order_audits) == 1
        assert order_audits[0][1]["symbol"] == "AAA"
        assert order_audits[0][1]["order_id"] == "abc123"
        assert executed == [{
            "symbol": "AAA", "side": "buy", "qty": 50.0,
            "dollar_amount": 5_000.0, "order_id": "abc123",
        }]


class TestAuditFailureDoesNotMisclassifySubmittedOrder:
    """Task-5 review finding (Important): the order_submitted audit call
    used to sit inside the same try block as the broker submission. If
    audit_trail.record() raised (disk full, permissions, lock contention --
    only import failure is guarded by _audit's own no-op fallback), the
    outer except caught it and filed a genuinely-submitted order into
    `skipped` instead of `executed`, even though the broker call had already
    succeeded. Fixed by giving the _audit() call its own try/except so a
    submitted order's bookkeeping is unconditional."""

    def test_audit_raising_still_lands_order_in_executed_not_skipped(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})
        monkeypatch.setattr(
            "ascent.execution.order_engine._get_approx_price",
            lambda symbol, positions: 100.0,
        )
        monkeypatch.setattr(runner, "submit_order", lambda **kw: {"id": "abc123"})

        def _boom(event_type, payload):
            raise OSError("disk full")

        monkeypatch.setattr(runner, "_audit", _boom)

        executed, skipped = runner._execute_order_batch(
            [_order(symbol="AAA")], pd.DataFrame(), 100_000.0, "2026-08-20",
        )

        assert executed == [{
            "symbol": "AAA", "side": "buy", "qty": 50.0,
            "dollar_amount": 5_000.0, "order_id": "abc123",
        }], "a broker-submitted order must stay executed even if the audit log write fails"
        assert skipped == []


class TestDivergence5_SkippedEntriesAreDictsWithReason:
    """Step-1 divergence #5: run_eod()'s `skipped` entries were always
    {"symbol", "reason"} dicts; run_eod_with_weights()'s were bare symbol
    strings with no reason recorded anywhere. Unified to the richer dict
    shape for both callers."""

    def test_price_unavailable_qty_too_small_and_submit_failure_all_carry_a_reason(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})

        def _price(symbol, positions):
            return {"NOPRICE": None, "TINY": 1_000_000.0, "FAILS": 100.0}[symbol]

        monkeypatch.setattr("ascent.execution.order_engine._get_approx_price", _price)

        def _submit(**kw):
            raise RuntimeError("broker rejected")

        monkeypatch.setattr(runner, "submit_order", _submit)

        orders = [
            _order(symbol="NOPRICE", dollar_amount=5_000.0),
            _order(symbol="TINY", dollar_amount=0.0001),   # qty rounds under 0.001
            _order(symbol="FAILS", dollar_amount=5_000.0),
        ]
        executed, skipped = runner._execute_order_batch(
            orders, pd.DataFrame(), 100_000.0, "2026-08-20",
        )

        assert executed == []
        by_symbol = {s["symbol"]: s for s in skipped}
        assert set(by_symbol) == {"NOPRICE", "TINY", "FAILS"}
        assert by_symbol["NOPRICE"]["reason"] == "price unavailable"
        assert by_symbol["TINY"]["reason"] == "qty too small"
        assert "broker rejected" in by_symbol["FAILS"]["reason"]


class TestDryRun:
    def test_dry_run_submits_nothing_and_returns_empty_lists(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})
        submitted = []
        monkeypatch.setattr(runner, "submit_order", lambda **kw: submitted.append(kw) or {"id": "x"})
        cancelled = []
        monkeypatch.setattr(runner, "cancel_all_orders", lambda: cancelled.append(True))

        executed, skipped = runner._execute_order_batch(
            [_order()], pd.DataFrame(), 100_000.0, "2026-08-20", dry_run=True,
        )

        assert executed == []
        assert skipped == []
        assert submitted == []
        assert cancelled == [], "dry_run must not even cancel open orders"


class TestComplianceGateFiresFromBothEntrypoints:
    """Task 4's shadow-mode compliance gate must fire from both run_eod()
    and run_eod_with_weights() post-refactor -- previously it only fired
    from run_eod_with_weights(); Task 5's whole point is that both paths
    now share it via _execute_order_batch()."""

    def test_gate_fires_directly_through_the_shared_helper(self, monkeypatch):
        # The autouse fixture stubs check_batch to return []; here we replace
        # it with a spy to prove it is actually called with the batch.
        calls = []

        def _spy_check_batch(orders, portfolio_value, buying_power, live_positions, **kw):
            calls.append((list(orders), portfolio_value))
            return []

        monkeypatch.setattr("ascent.execution.compliance_gate.check_batch", _spy_check_batch)
        monkeypatch.setattr(kill_switch, "check", lambda current_nav=None: {})
        monkeypatch.setattr(
            "ascent.execution.order_engine._get_approx_price",
            lambda symbol, positions: 100.0,
        )
        monkeypatch.setattr(runner, "submit_order", lambda **kw: {"id": "x"})

        runner._execute_order_batch(
            [_order()], pd.DataFrame(), 100_000.0, "2026-08-20",
        )

        assert len(calls) == 1, "compliance gate check_batch() must fire exactly once per batch"

    def test_run_eod_and_run_eod_with_weights_both_call_the_shared_helper(self):
        """Static guard: both public entrypoints must route order submission
        through _execute_order_batch() -- the only place Task 4's
        compliance-gate call now lives -- rather than reimplementing the
        kill-switch/cancel/submit sequence themselves."""
        import inspect

        run_eod_src = inspect.getsource(runner.run_eod)
        run_eod_with_weights_src = inspect.getsource(runner.run_eod_with_weights)

        assert "_execute_order_batch(" in run_eod_src
        assert "_execute_order_batch(" in run_eod_with_weights_src
        # And neither should still contain its own inline cancel_all_orders()
        # call outside the shared helper -- that call now lives exclusively
        # inside _execute_order_batch().
        assert "cancel_all_orders()" not in run_eod_src
        assert "cancel_all_orders()" not in run_eod_with_weights_src
