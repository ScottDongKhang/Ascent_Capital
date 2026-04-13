# tests/test_phase2_hardening.py
"""
Phase 2 hardening — Fix 2a: async approval gate.

Tests for wait_for_approval_async() in ascent/execution/approval_server.py.
"""
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pending(tmp_path, expires_delta_seconds: int) -> dict:
    """Return a pending-state dict with expires_at offset from now."""
    expires_at = (datetime.now() + timedelta(seconds=expires_delta_seconds)).isoformat()
    state = {
        "created_at": datetime.now().isoformat(),
        "expires_at": expires_at,
        "trades": [{"symbol": "AAPL", "side": "buy", "qty": 5.0}],
        "status": "pending",
    }
    pending_file = tmp_path / "approval_pending.json"
    pending_file.write_text(json.dumps(state))
    return state


# ---------------------------------------------------------------------------
# Test 1: approved path
# ---------------------------------------------------------------------------

def test_wait_for_approval_async_approved(tmp_path):
    """
    Mock check_approval_status to return 'pending' once, then 'approved'.
    Assert status == 'approved' and APPROVAL_PENDING_PATH is cleaned up.
    """
    from ascent.execution import approval_server

    trades = [{"symbol": "AAPL", "side": "buy", "qty": 5.0}]
    call_count = {"n": 0}

    def mock_status():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "pending"
        return "approved"

    with patch.object(approval_server, "APPROVAL_PENDING_PATH", tmp_path / "approval_pending.json"), \
         patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"), \
         patch.object(approval_server, "check_approval_status", side_effect=mock_status):
        result = approval_server.wait_for_approval_async(
            trades, timeout_seconds=90, poll_interval=1
        )

    assert result.status == "approved", f"Expected 'approved', got '{result.status}'"
    assert not (tmp_path / "approval_pending.json").exists(), \
        "APPROVAL_PENDING_PATH should be cleaned up after approval"


# ---------------------------------------------------------------------------
# Test 2: rejected path
# ---------------------------------------------------------------------------

def test_wait_for_approval_async_rejected(tmp_path):
    """
    Mock check_approval_status to return 'rejected' immediately.
    Assert status == 'rejected' and file cleaned up.
    """
    from ascent.execution import approval_server

    trades = [{"symbol": "MSFT", "side": "sell", "qty": 3.0}]

    with patch.object(approval_server, "APPROVAL_PENDING_PATH", tmp_path / "approval_pending.json"), \
         patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"), \
         patch.object(approval_server, "check_approval_status", return_value="rejected"):
        result = approval_server.wait_for_approval_async(
            trades, timeout_seconds=10, poll_interval=1
        )

    assert result.status == "rejected", f"Expected 'rejected', got '{result.status}'"
    assert not (tmp_path / "approval_pending.json").exists(), \
        "APPROVAL_PENDING_PATH should be cleaned up after rejection"


# ---------------------------------------------------------------------------
# Test 3: timeout path
# ---------------------------------------------------------------------------

def test_wait_for_approval_async_timeout(tmp_path):
    """
    Mock check_approval_status to always return 'pending'.
    With timeout_seconds=2 and poll_interval=1, result should be 'timeout'.
    """
    from ascent.execution import approval_server

    trades = [{"symbol": "GOOG", "side": "buy", "qty": 1.0}]

    with patch.object(approval_server, "APPROVAL_PENDING_PATH", tmp_path / "approval_pending.json"), \
         patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"), \
         patch.object(approval_server, "check_approval_status", return_value="pending"):
        result = approval_server.wait_for_approval_async(
            trades, timeout_seconds=2, poll_interval=1
        )

    assert result.status == "timeout", f"Expected 'timeout', got '{result.status}'"


# ---------------------------------------------------------------------------
# Test 4: resume expired path
# ---------------------------------------------------------------------------

def test_wait_for_approval_async_resume_expired(tmp_path):
    """
    Call with resume=True and a pending dict where expires_at is in the past.
    Should return status='expired' immediately without blocking.
    """
    from ascent.execution import approval_server

    pending = {
        "created_at": (datetime.now() - timedelta(hours=1)).isoformat(),
        "expires_at": (datetime.now() - timedelta(seconds=10)).isoformat(),
        "trades": [{"symbol": "TLT", "side": "buy", "qty": 2.0}],
        "status": "pending",
    }
    pending_file = tmp_path / "approval_pending.json"
    pending_file.write_text(json.dumps(pending))

    start = time.time()
    with patch.object(approval_server, "APPROVAL_PENDING_PATH", pending_file), \
         patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"):
        result = approval_server.wait_for_approval_async(
            pending_trades=[], resume=True, pending=pending
        )
    elapsed = time.time() - start

    assert result.status == "expired", f"Expected 'expired', got '{result.status}'"
    # Should return nearly instantly — not block for poll_interval
    assert elapsed < 5.0, f"Expired resume should return immediately, took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Test 5: resume approved path
# ---------------------------------------------------------------------------

def test_wait_for_approval_async_resume_approved(tmp_path):
    """
    Call with resume=True and pending with future expires_at.
    Mock check_approval_status → 'approved'. Assert status == 'approved'.
    """
    from ascent.execution import approval_server

    pending = {
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(minutes=10)).isoformat(),
        "trades": [{"symbol": "GLD", "side": "buy", "qty": 4.0}],
        "status": "pending",
    }
    pending_file = tmp_path / "approval_pending.json"
    pending_file.write_text(json.dumps(pending))

    with patch.object(approval_server, "APPROVAL_PENDING_PATH", pending_file), \
         patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"), \
         patch.object(approval_server, "check_approval_status", return_value="approved"):
        result = approval_server.wait_for_approval_async(
            pending_trades=[], resume=True, pending=pending,
            timeout_seconds=30, poll_interval=1
        )

    assert result.status == "approved", f"Expected 'approved', got '{result.status}'"


# ---------------------------------------------------------------------------
# Test 6: state file written before blocking
# ---------------------------------------------------------------------------

def test_approval_pending_json_written_before_wait(tmp_path):
    """
    APPROVAL_PENDING_PATH must exist immediately after wait_for_approval_async is called
    (before it returns), confirming state is written before blocking.
    """
    from ascent.execution import approval_server

    trades = [{"symbol": "NVDA", "side": "buy", "qty": 10.0}]
    pending_path = tmp_path / "approval_pending.json"
    file_existed_before_return = {"value": False}
    result_holder = {}

    def run_async():
        with patch.object(approval_server, "APPROVAL_PENDING_PATH", pending_path), \
             patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"), \
             patch.object(approval_server, "check_approval_status", return_value="pending"):
            r = approval_server.wait_for_approval_async(
                trades, timeout_seconds=2, poll_interval=1
            )
        result_holder["result"] = r

    t = threading.Thread(target=run_async)
    t.start()

    # Poll for up to 2 seconds to confirm file is written before function returns
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if pending_path.exists():
            file_existed_before_return["value"] = True
            break
        time.sleep(0.05)

    t.join(timeout=10)

    assert file_existed_before_return["value"], \
        "approval_pending.json must be written before wait_for_approval_async returns"


# ---------------------------------------------------------------------------
# Test 7: ApprovalResult dataclass importable with correct fields
# ---------------------------------------------------------------------------

def test_approval_result_importable():
    """ApprovalResult must be importable with status and trades fields."""
    from ascent.execution.approval_server import ApprovalResult

    r = ApprovalResult(status="approved", trades=[{"symbol": "AAPL"}])
    assert r.status == "approved"
    assert r.trades == [{"symbol": "AAPL"}]


# ---------------------------------------------------------------------------
# Test 8: check_approval_status raises — thread recovers and eventually succeeds
# ---------------------------------------------------------------------------

def test_wait_for_approval_async_check_status_raises(tmp_path):
    """
    Mock check_approval_status to raise OSError twice, then return 'approved'.
    Assert the function returns status='approved' without raising, confirming
    the _watcher thread survives transient exceptions.
    """
    from ascent.execution import approval_server

    trades = [{"symbol": "TSLA", "side": "buy", "qty": 2.0}]
    call_count = {"n": 0}

    def mock_status():
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise OSError("simulated file read error")
        return "approved"

    with patch.object(approval_server, "APPROVAL_PENDING_PATH", tmp_path / "approval_pending.json"), \
         patch.object(approval_server, "HEARTBEAT_LOG_PATH", tmp_path / "heartbeat.jsonl"), \
         patch.object(approval_server, "check_approval_status", side_effect=mock_status):
        result = approval_server.wait_for_approval_async(
            trades, timeout_seconds=30, poll_interval=1
        )

    assert result.status == "approved", f"Expected 'approved', got '{result.status}'"
    assert call_count["n"] >= 3, f"Expected at least 3 calls, got {call_count['n']}"
    assert not (tmp_path / "approval_pending.json").exists(), \
        "APPROVAL_PENDING_PATH should be cleaned up after approved status"


# ===========================================================================
# Fix 2b: Almgren-Chriss cost model
# ===========================================================================

import math
import json as _json
from unittest.mock import patch as _patch


# ---------------------------------------------------------------------------
# Test 9: normal small order — no flag
# ---------------------------------------------------------------------------

def test_cost_estimate_normal():
    """
    Trade $1_000 against ADV of 2_100_000_000 / 21 = 100_000_000.
    Participation = 1_000 / 100_000_000 = 0.001% → no flag.
    """
    from ascent.execution.cost_model import estimate

    features = {
        "dollar_vol_21d": {"AAPL": 2_100_000_000},
        "vol_21d": {"AAPL": 0.20},
    }
    est = estimate("AAPL", 1_000, features)

    assert est.flag is None, f"Expected no flag, got {est.flag}"
    assert est.total_one_way_bps > 0, "total_one_way_bps must be positive"
    expected_participation = 1_000 / (2_100_000_000 / 21)
    assert abs(est.participation_rate - expected_participation) < 1e-10, (
        f"Expected participation ~{expected_participation}, got {est.participation_rate}"
    )


# ---------------------------------------------------------------------------
# Test 10: high-impact order
# ---------------------------------------------------------------------------

def test_cost_estimate_high_impact():
    """
    Trade $100_000 against ADV of 500_000/21 ≈ 23_810.
    Participation ≈ 4.2 = 420% > max 10% ADV → HIGH_IMPACT.
    """
    from ascent.execution.cost_model import estimate

    features = {"dollar_vol_21d": {"AAPL": 500_000}}
    est = estimate("AAPL", 100_000, features)

    assert est.flag == "HIGH_IMPACT", f"Expected HIGH_IMPACT, got {est.flag}"


# ---------------------------------------------------------------------------
# Test 11: split recommended
# ---------------------------------------------------------------------------

def test_cost_estimate_split_recommended():
    """
    dollar_vol_21d = 10_000_000, trade $35_000.
    V_daily ≈ 476_190, participation ≈ 7.35% → SPLIT_RECOMMENDED.
    """
    from ascent.execution.cost_model import estimate

    features = {
        "dollar_vol_21d": {"AAPL": 10_000_000},
        "vol_21d": {"AAPL": 0.20},
    }
    est = estimate("AAPL", 35_000, features)

    assert est.flag == "SPLIT_RECOMMENDED", f"Expected SPLIT_RECOMMENDED, got {est.flag}"


# ---------------------------------------------------------------------------
# Test 12: IMPACT_UNKNOWN — missing volume
# ---------------------------------------------------------------------------

def test_cost_estimate_impact_unknown_missing_volume():
    """Empty features → IMPACT_UNKNOWN, all impact fields NaN."""
    from ascent.execution.cost_model import estimate

    est = estimate("AAPL", 100_000, {})

    assert est.flag == "IMPACT_UNKNOWN", f"Expected IMPACT_UNKNOWN, got {est.flag}"
    assert math.isnan(est.temporary_impact_bps)
    assert math.isnan(est.permanent_impact_bps)
    assert math.isnan(est.total_one_way_bps)
    assert math.isnan(est.participation_rate)


# ---------------------------------------------------------------------------
# Test 13: IMPACT_UNKNOWN — zero volume
# ---------------------------------------------------------------------------

def test_cost_estimate_impact_unknown_zero_volume():
    """dollar_vol_21d=0 → IMPACT_UNKNOWN."""
    from ascent.execution.cost_model import estimate

    features = {"dollar_vol_21d": {"AAPL": 0}}
    est = estimate("AAPL", 100_000, features)

    assert est.flag == "IMPACT_UNKNOWN", f"Expected IMPACT_UNKNOWN, got {est.flag}"


# ---------------------------------------------------------------------------
# Test 14: apply_cost_filter blocks HIGH_IMPACT
# ---------------------------------------------------------------------------

def test_apply_cost_filter_blocks_high_impact():
    """HIGH_IMPACT orders must be removed from the returned list."""
    from ascent.execution.order_engine import Order
    from ascent.execution.cost_model import apply_cost_filter

    order = Order(
        symbol="AAPL",
        side="buy",
        target_weight=0.10,
        current_weight=0.0,
        weight_delta=0.10,
        dollar_amount=500_000,
        estimated_shares=300,
    )
    features = {"dollar_vol_21d": {"AAPL": 500_000}}  # very low ADV → HIGH_IMPACT

    filtered, estimates, total_cost = apply_cost_filter([order], features, portfolio_value=5_000_000)

    assert len(filtered) == 0, "HIGH_IMPACT order must be blocked"
    assert estimates[0].flag == "HIGH_IMPACT"


# ---------------------------------------------------------------------------
# Test 15: apply_cost_filter passes small order
# ---------------------------------------------------------------------------

def test_apply_cost_filter_passes_small_order():
    """Small participation order must survive the filter and produce positive cost."""
    from ascent.execution.order_engine import Order
    from ascent.execution.cost_model import apply_cost_filter

    order = Order(
        symbol="AAPL",
        side="buy",
        target_weight=0.05,
        current_weight=0.0,
        weight_delta=0.05,
        dollar_amount=50_000,
        estimated_shares=30,
    )
    features = {
        "dollar_vol_21d": {"AAPL": 100_000_000},
        "vol_21d": {"AAPL": 0.20},
    }

    filtered, estimates, total_cost = apply_cost_filter([order], features, portfolio_value=1_000_000)

    assert len(filtered) == 1, "Small order must not be blocked"
    assert total_cost > 0, "total_cost_bps must be positive"


# ---------------------------------------------------------------------------
# Test 16: compute_orders with features applies cost filter
# ---------------------------------------------------------------------------

def test_compute_orders_with_features_applies_cost_filter():
    """
    Pass features with very low dollar_vol_21d for a symbol so its order is HIGH_IMPACT.
    Assert that symbol is absent from the returned orders list.
    """
    import pandas as pd
    from ascent.execution.order_engine import compute_orders

    target_weights = pd.Series({"AAPL": 0.10, "MSFT": 0.10})
    current_positions = pd.DataFrame(columns=["symbol", "weight", "current_price"])

    # AAPL will be HIGH_IMPACT (tiny ADV), MSFT has no volume data (IMPACT_UNKNOWN)
    features = {
        "dollar_vol_21d": {"AAPL": 100_000},  # very small → HIGH_IMPACT at $100k trade
        "vol_21d": {"AAPL": 0.20},
    }

    # Need a price lookup — patch _get_approx_price to return 150 for all symbols
    from unittest.mock import patch
    with patch("ascent.execution.order_engine._get_approx_price", return_value=150.0):
        orders, diff_df = compute_orders(
            target_weights,
            current_positions,
            portfolio_value=1_000_000,
            features=features,
        )

    symbols = [o.symbol for o in orders]
    assert "AAPL" not in symbols, "HIGH_IMPACT order for AAPL must be blocked"
    # MSFT should survive (IMPACT_UNKNOWN → warn only, don't block)
    assert "MSFT" in symbols, "MSFT (IMPACT_UNKNOWN) must not be blocked"


# ---------------------------------------------------------------------------
# Test 17: load_params falls back to defaults when file missing
# ---------------------------------------------------------------------------

def test_load_params_falls_back_to_defaults(tmp_path):
    """Nonexistent PARAMS_PATH → CostModelParams() with default values."""
    import ascent.execution.cost_model as cm
    from ascent.execution.cost_model import CostModelParams

    missing_path = tmp_path / "nonexistent.json"
    with _patch.object(cm, "PARAMS_PATH", missing_path):
        params = cm.load_params()

    defaults = CostModelParams()
    assert params.eta == defaults.eta
    assert params.gamma == defaults.gamma
    assert params.spread_bps == defaults.spread_bps


# ---------------------------------------------------------------------------
# Test 18: load_params reads from file
# ---------------------------------------------------------------------------

def test_load_params_reads_from_file(tmp_path):
    """JSON with eta=0.15, gamma=0.07 → loaded correctly."""
    import ascent.execution.cost_model as cm

    params_file = tmp_path / "cost_model_params.json"
    params_file.write_text(_json.dumps({"eta": 0.15, "gamma": 0.07}))

    with _patch.object(cm, "PARAMS_PATH", params_file):
        params = cm.load_params()

    assert params.eta == 0.15
    assert params.gamma == 0.07


# ---------------------------------------------------------------------------
# Test 19: NaN-skipping in weighted average
# ---------------------------------------------------------------------------

def test_apply_cost_filter_nan_skipped_in_weighted_average(tmp_path):
    """Orders with IMPACT_UNKNOWN (NaN cost) don't contaminate the weighted average."""
    from ascent.execution.order_engine import Order
    from ascent.execution.cost_model import apply_cost_filter

    # One order with known volume (known cost), one with unknown volume
    orders = [
        Order("AAPL", "buy", 0.10, 0.00, 0.10, 100_000, 500),
        Order("XYZ",  "buy", 0.05, 0.00, 0.05,  50_000, 100),
    ]
    features = {
        "dollar_vol_21d": {"AAPL": 2_100_000_000},  # very large — tiny participation
        # XYZ has no volume data → IMPACT_UNKNOWN
    }
    filtered, estimates, total_cost_bps = apply_cost_filter(orders, features, 1_000_000)

    assert len(filtered) == 2, "Both orders should be kept (neither is HIGH_IMPACT)"
    aapl_est = next(e for e in estimates if e.symbol == "AAPL")
    xyz_est  = next(e for e in estimates if e.symbol == "XYZ")
    assert not math.isnan(aapl_est.total_one_way_bps)
    assert math.isnan(xyz_est.total_one_way_bps)
    # total_cost_bps should be > 0 (from AAPL's known cost) even though XYZ is NaN
    assert total_cost_bps > 0, "NaN from XYZ should be skipped, not zero out the total"


# ---------------------------------------------------------------------------
# Test 20: All orders HIGH_IMPACT → empty return
# ---------------------------------------------------------------------------

def test_apply_cost_filter_all_high_impact_returns_empty(tmp_path):
    """When all orders are HIGH_IMPACT, filtered_orders is empty, cost is 0."""
    from ascent.execution.order_engine import Order
    from ascent.execution.cost_model import apply_cost_filter

    orders = [
        Order("AAPL", "buy", 0.10, 0.00, 0.10, 500_000, 2500),
    ]
    features = {
        "dollar_vol_21d": {"AAPL": 100_000},  # tiny volume → very high participation
    }
    filtered, estimates, total_cost_bps = apply_cost_filter(orders, features, 5_000_000)

    assert filtered == [], "All HIGH_IMPACT orders should be blocked"
    assert total_cost_bps == 0.0, "No non-blocked orders → cost is 0"
    assert estimates[0].flag == "HIGH_IMPACT"
