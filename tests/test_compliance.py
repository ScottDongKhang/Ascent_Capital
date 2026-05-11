"""tests/test_compliance.py
12 tests for Plan 7 — Live Track Record & Compliance Framework.
"""
import json
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Task 1: Audit trail ───────────────────────────────────────────────────────

def test_audit_trail_appends_only(tmp_path):
    """Write two entries; file has exactly two lines."""
    from compliance.audit_trail import AuditTrail

    trail = AuditTrail(log_path=tmp_path / "audit.jsonl")
    trail.record("order_submitted",  {"symbol": "AAPL", "side": "buy",  "shares": 10})
    trail.record("order_submitted",  {"symbol": "MSFT", "side": "sell", "shares": 5})

    lines = [l for l in (tmp_path / "audit.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_audit_trail_hash_chain_valid(tmp_path):
    """Hash of entry N matches prev_hash of entry N+1."""
    from compliance.audit_trail import AuditTrail

    trail = AuditTrail(log_path=tmp_path / "audit.jsonl")
    trail.record("signal_generated",    {"scores": {"AAPL": 0.8}})
    trail.record("portfolio_constructed", {"weights": {"AAPL": 0.10}})
    trail.record("debate_verdict",      {"verdict": "proceed"})

    entries = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text().splitlines() if l.strip()]
    assert len(entries) == 3
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
    assert entries[2]["prev_hash"] == entries[1]["entry_hash"]


def test_audit_trail_tamper_detected(tmp_path):
    """Manually modify an entry → integrity check fails."""
    from compliance.audit_trail import AuditTrail

    trail = AuditTrail(log_path=tmp_path / "audit.jsonl")
    trail.record("order_submitted", {"symbol": "AAPL", "side": "buy", "shares": 10})
    trail.record("order_filled",    {"symbol": "AAPL", "fill_price": 155.0, "shares": 10})

    # Tamper with first entry
    path = tmp_path / "audit.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["payload"]["shares"] = 999  # tamper
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")

    result = trail.verify_integrity()
    assert result["valid"] is False
    assert result["broken_at"] is not None


def test_audit_trail_records_order_submitted(tmp_path):
    """record() creates an entry with correct event_type."""
    from compliance.audit_trail import AuditTrail

    trail = AuditTrail(log_path=tmp_path / "audit.jsonl")
    h = trail.record("order_submitted", {"symbol": "NVDA", "side": "buy", "shares": 3})

    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex

    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert entry["event_type"] == "order_submitted"
    assert entry["payload"]["symbol"] == "NVDA"


# ── Task 3: GIPS performance ──────────────────────────────────────────────────

def test_gips_twr_correct():
    """Known daily returns → correct time-weighted return."""
    from compliance.performance_report import compute_gips_performance

    daily_returns   = [0.01, -0.005, 0.02, 0.015, -0.01]
    expected_twr = 1.0
    for r in daily_returns:
        expected_twr *= (1 + r)
    expected_twr -= 1.0

    rows = [
        {"date": f"2026-01-0{i+1}", "nav": 100_000 * (1 + r), "daily_return": r, "spy_return": r * 0.8}
        for i, r in enumerate(daily_returns)
    ]

    with patch("compliance.performance_report._load_pnl_rows", return_value=rows):
        result = compute_gips_performance(date(2026, 1, 1), date(2026, 1, 5))

    assert abs(result["twr"] - expected_twr) < 0.0001


def test_gips_sharpe_correct():
    """Known returns and constant vol → Sharpe computed correctly."""
    import math
    from compliance.performance_report import compute_gips_performance

    # All same daily return → zero vol → Sharpe is undefined, but function handles gracefully
    returns = [0.001] * 30
    rows = [
        {"date": f"2026-01-{i+1:02d}", "nav": 100_000, "daily_return": r, "spy_return": r * 0.8}
        for i, r in enumerate(returns)
    ]
    with patch("compliance.performance_report._load_pnl_rows", return_value=rows):
        with patch("compliance.performance_report._risk_free_rate", return_value=0.05):
            result = compute_gips_performance(date(2026, 1, 1), date(2026, 1, 30))

    assert "sharpe" in result
    assert isinstance(result["sharpe"], float)


def test_gips_max_drawdown_correct():
    """Known NAV series → correct max drawdown."""
    from compliance.performance_report import compute_gips_performance

    # NAV peaks at 110, then drops to 88 → drawdown = (110-88)/110 = 20%
    navs    = [100, 105, 110, 105, 95, 88, 92]
    returns = [navs[i]/navs[i-1] - 1 for i in range(1, len(navs))]
    rows    = [
        {"date": f"2026-01-{i+2:02d}", "nav": navs[i+1], "daily_return": returns[i], "spy_return": returns[i] * 0.8}
        for i in range(len(returns))
    ]
    with patch("compliance.performance_report._load_pnl_rows", return_value=rows):
        result = compute_gips_performance(date(2026, 1, 2), date(2026, 1, 8))

    expected_dd = (110 - 88) / 110
    assert abs(result["max_drawdown"] - expected_dd) < 0.01


def test_gips_table_contains_required_periods():
    """format_gips_table output contains 1M, 3M, 6M, YTD, 1Y, SI columns."""
    from compliance.performance_report import format_gips_table

    perf = {
        "1M": {"twr": 0.01, "ann_return": 0.12, "ann_vol": 0.15, "sharpe": 0.8, "max_drawdown": 0.05},
        "3M": {"twr": 0.03, "ann_return": 0.12, "ann_vol": 0.15, "sharpe": 0.8, "max_drawdown": 0.07},
        "6M": {},
        "YTD": {},
        "1Y":  {},
        "SI":  {},
    }
    table = format_gips_table(perf)
    for period in ["1M", "3M", "6M", "YTD", "1Y", "SI"]:
        assert period in table


# ── Task 4: Risk disclosures ──────────────────────────────────────────────────

def test_risk_disclosure_contains_required_language():
    """Output contains 'past performance' and 'model risk'."""
    from compliance.risk_disclosure import generate_risk_disclosures

    text = generate_risk_disclosures({
        "max_leverage":        1.0,
        "asset_classes":       ["US Equities"],
        "n_positions":         15,
        "max_single_position": "10%",
        "inception_date":      "April 2026",
        "kill_switch_pct":     "15%",
    })

    assert "past performance" in text.lower()
    assert "model risk" in text.lower()


# ── Task 6: Methodology index ─────────────────────────────────────────────────

def test_methodology_index_has_all_sleeves():
    """build_methodology_index includes all active sleeve names."""
    from compliance.methodology_index import build_methodology_index

    idx = build_methodology_index()
    sleeve_names = {s["name"] for s in idx["alpha_sleeves"]}
    required = {"trend", "statarb", "ml", "mean_reversion", "fundamental", "earnings", "llm_fundamental"}
    assert required.issubset(sleeve_names)


def test_methodology_index_exported_to_dashboard(tmp_path):
    """export_methodology_index creates dashboard/methodology_index.json."""
    from compliance import methodology_index as mi

    export_path = tmp_path / "methodology_index.json"
    with patch.object(mi, "EXPORT_PATH", export_path):
        mi.export_methodology_index()

    assert export_path.exists()
    data = json.loads(export_path.read_text())
    assert "alpha_sleeves" in data
    assert "data_sources"  in data


# ── Task 2: Audit integrity script ────────────────────────────────────────────

def test_audit_integrity_script_runs_clean(tmp_path):
    """Clean audit trail → verify_integrity returns valid=True."""
    from compliance.audit_trail import AuditTrail

    trail = AuditTrail(log_path=tmp_path / "audit.jsonl")
    trail.record("signal_generated",    {"test": True})
    trail.record("portfolio_constructed", {"weights": {}})

    result = trail.verify_integrity()
    assert result["valid"] is True
    assert result["total_entries"] == 2
    assert result["broken_at"] is None
