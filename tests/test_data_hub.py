"""
tests/test_data_hub.py
TDD tests for ascent/data/hub.py — centralized data ingestion hub.

Run: .venv/bin/python -m pytest tests/test_data_hub.py -v
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from ascent.config.settings import UniverseConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_price_df(symbols=("SPY", "AAPL"), n_days=5) -> pd.DataFrame:
    """Minimal valid price DataFrame matching fetch_universe_daily schema."""
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    for sym in symbols:
        for dt in dates:
            rows.append({
                "symbol": sym, "date": dt,
                "close": 100.0, "high": 101.0, "low": 99.0,
                "open": 100.0, "volume": 1_000_000,
                "adj_close": 100.0, "source": "yahoo_hub",
            })
    return pd.DataFrame(rows)


def _make_manifest(age_hours: float = 0.5, status: str = "ok") -> dict:
    fetched_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return {
        "fetched_at": fetched_at.isoformat(),
        "status": status,
        "symbols_requested": 10,
        "symbols_fetched": 10,
        "prices_rows": 500,
        "macro_series": 8,
        "start_date": "2020-01-02",
        "end_date": "2026-01-14",
    }


# ── collect_all_symbols ────────────────────────────────────────────────────────

def test_collect_all_symbols_includes_spy():
    from ascent.data.hub import collect_all_symbols
    syms = collect_all_symbols()
    assert "SPY" in syms


def test_collect_all_symbols_no_duplicates():
    from ascent.data.hub import collect_all_symbols
    syms = collect_all_symbols()
    assert isinstance(syms, set)  # sets have no duplicates by definition


def test_collect_all_symbols_includes_all_universes():
    from ascent.data.hub import collect_all_symbols
    cfg = UniverseConfig()
    syms = collect_all_symbols()

    # US equities
    for s in cfg.symbols[:5]:
        assert s in syms, f"{s} missing from hub symbol set"

    # Macro ETFs
    for s in cfg.macro_symbols:
        assert s in syms, f"{s} missing from hub symbol set"

    # International ETFs
    for s in cfg.international_symbols:
        assert s in syms, f"{s} missing from hub symbol set"

    # Alternatives
    for s in cfg.alternatives_symbols:
        assert s in syms, f"{s} missing from hub symbol set"


def test_collect_all_symbols_minimum_size():
    from ascent.data.hub import collect_all_symbols
    syms = collect_all_symbols()
    # 135 US + 12 macro + 12 intl + 10 alts + SPY — some overlap, but should exceed 150
    assert len(syms) >= 150


# ── hub_is_fresh ───────────────────────────────────────────────────────────────

def test_hub_is_fresh_no_manifest(tmp_path):
    from ascent.data import hub
    with patch.object(hub, "_MANIFEST_PATH", tmp_path / "hub_manifest.json"):
        assert hub.hub_is_fresh() is False


def test_hub_is_fresh_recent_manifest(tmp_path):
    from ascent.data import hub
    manifest_path = tmp_path / "hub_manifest.json"
    manifest_path.write_text(json.dumps(_make_manifest(age_hours=0.5)))
    with patch.object(hub, "_MANIFEST_PATH", manifest_path):
        assert hub.hub_is_fresh(max_age_hours=2.0) is True


def test_hub_is_fresh_stale_manifest(tmp_path):
    from ascent.data import hub
    manifest_path = tmp_path / "hub_manifest.json"
    manifest_path.write_text(json.dumps(_make_manifest(age_hours=3.0)))
    with patch.object(hub, "_MANIFEST_PATH", manifest_path):
        assert hub.hub_is_fresh(max_age_hours=2.0) is False


def test_hub_is_fresh_corrupt_manifest(tmp_path):
    from ascent.data import hub
    manifest_path = tmp_path / "hub_manifest.json"
    manifest_path.write_text("not-valid-json{{{")
    with patch.object(hub, "_MANIFEST_PATH", manifest_path):
        assert hub.hub_is_fresh() is False


# ── run_hub ────────────────────────────────────────────────────────────────────

def test_run_hub_returns_ok_status(tmp_path):
    from ascent.data import hub

    price_df = _make_price_df()

    with (
        patch.object(hub, "_MANIFEST_PATH", tmp_path / "hub_manifest.json"),
        patch.object(hub, "_fetch_all_prices", return_value=(price_df, 10, 2)),
        patch.object(hub, "save_parquet"),
        patch.object(hub, "normalize_prices", return_value=price_df),
        patch("ascent.data.hub.fetch_all_macro", return_value=pd.DataFrame()),
    ):
        result = hub.run_hub("2020-01-02", "2026-04-14")

    assert result["status"] == "ok"


def test_run_hub_writes_manifest(tmp_path):
    from ascent.data import hub

    price_df = _make_price_df()
    manifest_path = tmp_path / "hub_manifest.json"

    with (
        patch.object(hub, "_MANIFEST_PATH", manifest_path),
        patch.object(hub, "_fetch_all_prices", return_value=(price_df, 10, 2)),
        patch.object(hub, "save_parquet"),
        patch.object(hub, "normalize_prices", return_value=price_df),
        patch("ascent.data.hub.fetch_all_macro", return_value=pd.DataFrame()),
    ):
        hub.run_hub("2020-01-02", "2026-04-14")

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "ok"
    assert "fetched_at" in manifest
    assert manifest["symbols_requested"] == 10
    assert manifest["symbols_fetched"] == 2


def test_run_hub_skips_if_fresh(tmp_path):
    from ascent.data import hub

    manifest_path = tmp_path / "hub_manifest.json"
    cached = _make_manifest(age_hours=0.5)
    manifest_path.write_text(json.dumps(cached))

    with (
        patch.object(hub, "_MANIFEST_PATH", manifest_path),
        patch.object(hub, "_fetch_all_prices") as mock_fetch,
    ):
        result = hub.run_hub("2020-01-02", "2026-04-14")
        mock_fetch.assert_not_called()

    assert result["status"] == "ok"


def test_run_hub_force_bypasses_freshness(tmp_path):
    from ascent.data import hub

    price_df = _make_price_df()
    manifest_path = tmp_path / "hub_manifest.json"
    manifest_path.write_text(json.dumps(_make_manifest(age_hours=0.5)))

    with (
        patch.object(hub, "_MANIFEST_PATH", manifest_path),
        patch.object(hub, "_fetch_all_prices", return_value=(price_df, 10, 2)) as mock_fetch,
        patch.object(hub, "save_parquet"),
        patch.object(hub, "normalize_prices", return_value=price_df),
        patch("ascent.data.hub.fetch_all_macro", return_value=pd.DataFrame()),
    ):
        hub.run_hub("2020-01-02", "2026-04-14", force=True)
        mock_fetch.assert_called_once()


def test_run_hub_returns_failed_on_empty_prices(tmp_path):
    from ascent.data import hub

    with (
        patch.object(hub, "_MANIFEST_PATH", tmp_path / "hub_manifest.json"),
        patch.object(hub, "_fetch_all_prices", return_value=(pd.DataFrame(), 10, 0)),
        patch.object(hub, "normalize_prices", return_value=pd.DataFrame()),
    ):
        result = hub.run_hub("2020-01-02", "2026-04-14")

    assert result["status"] == "failed"
    assert "error" in result


def test_run_hub_macro_failure_does_not_abort(tmp_path):
    """FRED failure should not prevent prices from being written."""
    from ascent.data import hub

    price_df = _make_price_df()

    with (
        patch.object(hub, "_MANIFEST_PATH", tmp_path / "hub_manifest.json"),
        patch.object(hub, "_fetch_all_prices", return_value=(price_df, 10, 2)),
        patch.object(hub, "save_parquet"),
        patch.object(hub, "normalize_prices", return_value=price_df),
        patch("ascent.data.hub.fetch_all_macro", side_effect=RuntimeError("FRED down")),
    ):
        result = hub.run_hub("2020-01-02", "2026-04-14")

    # Prices succeeded — overall status still ok
    assert result["status"] == "ok"
    assert result.get("macro_series", 0) == 0


def test_run_hub_records_row_count(tmp_path):
    from ascent.data import hub

    price_df = _make_price_df(symbols=("SPY", "AAPL", "MSFT"), n_days=10)

    with (
        patch.object(hub, "_MANIFEST_PATH", tmp_path / "hub_manifest.json"),
        patch.object(hub, "_fetch_all_prices", return_value=(price_df, 3, 3)),
        patch.object(hub, "save_parquet"),
        patch.object(hub, "normalize_prices", return_value=price_df),
        patch("ascent.data.hub.fetch_all_macro", return_value=pd.DataFrame()),
    ):
        result = hub.run_hub("2020-01-02", "2026-04-14")

    assert result["prices_rows"] == len(price_df)


# ── _fetch_all_prices (unit) ───────────────────────────────────────────────────

def test_fetch_all_prices_returns_combined_df():
    from ascent.data.hub import _fetch_all_prices

    def mock_ticker_history(sym, start, end):
        return _make_price_df(symbols=[sym], n_days=3).drop(columns=["source"])

    with patch("ascent.data.hub._fetch_symbol", side_effect=lambda s, st, en: _make_price_df([s], 3)):
        df, n_req, n_fetched = _fetch_all_prices({"AAPL", "MSFT"}, "2026-01-01", "2026-01-10")

    assert not df.empty
    assert n_req == 2


def test_fetch_all_prices_handles_partial_failure():
    """Some symbols fail — should still return data for successful ones."""
    from ascent.data.hub import _fetch_all_prices

    def side_effect(sym, start, end):
        if sym == "FAIL":
            return None
        return _make_price_df([sym], 3)

    with patch("ascent.data.hub._fetch_symbol", side_effect=side_effect):
        df, n_req, n_fetched = _fetch_all_prices({"AAPL", "FAIL"}, "2026-01-01", "2026-01-10")

    assert not df.empty
    assert "FAIL" not in df["symbol"].unique()
