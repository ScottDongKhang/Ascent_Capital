"""Unit tests for compliance.data_integrity.

Standalone module, not yet wired into the live pipeline. These tests build
real parquet fixture caches under a temp `data_cache/` directory and exercise
`check_cache()` end to end -- including through the real
`scripts.reconcile_numbers.data_integrity()` call it wraps -- rather than
mocking that function out.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from compliance.data_integrity import (
    ACTION_FALLBACK,
    ACTION_PASS,
    ACTION_WARN,
    DEFAULT_STALE_DAYS,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_NONE,
    STATUS_CLEAN,
    STATUS_DIRTY,
    STATUS_MISSING,
    STATUS_STALE,
    DataIntegrityOfficer,
    check_cache,
)

AS_OF = date(2026, 8, 19)


def _write_cache(data_dir, cache_name: str, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(data_dir / f"{cache_name}.parquet", index=False)


def _clean_rows(as_of: date, n_days: int = 5):
    """`n_days` consecutive midnight-stamped trading days, one symbol/day, no dupes."""
    rows = []
    d = as_of - timedelta(days=n_days - 1)
    for i in range(n_days):
        day = d + timedelta(days=i)
        rows.append({
            "date": datetime(day.year, day.month, day.day, 0, 0),
            "symbol": "AAPL",
            "close": 100.0 + i,
            "source": "yahoo_live",
        })
    return rows


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data_cache"
    d.mkdir()
    # ascent.data.store.parquet.DATA_DIR is what check_cache() reads to build
    # the path it points reconcile_numbers.PRICES at.
    monkeypatch.setattr("compliance.data_integrity.DATA_DIR", d)
    return d


class TestCleanCache:
    def test_clean_cache_passes(self, data_dir):
        _write_cache(data_dir, "prices_live", _clean_rows(AS_OF, n_days=5))

        result = check_cache("prices_live", as_of=AS_OF)

        assert result.cache == "prices_live"
        assert result.status == STATUS_CLEAN
        assert result.duplicate_rows == 0
        assert result.phantom_rows == 0
        assert result.action == ACTION_PASS
        assert result.severity == SEVERITY_NONE


class TestDuplicateRows:
    def test_injected_duplicate_row_triggers_fallback(self, data_dir, capsys):
        rows = _clean_rows(AS_OF, n_days=5)
        # Inject an exact duplicate of the last (symbol, calendar-day) pair --
        # same signature as the recurring prices_live incident CLAUDE.md
        # documents.
        rows.append(dict(rows[-1]))
        _write_cache(data_dir, "prices_live", rows)

        result = check_cache("prices_live", as_of=AS_OF)

        assert result.status == STATUS_DIRTY
        assert result.duplicate_rows >= 1
        assert result.action == ACTION_FALLBACK
        assert result.severity == SEVERITY_HIGH
        # Threshold requires this be printed, not just returned.
        out = capsys.readouterr().out
        assert "HIGH" in out
        assert "prices_live" in out


class TestPhantomRows:
    def test_injected_phantom_row_triggers_fallback(self, data_dir, capsys):
        rows = _clean_rows(AS_OF, n_days=5)
        # Phantom-row corruption signature: a bar stamped at a non-midnight
        # time of day (e.g. a late hub fetch at 19:00/20:00) on a day/symbol
        # combination that does not collide with any existing midnight row,
        # so the plain duplicate check above stays at zero.
        phantom_day = AS_OF - timedelta(days=10)
        rows.append({
            "date": datetime(phantom_day.year, phantom_day.month, phantom_day.day, 19, 0),
            "symbol": "AAPL",
            "close": 101.5,
            "source": "yfinance_hub",
        })
        _write_cache(data_dir, "prices_live", rows)

        result = check_cache("prices_live", as_of=AS_OF)

        assert result.status == STATUS_DIRTY
        assert result.phantom_rows >= 1
        assert result.action == ACTION_FALLBACK
        assert result.severity == SEVERITY_HIGH
        out = capsys.readouterr().out
        assert "HIGH" in out


class TestStaleness:
    def test_stale_cache_warns_without_fallback(self, data_dir):
        # Latest row is older than DEFAULT_STALE_DAYS relative to as_of, but
        # otherwise clean (no dup/phantom rows).
        stale_as_of = AS_OF
        rows = _clean_rows(AS_OF - timedelta(days=DEFAULT_STALE_DAYS + 5), n_days=3)
        _write_cache(data_dir, "prices_macro", rows)

        result = check_cache("prices_macro", as_of=stale_as_of)

        assert result.status == STATUS_STALE
        assert result.duplicate_rows == 0
        assert result.phantom_rows == 0
        assert result.action == ACTION_WARN
        assert result.severity == SEVERITY_MEDIUM


class TestMissingCache:
    def test_missing_cache_falls_back(self, data_dir):
        result = check_cache("prices_alternatives", as_of=AS_OF)

        assert result.status == STATUS_MISSING
        assert result.action == ACTION_FALLBACK
        assert result.severity == SEVERITY_HIGH


class TestDataIntegrityOfficerClass:
    def test_wraps_check_cache(self, data_dir):
        _write_cache(data_dir, "prices_live", _clean_rows(AS_OF, n_days=5))
        officer = DataIntegrityOfficer(stale_days=DEFAULT_STALE_DAYS)

        result = officer.check_cache("prices_live", as_of=AS_OF)

        assert result.status == STATUS_CLEAN
        assert result.action == ACTION_PASS
