# tests/data/test_cboe_options_bounds.py
"""
update_options_cache must be bounded on time and on consecutive failures.

Regression test for a live incident on 2026-07-27: a degraded provider returned
{'unavailable': True} after ~14s per symbol. Looping the ~900-symbol universe
was ~3.5h of no-op work, which blocked the entire trading run before the agents
started. The call site is wrapped in try/except, but slowness raises nothing, so
the bounds have to live inside the function.
"""
import time

import pytest

import ascent.data.ingest.cboe_options as cboe


@pytest.fixture
def symbols():
    return [f"S{i}" for i in range(900)]


def test_circuit_breaker_stops_after_consecutive_failures(monkeypatch, tmp_path, symbols):
    """A provider returning nothing must not be asked 900 times."""
    calls = []
    monkeypatch.setattr(cboe, "fetch_cboe_options_row",
                        lambda s, d: calls.append(s) or None)

    n = cboe.update_options_cache(symbols, "2026-07-27",
                                  cache_path=tmp_path / "opts.parquet",
                                  max_consecutive_failures=8)

    assert n == 0
    assert len(calls) == 8, f"expected 8 attempts before tripping, got {len(calls)}"


def test_time_budget_bounds_the_loop_and_keeps_partial_results(monkeypatch, tmp_path, symbols):
    """Slow-but-successful calls stop on wall clock; rows already fetched are kept."""
    calls = []

    def slow(sym, date):
        calls.append(sym)
        time.sleep(0.02)
        return {"symbol": sym, "date": date, "put_call_ratio": 1.0, "atm_iv": 0.2,
                "iv_skew": 0.0, "iv_rank_52w": 50, "source": "cboe"}

    monkeypatch.setattr(cboe, "fetch_cboe_options_row", slow)

    started = time.monotonic()
    n = cboe.update_options_cache(symbols, "2026-07-27",
                                  cache_path=tmp_path / "opts.parquet",
                                  time_budget_s=0.3)
    elapsed = time.monotonic() - started

    assert len(calls) < len(symbols), "time budget did not bound the loop"
    assert elapsed < 10, f"loop ran {elapsed:.1f}s despite a 0.3s budget"
    assert n > 0, "partial results collected before the cutoff must still be written"
    assert (tmp_path / "opts.parquet").exists()


def test_a_raising_provider_counts_as_a_failure_not_a_crash(monkeypatch, tmp_path, symbols):
    """An exception from one symbol must not abort the whole ingest."""
    def boom(sym, date):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(cboe, "fetch_cboe_options_row", boom)

    n = cboe.update_options_cache(symbols, "2026-07-27",
                                  cache_path=tmp_path / "opts.parquet",
                                  max_consecutive_failures=3)
    assert n == 0  # trips the breaker, returns cleanly rather than propagating


def test_healthy_provider_is_not_cut_short(monkeypatch, tmp_path):
    """The bounds must not interfere with a normal, working fetch."""
    monkeypatch.setattr(cboe, "fetch_cboe_options_row",
                        lambda s, d: {"symbol": s, "date": d, "put_call_ratio": 1.0,
                                      "atm_iv": 0.2, "iv_skew": 0.0,
                                      "iv_rank_52w": 50, "source": "cboe"})

    n = cboe.update_options_cache(["AAPL", "MSFT", "NVDA"], "2026-07-27",
                                  cache_path=tmp_path / "opts.parquet")
    assert n == 3
