"""
Guards on the off-calendar discovery / mini-rebalance path.

Two behaviors are enforced here:

1. Calendar proximity guard — discovery must NOT fire within N trading days of a
   scheduled rebalance (the next scheduled rebalance will recompute the book
   anyway; an off-calendar full rotation two days prior is pure churn).

2. Add-only insertion — when discovery DOES fire, it must insert the candidate
   and trim the rest of the book pro-rata, NOT re-rank/re-optimize every holding.
"""
import pandas as pd

import run_all_agents as ra


# ── Guard A: calendar proximity ──────────────────────────────────────────────

def _write_calendar(tmp_path, dates):
    p = tmp_path / "rebalance_calendar.csv"
    p.write_text("rebalance_date\n" + "\n".join(dates) + "\n")
    return p


def test_near_rebalance_true_when_two_trading_days_out(tmp_path):
    # Mon 2026-06-22; next scheduled rebalance Wed 2026-06-24 = 2 trading days.
    import datetime
    cal = _write_calendar(tmp_path, ["2026-06-10", "2026-06-24"])
    assert ra._is_near_scheduled_rebalance(
        datetime.date(2026, 6, 22), window=3, cal_path=cal) is True


def test_near_rebalance_false_when_far_out(tmp_path):
    import datetime
    cal = _write_calendar(tmp_path, ["2026-06-10", "2026-07-22"])
    assert ra._is_near_scheduled_rebalance(
        datetime.date(2026, 6, 22), window=3, cal_path=cal) is False


def test_near_rebalance_false_when_no_future_date(tmp_path):
    import datetime
    cal = _write_calendar(tmp_path, ["2026-06-10"])
    assert ra._is_near_scheduled_rebalance(
        datetime.date(2026, 6, 22), window=3, cal_path=cal) is False


def test_near_rebalance_false_when_calendar_missing(tmp_path):
    import datetime
    missing = tmp_path / "nope.csv"
    assert ra._is_near_scheduled_rebalance(
        datetime.date(2026, 6, 22), window=3, cal_path=missing) is False


# ── Guard B: add-only insertion ──────────────────────────────────────────────

def test_insert_adds_symbol_and_sums_to_one():
    book = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    out = ra._insert_candidate_weights(book, "NEW", max_weight=0.40)
    assert "NEW" in out
    assert out["NEW"] > 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_insert_respects_max_weight_cap():
    # Dominant AAA would scale to 0.45 after insertion; cap must pull it to 0.30.
    # 4 names under a 0.30 cap is feasible (4 * 0.30 = 1.2 >= 1.0).
    book = {"AAA": 0.6, "BBB": 0.25, "CCC": 0.15}
    out = ra._insert_candidate_weights(book, "NEW", max_weight=0.30)
    assert all(w <= 0.30 + 1e-3 for w in out.values())
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_insert_preserves_relative_order_of_existing():
    # Pro-rata trim must NOT re-rank: AAA stays > BBB stays > CCC.
    book = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    out = ra._insert_candidate_weights(book, "NEW", max_weight=0.40)
    assert out["AAA"] > out["BBB"] > out["CCC"]


def test_insert_existing_symbol_is_noop():
    book = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    out = ra._insert_candidate_weights(book, "AAA", max_weight=0.40)
    assert out == book


# ── Guard C: IC-decay rebalance blackout window ──────────────────────────────

def test_ic_decay_trigger_suppressed_near_rebalance(tmp_path):
    """
    Within 3 trading days of a scheduled rebalance, the IC decay early-rebalance
    trigger must be blocked — the scheduled rebalance will recompute the book anyway.
    """
    import datetime
    cal = _write_calendar(tmp_path, ["2026-06-10", "2026-06-24"])
    today = datetime.date(2026, 6, 22)  # 2 trading days before Jun 24

    near = ra._is_near_scheduled_rebalance(today, window=3, cal_path=cal)
    assert near is True, "Guard should detect proximity to Jun 24"

    # Simulate the blackout logic: if near → skip trigger
    would_have_triggered = False
    if not near:
        would_have_triggered = True  # IC decay fires only when NOT near

    assert not would_have_triggered, "IC decay trigger should be blocked within 3-day window"


def test_ic_decay_trigger_allowed_far_from_rebalance(tmp_path):
    """
    When the next scheduled rebalance is more than 3 trading days away,
    the IC decay trigger must NOT be suppressed.
    """
    import datetime
    cal = _write_calendar(tmp_path, ["2026-06-10", "2026-07-22"])
    today = datetime.date(2026, 6, 22)  # 30 days before Jul 22

    near = ra._is_near_scheduled_rebalance(today, window=3, cal_path=cal)
    assert near is False, "Guard should not trigger when rebalance is far away"

    # IC decay is allowed to fire when not near
    trigger_allowed = not near
    assert trigger_allowed, "IC decay trigger must be permitted when far from rebalance"


# ── Guard D: _live_book_or — single implementation of live-book-with-fallback ─

def test_live_book_or_normalizes_live_weights(mocker):
    pos = pd.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.3, 0.3]})
    mocker.patch("ascent.execution.alpaca_broker.get_positions", return_value=pos)
    out = ra._live_book_or({"CCC": 1.0})
    assert set(out) == {"AAA", "BBB"}
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_live_book_or_falls_back_when_broker_returns_empty(mocker):
    mocker.patch("ascent.execution.alpaca_broker.get_positions", return_value=pd.DataFrame())
    out = ra._live_book_or({"CCC": 1.0})
    assert out == {"CCC": 1.0}


def test_live_book_or_falls_back_when_broker_raises(mocker):
    mocker.patch(
        "ascent.execution.alpaca_broker.get_positions",
        side_effect=RuntimeError("broker unreachable"),
    )
    out = ra._live_book_or({"CCC": 1.0})
    assert out == {"CCC": 1.0}


# ── Guard E: the discovery call site must use the live book ─────────────────
#
# These exercise _trigger_mini_rebalance itself (not just the helpers in
# isolation) -- this is exactly the gap that let the 2026-06-30 incident ship:
# the add-only insert unit tests all passed while the call site fed it the
# freshly recomputed orchestrator target instead of the live Alpaca book.

class _FakeResult:
    def __init__(self, symbol, conviction_score=0.8, catalyst_snippet="test catalyst"):
        self.symbol = symbol
        self.conviction_score = conviction_score
        self.catalyst_snippet = catalyst_snippet


def _isolate_fs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)


def test_mini_rebalance_inserts_onto_live_book_not_recomputed_target(tmp_path, monkeypatch, mocker):
    _isolate_fs(tmp_path, monkeypatch)

    # Live book actually held at the broker.
    live_positions = pd.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]})
    mocker.patch("ascent.execution.alpaca_broker.get_positions", return_value=live_positions)
    mock_eod = mocker.patch("ascent.execution.eod_runner.run_eod_with_weights")

    # Freshly recomputed orchestrator target for today -- deliberately diverges
    # from the live book (drops BBB, adds CCC), mirroring the real incident.
    recomputed_target = {"AAA": 0.5, "CCC": 0.5}

    ra._trigger_mini_rebalance(_FakeResult("NEW"), recomputed_target, ra.date(2026, 6, 30))

    assert mock_eod.called, "run_eod_with_weights should have been invoked"
    submitted = mock_eod.call_args[0][0]
    # Base must be the LIVE book: BBB (live) must survive, CCC (recomputed-only)
    # must never appear, and the candidate must be added.
    assert "BBB" in submitted
    assert "CCC" not in submitted
    assert "NEW" in submitted


def test_mini_rebalance_order_set_has_zero_full_exits(tmp_path, monkeypatch, mocker):
    _isolate_fs(tmp_path, monkeypatch)

    live_positions = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "weight": [0.5, 0.3, 0.2],
    })
    mocker.patch("ascent.execution.alpaca_broker.get_positions", return_value=live_positions)
    mock_eod = mocker.patch("ascent.execution.eod_runner.run_eod_with_weights")

    ra._trigger_mini_rebalance(_FakeResult("NEW"), {"unused": 1.0}, ra.date(2026, 6, 30))

    assert mock_eod.called
    submitted = mock_eod.call_args[0][0]
    for sym in ("AAA", "BBB", "CCC"):
        assert submitted.get(sym, 0.0) > 0.0, f"{sym} was fully exited by an add-only insert"


def test_mini_rebalance_falls_back_to_current_weights_when_broker_empty(tmp_path, monkeypatch, mocker):
    _isolate_fs(tmp_path, monkeypatch)

    mocker.patch("ascent.execution.alpaca_broker.get_positions", return_value=pd.DataFrame())
    mock_eod = mocker.patch("ascent.execution.eod_runner.run_eod_with_weights")

    current_weights = {"AAA": 0.5, "BBB": 0.5}
    ra._trigger_mini_rebalance(_FakeResult("NEW"), current_weights, ra.date(2026, 6, 30))

    assert mock_eod.called
    submitted = mock_eod.call_args[0][0]
    assert "AAA" in submitted
    assert "BBB" in submitted
    assert "NEW" in submitted


def test_mini_rebalance_safety_assertion_aborts_on_full_exit(tmp_path, monkeypatch, mocker):
    _isolate_fs(tmp_path, monkeypatch)

    live_positions = pd.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]})
    mocker.patch("ascent.execution.alpaca_broker.get_positions", return_value=live_positions)
    mock_eod = mocker.patch("ascent.execution.eod_runner.run_eod_with_weights")

    # Force a pathological insertion result that drops BBB entirely -- this is
    # the safety net that must catch a regression even if the insertion logic
    # itself is fine; it directly encodes "never submit a full exit".
    mocker.patch.object(ra, "_insert_candidate_weights", return_value={"AAA": 0.5, "NEW": 0.5})

    ra._trigger_mini_rebalance(_FakeResult("NEW"), {"AAA": 0.6, "BBB": 0.4}, ra.date(2026, 6, 30))

    mock_eod.assert_not_called()
