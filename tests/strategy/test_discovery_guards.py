"""
Guards on the off-calendar discovery / mini-rebalance path.

Two behaviors are enforced here:

1. Calendar proximity guard — discovery must NOT fire within N trading days of a
   scheduled rebalance (the next scheduled rebalance will recompute the book
   anyway; an off-calendar full rotation two days prior is pure churn).

2. Add-only insertion — when discovery DOES fire, it must insert the candidate
   and trim the rest of the book pro-rata, NOT re-rank/re-optimize every holding.
"""
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
