"""AI pre-thesis / regime-assessment priors must expire.

`ascent/main.py` reads `data_cache/ai_prethesis_latest.json` and
`data_cache/ai_regime_assessment.json` and uses them to floor conviction names'
alpha, zero out the avoid-list, and blend the regime label. Neither read had any
staleness check: on the 2026-07-27 run it consumed a pre-thesis dated
2026-06-24 — 33 days old — and six of those ten conviction names were in the
executed book.

That is uncapped by earned authority and silently steers the pipeline that
Track A-star is snapshotted from, i.e. the supposed no-AI-PM baseline used to
judge the AI PM.
"""

import datetime as dt

from ascent.utils.freshness import AI_PRIOR_MAX_AGE_DAYS, ai_prior_is_fresh


class TestAiPriorIsFresh:
    def test_same_day_prior_is_fresh(self):
        assert ai_prior_is_fresh("2026-07-27", today=dt.date(2026, 7, 27)) is True

    def test_prior_from_current_rebalance_cycle_is_fresh(self):
        """Rebalances are ~14 calendar days apart; the current cycle's view counts."""
        assert ai_prior_is_fresh("2026-07-22", today=dt.date(2026, 7, 27)) is True

    def test_the_actual_incident_33_days_stale_is_rejected(self):
        assert ai_prior_is_fresh("2026-06-24", today=dt.date(2026, 7, 27)) is False

    def test_boundary_is_inclusive(self):
        today = dt.date(2026, 7, 27)
        edge = (today - dt.timedelta(days=AI_PRIOR_MAX_AGE_DAYS)).isoformat()
        assert ai_prior_is_fresh(edge, today=today) is True
        over = (today - dt.timedelta(days=AI_PRIOR_MAX_AGE_DAYS + 1)).isoformat()
        assert ai_prior_is_fresh(over, today=today) is False

    def test_missing_or_malformed_date_is_not_fresh(self):
        """Fail CLOSED: an unstamped prior must not steer the portfolio."""
        for bad in ("", None, "not-a-date", "2026-13-45", 20260727):
            assert ai_prior_is_fresh(bad, today=dt.date(2026, 7, 27)) is False

    def test_future_dated_prior_is_rejected(self):
        """A prior dated ahead of the run is a clock or override bug, not signal."""
        assert ai_prior_is_fresh("2026-07-28", today=dt.date(2026, 7, 27)) is False

    def test_defaults_to_market_today_when_today_omitted(self):
        from ascent.utils.market_time import market_today
        assert ai_prior_is_fresh(market_today().isoformat()) is True


class TestWiredIntoPipeline:
    """The helper is only useful if main.py actually consults it. Asserting on
    source text because run_pipeline() needs the whole engine to execute."""

    def _src(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "ascent/main.py")) as f:
            return f.read()

    def test_both_prior_reads_are_gated(self):
        src = self._src()
        assert src.count("ai_prior_is_fresh(") >= 2, (
            "both the regime assessment and the pre-thesis read must be gated"
        )

    def test_prethesis_symbols_only_assigned_when_fresh(self):
        """The conviction/avoid lists must be populated inside the fresh branch,
        not before it — otherwise the gate logs a warning and steers anyway."""
        src = self._src()
        i_gate = src.index("if not ai_prior_is_fresh(_pt_date)")
        i_assign = src.index("_ai_conviction_syms = [n[\"symbol\"]")
        assert i_assign > i_gate, "conviction list is assigned before the freshness gate"
