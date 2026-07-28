# tests/strategy/test_stop_loss_wiring.py
"""
Production wiring: the stop must run on the LIVE book, after every cap and
overlay, and must not be undone by redistribution.
"""
from unittest.mock import patch

import pandas as pd
import pytest


def _live_book():
    return pd.DataFrame([
        {"symbol": "ALGM", "qty": 66.5, "market_value": 3062.97,
         "current_price": 46.05, "weight": 0.029,
         "avg_entry_price": 66.370003, "unrealized_plpc": -0.30646},
        {"symbol": "TLT", "qty": 108.5, "market_value": 9090.78,
         "current_price": 83.80, "weight": 0.086,
         "avg_entry_price": 87.41, "unrealized_plpc": -0.04130},
    ])


class TestStopLossWiring:
    def test_disabled_flag_is_a_noop(self):
        import run_all_agents as raa
        from ascent.config.settings import get_config
        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", False):
            out, stopped = raa._apply_stop_loss_to_book(target, "2026-07-27")
        assert out == target
        assert stopped == []

    def test_breached_name_is_zeroed_and_others_untouched(self, tmp_path):
        import run_all_agents as raa
        from ascent.config.settings import get_config
        from ascent.portfolio import stop_loss as sl

        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", True), \
             patch.object(cfg.backtest, "stop_loss_threshold", 0.10), \
             patch.object(sl, "DEFAULT_STATE_PATH", str(tmp_path / "s.json")), \
             patch("ascent.execution.alpaca_broker.get_positions",
                   return_value=_live_book()):
            out, stopped = raa._apply_stop_loss_to_book(target, "2026-07-27")

        assert stopped == ["ALGM"]
        assert out["ALGM"] == 0.0
        assert out["TLT"] == pytest.approx(0.08)   # NOT refilled
        assert sum(out.values()) == pytest.approx(0.08)

    def test_blocked_symbol_is_not_re_added(self, tmp_path):
        import run_all_agents as raa
        from ascent.config.settings import get_config
        from ascent.portfolio import stop_loss as sl

        state = str(tmp_path / "s.json")
        sl.record_stops(["ALGM"], "2026-07-20", path=state)

        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", True), \
             patch.object(sl, "DEFAULT_STATE_PATH", state), \
             patch("ascent.execution.alpaca_broker.get_positions",
                   return_value=pd.DataFrame(columns=[
                       "symbol", "qty", "market_value", "current_price",
                       "weight", "avg_entry_price", "unrealized_plpc"])):
            out, _ = raa._apply_stop_loss_to_book(target, "2026-07-27")

        assert out["ALGM"] == 0.0      # inside cooldown
        assert out["TLT"] == pytest.approx(0.08)

    def test_broker_failure_leaves_book_unchanged(self):
        """Fail-open: a broker outage must not silently liquidate the book."""
        import run_all_agents as raa
        from ascent.config.settings import get_config
        target = {"ALGM": 0.04, "TLT": 0.08}
        cfg = get_config()
        with patch.object(cfg.backtest, "stop_loss_enabled", True), \
             patch("ascent.execution.alpaca_broker.get_positions",
                   side_effect=RuntimeError("broker down")):
            out, stopped = raa._apply_stop_loss_to_book(target, "2026-07-27")
        assert out == target
        assert stopped == []
