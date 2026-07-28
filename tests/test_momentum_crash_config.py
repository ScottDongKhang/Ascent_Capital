# tests/test_momentum_crash_config.py
"""Momentum-crash overlay config. Ships DISABLED pending validation."""
from ascent.config.settings import get_config


def test_crash_overlay_ships_disabled():
    bt = get_config().backtest
    assert bt.momentum_crash_overlay_enabled is False
    assert bt.momentum_crash_multiplier == 0.50
