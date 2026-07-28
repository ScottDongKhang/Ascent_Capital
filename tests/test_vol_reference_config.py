# tests/test_vol_reference_config.py
"""Vol-targeting reference series. Defaults to the CURRENT behaviour."""
from ascent.config.settings import get_config


def test_vol_reference_defaults_to_spy():
    assert get_config().backtest.vol_target_reference == "spy", (
        "must default to the existing market-referenced behaviour until the "
        "walk-forward comparison in the strategy-own-vol-targeting plan"
    )


def test_vol_reference_is_a_known_value():
    assert get_config().backtest.vol_target_reference in ("spy", "strategy")
