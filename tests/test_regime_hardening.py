# tests/test_regime_hardening.py
import pytest
import numpy as np
import pandas as pd


def test_regime_config_has_asymmetric_threshold_keys():
    from ascent.regime.types import REGIME_CONFIG_DEFAULTS
    assert "regime_downgrade_threshold" in REGIME_CONFIG_DEFAULTS, \
        "regime_downgrade_threshold must be in REGIME_CONFIG_DEFAULTS"
    assert "regime_upgrade_threshold" in REGIME_CONFIG_DEFAULTS, \
        "regime_upgrade_threshold must be in REGIME_CONFIG_DEFAULTS"
    assert REGIME_CONFIG_DEFAULTS["regime_downgrade_threshold"] == 0.40
    assert REGIME_CONFIG_DEFAULTS["regime_upgrade_threshold"] == 0.70
