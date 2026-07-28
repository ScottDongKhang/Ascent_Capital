# tests/test_stop_loss_config.py
"""Stop-loss config surface. Ships DISABLED pending WF validation."""
from ascent.config.settings import get_config


def test_stop_loss_flags_exist_with_paper_defaults():
    bt = get_config().backtest
    assert bt.stop_loss_enabled is False, (
        "stop-loss must ship disabled until walk-forward validation "
        "(Task 7 of the position-stop-loss plan) says otherwise"
    )
    assert bt.stop_loss_threshold == 0.10       # Han, Zhou & Zhu (2014)
    assert bt.stop_loss_cooldown_days == 30     # ~21 trading days
    assert bt.stop_loss_redistribute is False   # freed weight -> cash
