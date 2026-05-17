# tests/test_ai_pm_agent.py
import pytest
from unittest.mock import patch


# ── pm_risk_validator ──────────────────────────────────────────────────────────

def test_validator_accepts_clean_portfolio():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.08, "MSFT": 0.07, "GOOG": 0.06, "AMZN": 0.06,
        "META": 0.07, "NVDA": 0.08, "TSLA": 0.05, "JPM": 0.07,
        "V": 0.06, "UNH": 0.06,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is True
    assert violations == []


def test_validator_rejects_concentrated_position():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.50, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("AAPL" in v for v in violations)


def test_validator_rejects_too_few_positions():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {"AAPL": 0.50, "MSFT": 0.50}
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("positions" in v.lower() for v in violations)


def test_validator_rejects_empty_portfolio():
    from ascent.risk.pm_risk_validator import validate
    ok, violations = validate({})
    assert ok is False


def test_validator_rejects_distressed_name():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10, "BAD": 0.10, "JPM": 0.08,
        "V": 0.07, "UNH": 0.07, "GS": 0.08,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=["BAD"]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("BAD" in v for v in violations)


def test_validator_rejects_sector_overweight():
    from ascent.risk.pm_risk_validator import validate
    sector_map = {
        "AAPL": "tech", "MSFT": "tech", "GOOG": "tech", "AMZN": "tech",
        "META": "tech", "NVDA": "tech",
        "JPM": "finance", "V": "finance", "GS": "finance", "WFC": "finance",
    }
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "NVDA": 0.10,
        "JPM": 0.10, "V": 0.10, "GS": 0.10, "WFC": 0.10,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value=sector_map):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    assert any("tech" in v.lower() for v in violations)


def test_validator_rejects_negative_weight():
    from ascent.risk.pm_risk_validator import validate
    portfolio = {
        "AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
        "META": 0.10, "BAD": -0.05,
    }
    with patch("ascent.risk.pm_risk_validator._load_sector_map", return_value={}):
        with patch("ascent.risk.pm_risk_validator._get_distressed_names", return_value=[]):
            ok, violations = validate(portfolio)
    assert ok is False
    # Only the negative-weight violation — no spurious position-limit violations
    assert any("BAD" in v for v in violations)
    assert not any("exceeds max" in v for v in violations)
