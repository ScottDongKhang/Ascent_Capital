# tests/test_plan_b.py
def test_em_commodity_cap_enforced():
    """Merged weights must not exceed 20% in EM+commodity+gold combined."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {
        "EEM":  0.12, "VWO":  0.10, "GLD":  0.08, "PDBC": 0.07,
        "AAPL": 0.25, "MSFT": 0.20, "JPM":  0.18,
    }
    capped = _cap_em_commodity(weights)

    em_commodity_total = (
        capped.get("EEM", 0) + capped.get("VWO", 0) +
        capped.get("GLD", 0) + capped.get("PDBC", 0)
    )
    assert em_commodity_total <= 0.201, f"EM+commodity {em_commodity_total:.1%} exceeds 20%"
    assert abs(sum(capped.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


def test_em_commodity_cap_no_op_when_under():
    """Cap must be a no-op when EM+commodity is already under 20%."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {"EEM": 0.05, "GLD": 0.06, "AAPL": 0.50, "MSFT": 0.39}
    capped = _cap_em_commodity(weights)

    assert abs(capped["EEM"] - 0.05) < 0.0001
    assert abs(capped["GLD"] - 0.06) < 0.0001


def test_em_commodity_cap_preserves_non_em():
    """Non-EM symbols should gain weight when EM is trimmed."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {"EEM": 0.20, "GLD": 0.15, "AAPL": 0.40, "JPM": 0.25}
    capped = _cap_em_commodity(weights)

    em_total = capped.get("EEM", 0) + capped.get("GLD", 0)
    assert em_total <= 0.201
    assert capped.get("AAPL", 0) > 0.40
