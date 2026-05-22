import pytest
from unittest.mock import patch, MagicMock


def _make_calibration_report(ic: float) -> dict:
    status = "Calibrated" if ic >= 0.20 else ("Weak" if ic >= 0.05 else "Uncalibrated")
    return {"spearman_ic": ic, "status": status, "n_rebalances": 5}


def test_uncalibrated_injects_warning():
    """When IC < 0.05, system prompt contains calibration warning."""
    from agents import ai_pm_agent
    prompt = ai_pm_agent._build_system_prompt(ic=0.02)
    assert "CALIBRATION WARNING" in prompt or "uncalibrated" in prompt.lower()


def test_calibrated_no_warning():
    from agents import ai_pm_agent
    prompt = ai_pm_agent._build_system_prompt(ic=0.25)
    assert "CALIBRATION WARNING" not in prompt
    assert "uncalibrated" not in prompt.lower()


def test_build_system_prompt_none_ic():
    """None IC (no data yet) — no warning injected."""
    from agents import ai_pm_agent
    prompt = ai_pm_agent._build_system_prompt(ic=None)
    assert isinstance(prompt, str)
    assert len(prompt) > 100
