"""
ascent/execution/debate_gate.py

The Adversarial Intelligence layer fires every rebalance.
No entropy gate — every rebalance is worth the intervention check.

The former conditional gate (entropy > 0.70, position > 12%, VaR < -3.5%) is removed.
Rationale: the adversarial engine's job is to find risks the quant model can't see.
That risk exists on every rebalance, not just high-entropy ones.
"""


def should_run_debate(portfolio_state: dict, regime_signal: dict) -> bool:
    """Always returns True — Adversarial Intelligence fires every rebalance."""
    print("[DebateGate] Adversarial Intelligence: firing every rebalance (no entropy gate)")
    return True
