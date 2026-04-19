"""
ascent/execution/debate_gate.py

Decides whether the debate layer should run on a given rebalance day.
Debate only fires when uncertainty is elevated — not on every rebalance.
This creates the control group needed to measure AI contribution.

Criteria (any one triggers debate):
  - Regime entropy > 0.70 (uncertain regime)
  - Top position > 12% (concentration risk)
  - VaR 99th percentile < -3.5% (tail risk elevated)
  - Catalyst detected in last 48 hours
"""
from __future__ import annotations

ENTROPY_THRESHOLD   = 0.70
POSITION_THRESHOLD  = 0.12
VAR_99_THRESHOLD    = -0.035


def should_run_debate(portfolio_state: dict, regime_signal: dict) -> bool:
    """
    Return True if the debate layer should run on this rebalance.

    Args:
        portfolio_state: dict with 'weights', 'quant_context', 'catalyst_detected'
        regime_signal:   dict with 'entropy', 'label'

    Returns:
        True if at least one trigger condition is met.
    """
    # Trigger 1: regime uncertainty
    entropy = float(regime_signal.get("entropy", 0.0) or 0.0)
    if entropy > ENTROPY_THRESHOLD:
        print(f"[DebateGate] TRIGGER: regime entropy {entropy:.2f} > {ENTROPY_THRESHOLD}")
        return True

    # Trigger 2: position concentration
    weights = portfolio_state.get("weights") or {}
    top_position = max(weights.values(), default=0.0)
    if top_position > POSITION_THRESHOLD:
        print(f"[DebateGate] TRIGGER: top position {top_position:.1%} > {POSITION_THRESHOLD:.0%}")
        return True

    # Trigger 3: tail risk
    qctx   = portfolio_state.get("quant_context") or {}
    var_99 = float(qctx.get("portfolio_var_99", 0.0) or 0.0)
    if var_99 < VAR_99_THRESHOLD:
        print(f"[DebateGate] TRIGGER: VaR 99 {var_99:.2%} < {VAR_99_THRESHOLD:.1%}")
        return True

    # Trigger 4: catalyst detected
    if portfolio_state.get("catalyst_detected"):
        print(f"[DebateGate] TRIGGER: catalyst detected")
        return True

    print(f"[DebateGate] SKIP: entropy={entropy:.2f}, top={top_position:.1%}, "
          f"var99={var_99:.2%}, catalyst=False — no trigger")
    return False
