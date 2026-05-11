"""
compliance/risk_disclosure.py
Risk disclosure generator.

NOT legal advice. Template output must be reviewed by a lawyer before
sharing with any investor. Populate with current strategy parameters.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def _get_current_risk_parameters() -> Dict[str, Any]:
    """Read live system state to populate disclosure parameters."""
    params: Dict[str, Any] = {
        "max_leverage":        1.0,
        "asset_classes":       ["US Equities", "Fixed Income ETFs", "Commodities", "International Equities", "Alternatives"],
        "n_positions":         15,
        "event_trading":       False,
        "twap_enabled":        False,
        "uses_options":        False,
        "inception_date":      "April 1, 2026",
        "max_single_position": "10%",
        "kill_switch_pct":     "15%",
        "rebalance_freq_days": 10,
    }

    # Try to get live values
    try:
        weights = json.loads(Path("execution/merged_weights.json").read_text())
        w = weights.get("weights", {})
        if w:
            params["n_positions"] = len(w)
    except Exception:
        pass

    try:
        from ascent.config.settings import get_config
        cfg = get_config()
        params["max_single_position"] = f"{int(cfg.portfolio.max_weight * 100)}%"
        params["n_positions"]         = cfg.portfolio.top_n
    except Exception:
        pass

    try:
        from ascent.execution.event_runner import EVENT_TRADING_ENABLED
        params["event_trading"] = EVENT_TRADING_ENABLED
    except Exception:
        pass

    try:
        from ascent.execution.twap_executor import TWAP_ENABLED
        params["twap_enabled"] = TWAP_ENABLED
    except Exception:
        pass

    return params


def generate_risk_disclosures(strategy_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate standard risk disclosure language.

    NOT legal advice — review with counsel before distribution.
    """
    p = strategy_params or _get_current_risk_parameters()

    asset_class_str = ", ".join(p.get("asset_classes", ["various asset classes"]))
    n_pos           = p.get("n_positions", 15)
    max_pos         = p.get("max_single_position", "10%")
    leverage        = p.get("max_leverage", 1.0)
    inception       = p.get("inception_date", "April 2026")
    kill_switch     = p.get("kill_switch_pct", "15%")

    disclosures = f"""RISK DISCLOSURES

As of: {date.today().isoformat()}
Strategy: Ascent Capital Systematic Multi-Factor Strategy
Inception: {inception}

IMPORTANT — PLEASE READ CAREFULLY

1. PAST PERFORMANCE
Past performance is not indicative of future results. The strategy's historical and walk-forward simulated returns do not guarantee similar future performance. All return figures presented are net of estimated transaction costs unless stated otherwise.

2. SYSTEMATIC STRATEGY RISK
Ascent Capital is a fully systematic, algorithmic trading strategy. Systematic strategies may fail without warning when market dynamics shift or when their underlying statistical assumptions no longer hold. The strategy was developed using historical data and may underperform or incur losses in market conditions that are structurally different from the training period.

3. LEVERAGE
{"This strategy does not use leverage. Maximum gross exposure is 100% of NAV." if leverage <= 1.0 else f"This strategy uses up to {leverage}x leverage. Leverage amplifies both gains and losses."}

4. LIQUIDITY RISK
The strategy holds approximately {n_pos} positions in {asset_class_str}. In periods of severe market stress, positions may be difficult to liquidate at quoted prices. The strategy imposes automatic exposure reduction (kill switch at -{kill_switch} drawdown) to manage liquidity risk.

5. MODEL RISK
All alpha signals are derived from historical price data, fundamental data, and alternative data sources. These signals may not predict future returns. The machine learning components of the strategy (XGBoost alpha sleeve) are periodically retrained but may overfit to historical patterns. Factor models used for portfolio construction are estimates subject to estimation error.

6. CONCENTRATION RISK
The strategy holds approximately {n_pos} positions with a maximum single-position size of {max_pos}. Loss of any single position has material impact on portfolio performance.

7. EXECUTION RISK
The strategy uses algorithmic order execution including TWAP (time-weighted average price) execution for large orders. Actual fills may differ from estimated prices due to market impact, slippage, and execution timing.

8. REGULATORY RISK
The strategy is operated as a proprietary systematic trading system. It is not registered with the SEC or any regulatory authority. This does not constitute an offer to sell or solicitation to buy any security. Before accepting any outside capital, consult with securities counsel regarding applicable exemptions and registration requirements.

9. TECHNOLOGY RISK
The strategy relies on automated systems, data providers, and cloud infrastructure. System failures, data provider outages, or connectivity issues may prevent timely order execution or cause unintended trades.

10. AI AND LLM RISK
This strategy employs large language models (LLMs) as advisory components in its debate layer and as alpha signal generators. LLM outputs are not guaranteed to be accurate, consistent, or free of hallucination. All LLM components are advisory only and do not directly control execution without quantitative validation.

DISCLAIMER
This document is provided for informational purposes only and does not constitute legal, tax, or investment advice. This is not an offer to sell or a solicitation of an offer to buy any securities. Potential investors should conduct their own due diligence and consult appropriate professional advisors before making any investment decision.
"""
    return disclosures


