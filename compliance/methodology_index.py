"""
compliance/methodology_index.py
Machine-readable index of all active strategy components.

Exported daily to dashboard/methodology_index.json.
Used by the audit trail to record the exact system configuration at each rebalance.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

EXPORT_PATH = Path("dashboard/methodology_index.json")


def _load_active_config() -> dict:
    p = Path("data_cache/active_alpha_config.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def build_methodology_index() -> Dict[str, Any]:
    """
    Build a machine-readable registry of all strategy components.

    Returns dict with keys: alpha_sleeves, data_sources, risk_controls, execution_methods.
    Each entry: {name, file, description, ic_threshold_required, current_weight, status}
    """
    config = _load_active_config()
    weights = config.get("alpha_weights", {})

    alpha_sleeves: List[dict] = [
        {
            "name":                 "trend",
            "file":                 "ascent/alpha/trend.py",
            "description":          "Cross-sectional 12M-1M momentum (skip-last-month variant at 20% sub-weight)",
            "ic_threshold_required": False,
            "current_weight":        weights.get("trend", 0.41),
            "status":                "active",
        },
        {
            "name":                 "statarb",
            "file":                 "ascent/alpha/statarb.py",
            "description":          "Sector residual mean reversion using sector profiles",
            "ic_threshold_required": False,
            "current_weight":        weights.get("statarb", 0.15),
            "status":                "active",
        },
        {
            "name":                 "ml",
            "file":                 "ascent/alpha/ml_sleeve.py",
            "description":          "XGBoost CPCV C(6,2)=15 folds; 6 features by IC/IR; p5 guard > -0.05",
            "ic_threshold_required": True,
            "current_weight":        weights.get("ml", 0.10),
            "status":                "active",
        },
        {
            "name":                 "mean_reversion",
            "file":                 "ascent/alpha/meanrev.py",
            "description":          "Short-term reversal (5-day zscore)",
            "ic_threshold_required": False,
            "current_weight":        weights.get("mean_reversion", 0.05),
            "status":                "active",
        },
        {
            "name":                 "volatility",
            "file":                 "ascent/alpha/stack.py",
            "description":          "Long names with declining and stable volatility: -(vol_trend_10d / vol_of_vol_21d)",
            "ic_threshold_required": False,
            "current_weight":        weights.get("volatility", 0.05),
            "status":                "active",
        },
        {
            "name":                 "fundamental",
            "file":                 "ascent/alpha/fundamental.py",
            "description":          "Gross profitability, accruals, asset growth; 45-day filing lag",
            "ic_threshold_required": False,
            "current_weight":        weights.get("fundamental", 0.05),
            "status":                "active",
        },
        {
            "name":                 "earnings",
            "file":                 "ascent/alpha/earnings.py",
            "description":          "PEAD: EPS surprise z-score, OLS momentum-beta residual; 1-bday lag",
            "ic_threshold_required": False,
            "current_weight":        weights.get("earnings", 0.05),
            "status":                "active",
        },
        {
            "name":                 "analyst",
            "file":                 "ascent/alpha/stack.py",
            "description":          "Analyst revision signal; sparse — zero-filled if cache absent",
            "ic_threshold_required": False,
            "current_weight":        weights.get("analyst", 0.05),
            "status":                "sparse",
        },
        {
            "name":                 "llm_fundamental",
            "file":                 "ascent/alpha/llm_fundamental.py",
            "description":          "Chicago Booth 6-step CoT via Haiku; cached by (symbol, quarter_end); 45-day lag",
            "ic_threshold_required": False,
            "current_weight":        weights.get("llm_fundamental", 0.03),
            "status":                "active",
        },
        {
            "name":                 "options_flow",
            "file":                 "ascent/alpha/stack.py",
            "description":          "Options sentiment; sparse — zero-filled if cache absent",
            "ic_threshold_required": False,
            "current_weight":        weights.get("options_flow", 0.02),
            "status":                "sparse",
        },
        {
            "name":                 "insider",
            "file":                 "ascent/alpha/stack.py",
            "description":          "Insider net transaction score; sparse",
            "ic_threshold_required": False,
            "current_weight":        weights.get("insider", 0.02),
            "status":                "sparse",
        },
        {
            "name":                 "short_interest",
            "file":                 "ascent/alpha/stack.py",
            "description":          "Short squeeze signal; sparse",
            "ic_threshold_required": False,
            "current_weight":        weights.get("short_interest", 0.02),
            "status":                "sparse",
        },
        {
            "name":                 "altdata",
            "file":                 "ascent/alpha/altdata_alpha.py",
            "description":          "IC-validated alternative data (SEC 10-K, transcripts, Reddit, Trends); 0% until first source validated",
            "ic_threshold_required": True,
            "current_weight":        weights.get("altdata", 0.00),
            "status":                "gated",
        },
    ]

    data_sources = [
        {"name": "Yahoo Finance",     "file": "ascent/data/ingest/yahoo.py",          "lag_days": 0,  "status": "active"},
        {"name": "FRED",              "file": "ascent/data/ingest/fred.py",            "lag_days": 1,  "status": "active"},
        {"name": "EDGAR (8-K)",       "file": "ascent/data/ingest/edgar_listener.py", "lag_days": 0,  "status": "live_disabled"},
        {"name": "SEC Full-Text",     "file": "ascent/data/ingest/sec_filings.py",    "lag_days": 45, "status": "active"},
        {"name": "Earnings Trans.",   "file": "ascent/data/ingest/earnings_transcripts.py", "lag_days": 1, "status": "active"},
        {"name": "Reddit Sentiment",  "file": "ascent/data/ingest/reddit_sentiment.py","lag_days": 0, "status": "active"},
        {"name": "Google Trends",     "file": "ascent/data/ingest/google_trends.py",  "lag_days": 1,  "status": "active"},
        {"name": "Capitol Trades",    "file": "ascent/data/ingest/capitol_trades.py", "lag_days": 2,  "status": "live_disabled"},
        {"name": "Alpaca Options",    "file": "ascent/data/ingest/options_scanner.py","lag_days": 0,  "status": "live_disabled"},
    ]

    risk_controls = [
        {"name": "Kill Switch",    "file": "ascent/execution/kill_switch.py",    "threshold": "15% drawdown"},
        {"name": "Approval Gate",  "file": "ascent/execution/eod_runner.py",     "threshold": "> 2% NAV per order"},
        {"name": "SPY 200MA",      "file": "ascent/main.py",                     "threshold": "SPY < 200MA → 70% exposure"},
        {"name": "Factor VaR",     "file": "ascent/risk/var_cvar.py",            "threshold": "95% 1-day VaR"},
        {"name": "Sector Cap",     "file": "ascent/portfolio/optimizer.py",      "threshold": "max 1 per sector"},
        {"name": "Position Cap",   "file": "ascent/portfolio/optimizer.py",      "threshold": "max 10% per name"},
        {"name": "EM+Commodity Cap","file":"orchestrator/central_intelligence.py","threshold": "max 20% aggregate"},
        {"name": "Intraday Trigger","file":"ascent/execution/intraday_trigger.py","threshold": "SPY -3%+VIX>30 or DD 12%"},
    ]

    execution_methods = [
        {"name": "Market Orders",  "file": "ascent/execution/alpaca_broker.py",  "status": "active"},
        {"name": "TWAP Executor",  "file": "ascent/execution/twap_executor.py",  "status": "kill_switched_off", "gate": "> 5% ADV"},
        {"name": "IS Measurement", "file": "ascent/execution/implementation_shortfall.py", "status": "active"},
        {"name": "Capacity Model", "file": "ascent/execution/capacity_model.py", "status": "informational"},
        {"name": "Event Trades",   "file": "ascent/execution/event_runner.py",   "status": "kill_switched_off"},
    ]

    return {
        "as_of_date":       date.today().isoformat(),
        "alpha_sleeves":    alpha_sleeves,
        "data_sources":     data_sources,
        "risk_controls":    risk_controls,
        "execution_methods": execution_methods,
        "active_weight_sum": round(sum(s["current_weight"] for s in alpha_sleeves), 4),
    }


def export_methodology_index() -> None:
    """Write methodology index to dashboard/methodology_index.json."""
    idx = build_methodology_index()
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(idx, indent=2))
    log.info("[MethodologyIndex] Exported to %s", EXPORT_PATH)
