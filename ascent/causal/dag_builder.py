"""ascent/causal/dag_builder.py

Per-symbol causal graph builder. Haiku reads fundamental + transcript +
SEC summary and returns 1-3 causal mechanisms per holding.
Cache: data_cache/causal_graphs/{symbol}_{quarter_end}.json
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data_cache/causal_graphs")

try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL
except ImportError:
    generate_structured = None  # type: ignore
    HAIKU_MODEL = "claude-haiku-4-5-20251001"

_MECHANISM_TYPES = (
    "momentum_catalyst", "quality_defensive", "macro_hedge",
    "mean_reversion", "valuation", "supply_demand_inflection",
)

_SYSTEM_PROMPT = (
    "You are a financial analyst building a causal model for a portfolio holding. "
    "Identify 1-3 causal mechanisms that explain the current investment thesis. "
    "Each mechanism must be falsifiable: state a specific observable condition "
    "that would break the thesis. Base your analysis only on the data provided — "
    "do not use training-data knowledge about the company beyond what is given. "
    "Respond with valid JSON matching the provided schema exactly. No other text."
)

_DAG_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanisms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mechanism": {
                        "type": "string",
                        "description": "One sentence: 'X causes Y via Z'",
                    },
                    "intervention": {
                        "type": "string",
                        "description": "IF [observable trigger] THEN [expected outcome]",
                    },
                    "falsification_condition": {
                        "type": "string",
                        "description": "IF [observable] < [threshold], thesis broken",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "Trading days until falsification check (21, 42, or 63)",
                    },
                    "timing": {
                        "type": "string",
                        "enum": ["priced_in", "not_yet_priced", "catalyst_imminent"],
                    },
                    "mechanism_type": {
                        "type": "string",
                        "enum": list(_MECHANISM_TYPES),
                    },
                },
                "required": [
                    "mechanism", "intervention", "falsification_condition",
                    "horizon_days", "timing", "mechanism_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mechanisms"],
    "additionalProperties": False,
}


def build_graph(
    symbol: str,
    quarter_end: str,
    fundamental_text: str,
    transcript_summary: str,
    sec_summary: str,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Build and cache a causal graph for a single symbol.

    Args:
        symbol: ticker (e.g. "WDC")
        quarter_end: ISO date string for the quarter (e.g. "2026-03-31")
        fundamental_text: formatted fundamental ratios (Q-3 to Q0)
        transcript_summary: short earnings call summary
        sec_summary: short 10-K summary
        cache_dir: directory for JSON cache files (default: DEFAULT_CACHE_DIR)

    Returns:
        Dict with {symbol, quarter_end, built_at, mechanisms[]}
    """
    from ascent.llm.client import generate_structured, HAIKU_MODEL

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{symbol}_{quarter_end}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            log.debug("[DagBuilder] Cache hit: %s %s", symbol, quarter_end)
            return cached
        except Exception:
            pass

    user_prompt = f"""Symbol: {symbol} | Quarter end: {quarter_end}

Fundamental ratios (Q-3 = three quarters ago, Q0 = most recent):
{fundamental_text}

Earnings call summary:
{transcript_summary or 'Not available'}

10-K / SEC filing summary:
{sec_summary or 'Not available'}

Build 1-3 causal mechanisms that explain this company's current investment dynamics.
For each: state a mechanism (X causes Y via Z), an intervention condition, a falsification condition,
a horizon, whether the mechanism is already priced in, and the mechanism type."""

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.3,
            use_cache=True,
            json_schema=_DAG_SCHEMA,
        )
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        mechanisms = parsed.get("mechanisms", [])
    except Exception as exc:
        log.warning("[DagBuilder] Haiku call failed for %s: %s", symbol, exc)
        mechanisms = []

    result = {
        "symbol": symbol,
        "quarter_end": quarter_end,
        "built_at": str(date.today()),
        "mechanisms": mechanisms,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    log.info("[DagBuilder] Built graph for %s: %d mechanisms", symbol, len(mechanisms))
    return result


def load_or_build(
    symbol: str,
    quarter_end: str,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Load cached graph if available; otherwise return empty graph.
    Does NOT trigger a Haiku call — use build_graph() for that.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = Path(cache_dir) / f"{symbol}_{quarter_end}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return {"symbol": symbol, "quarter_end": quarter_end, "built_at": None, "mechanisms": []}


def get_quarter_end(symbol: str) -> str:
    """
    Derive the most recent quarter_end date for a symbol from earnings.parquet.
    Falls back to current calendar quarter end if no earnings data.
    """
    try:
        import pandas as pd
        ep = Path("data_cache/earnings.parquet")
        if ep.exists():
            earnings = pd.read_parquet(ep)
            sym_rows = earnings[earnings["symbol"] == symbol].sort_values("date", ascending=False)
            if not sym_rows.empty:
                last_date = pd.to_datetime(sym_rows.iloc[0]["date"])
                month = ((last_date.month - 1) // 3 * 3) + 3
                import calendar
                day = calendar.monthrange(last_date.year, month)[1]
                return f"{last_date.year}-{month:02d}-{day:02d}"
    except Exception:
        pass

    # Fallback: current calendar quarter end
    today = date.today()
    month = ((today.month - 1) // 3 * 3) + 3
    import calendar
    day = calendar.monthrange(today.year, month)[1]
    return f"{today.year}-{month:02d}-{day:02d}"


def build_portfolio_graphs(
    symbols: list,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Build causal graphs for all holdings in the current portfolio.
    Called by weekend_runner. Returns {symbol: graph_dict}.
    """
    results = {}
    for symbol in symbols:
        quarter_end = get_quarter_end(symbol)
        check_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_path = check_dir / f"{symbol}_{quarter_end}.json"

        if cache_path.exists():
            log.debug("[DagBuilder] Already have graph for %s %s", symbol, quarter_end)
            results[symbol] = load_or_build(symbol, quarter_end, cache_dir)
            continue

        fundamental_text = _get_fundamental_text(symbol)
        transcript_summary = _get_transcript_summary(symbol)
        sec_summary = _get_sec_summary(symbol)

        results[symbol] = build_graph(
            symbol, quarter_end,
            fundamental_text, transcript_summary, sec_summary,
            cache_dir=cache_dir,
        )

    return results


def _get_fundamental_text(symbol: str) -> str:
    try:
        import pandas as pd
        fp = Path("data_cache/fundamentals.parquet")
        if not fp.exists():
            return "No fundamental data available"
        df = pd.read_parquet(fp)
        rows = df[df["symbol"] == symbol].sort_values("date", ascending=False).head(4)
        if rows.empty:
            return "No fundamental data available"
        lines = []
        for i, (_, row) in enumerate(rows.iterrows()):
            cols = [c for c in ["gross_profitability", "accruals_ratio", "asset_growth"]
                    if c in row.index and pd.notna(row[c])]
            vals = ", ".join(f"{c}={row[c]:.3f}" for c in cols)
            lines.append(f"Q-{i}: {vals}")
        return "\n".join(lines) if lines else "No data"
    except Exception:
        return "Fundamental data load error"


def _get_transcript_summary(symbol: str) -> str:
    try:
        import pandas as pd
        tp = Path("data_cache/altdata_transcripts.parquet")
        if not tp.exists():
            return ""
        df = pd.read_parquet(tp)
        rows = df[df["symbol"] == symbol].sort_values("date", ascending=False)
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("summary", ""))[:500]
    except Exception:
        return ""


def _get_sec_summary(symbol: str) -> str:
    try:
        import pandas as pd
        sp = Path("data_cache/altdata_sec.parquet")
        if not sp.exists():
            return ""
        df = pd.read_parquet(sp)
        rows = df[df["symbol"] == symbol].sort_values("date", ascending=False)
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("summary", ""))[:500]
    except Exception:
        return ""
