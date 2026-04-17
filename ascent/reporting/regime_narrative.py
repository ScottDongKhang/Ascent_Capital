"""
ascent/reporting/regime_narrative.py
Morning regime narrative generator.

Reads the current regime signal + recent macro ETF moves and writes
two plain-English sentences explaining WHY the regime is what it is.

Replaces the raw "calm_bull" label with something readable like:
"Equities are trending with low volatility and broad participation.
Rates are stable and the dollar is range-bound — no macro headwinds."

Uses Haiku. Runs once per day, cached so it doesn't re-fire.

Called by:
    eod_runner.py at the start of each EOD run
    (or standalone: python3 -m ascent.reporting.regime_narrative)

Output:
    dashboard/regime_narrative.json
    {
        "date": "...",
        "regime": "calm_bull",
        "narrative": "two sentences",
        "macro_context": {...}
    }
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

from ascent.llm.client import generate_structured, HAIKU_MODEL
NARRATIVE_PATH   = Path("dashboard/regime_narrative.json")
REGIME_JSON_PATH = Path("dashboard/regime_signal.json")
MACRO_ETF_CACHE  = Path("data_cache/macro_live.parquet")

MACRO_ETFS = ["TLT", "IEF", "UUP", "GLD", "PDBC", "HYG", "LQD"]


def _load_regime_signal() -> Optional[dict]:
    """Load latest regime signal from dashboard JSON."""
    if not REGIME_JSON_PATH.exists():
        return None
    try:
        data = json.loads(REGIME_JSON_PATH.read_text())
        # Handle both list and dict formats
        if isinstance(data, list) and data:
            return data[-1]
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _load_macro_moves() -> dict:
    """
    Load recent macro ETF returns from parquet cache.
    Returns {symbol: 5d_return} for context.
    Falls back to empty dict if unavailable.
    """
    try:
        import pandas as pd
        if not MACRO_ETF_CACHE.exists():
            return {}
        df = pd.read_parquet(MACRO_ETF_CACHE)
        # Expect wide format with dates as index
        if df.empty:
            return {}
        recent = df.tail(6)  # last 5 trading days + 1
        moves  = {}
        for col in recent.columns:
            sym = str(col).upper()
            if sym in MACRO_ETFS and len(recent[col].dropna()) >= 2:
                start = recent[col].dropna().iloc[0]
                end   = recent[col].dropna().iloc[-1]
                if start and start != 0:
                    moves[sym] = round((end - start) / start, 4)
        return moves
    except Exception:
        return {}


def _already_generated_today() -> bool:
    if not NARRATIVE_PATH.exists():
        return False
    try:
        rec = json.loads(NARRATIVE_PATH.read_text())
        return rec.get("date") == date.today().isoformat()
    except Exception:
        return False


def generate_regime_narrative(force: bool = False) -> Optional[str]:
    """
    Generate and cache today's regime narrative.

    Args:
        force: Regenerate even if already done today

    Returns:
        The narrative string, or None on failure
    """
    if not force and _already_generated_today():
        try:
            rec = json.loads(NARRATIVE_PATH.read_text())
            return rec.get("narrative")
        except Exception:
            pass

    regime_data = _load_regime_signal()
    regime      = "unknown"
    if regime_data:
        regime = regime_data.get("regime") or regime_data.get("label") or "unknown"

    macro_moves = _load_macro_moves()

    # Build macro context string
    macro_lines = []
    for sym, move in sorted(macro_moves.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "↑" if move > 0 else "↓"
        macro_lines.append(f"{sym} {direction}{abs(move):.1%}")
    macro_str = ", ".join(macro_lines) if macro_lines else "no macro data available"

    regime_descriptions = {
        "calm_bull":  "trending higher with low volatility and broad sector participation",
        "stressed":   "under pressure with elevated volatility and defensive rotation",
        "crisis":     "in crisis mode with sharp drawdowns and correlation spikes",
        "neutral":    "range-bound with mixed signals and no clear directional trend",
        "uncertain":  "sending mixed signals — regime classification is uncertain",
        "euphoric":   "in a late-stage rally with stretched valuations and narrow leadership",
    }
    regime_context = regime_descriptions.get(str(regime).lower(), "in an unclassified state")

    user_prompt = f"""
Today's date: {date.today().isoformat()}
Current regime: {regime} — equities are {regime_context}
Macro ETF moves (last 5 days): {macro_str}

Write exactly two sentences explaining WHY the market is in this regime right now.
First sentence: what the equity market is doing and why.
Second sentence: what the macro backdrop (rates, dollar, commodities) is doing and what it means.
Be specific. Use the ETF moves as evidence. No preamble, no labels, just the two sentences.
"""

    try:
        narrative = generate_structured(
            system_prompt=(
                "You are the morning market commentator at Ascent Capital. "
                "Write clear, specific, two-sentence regime narratives for the portfolio team. "
                "Reference actual instrument moves. Sound like a seasoned macro analyst, "
                "not a chatbot. Never use the word 'regime'."
            ),
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=150,
            temperature=0.5,
        )
    except Exception as e:
        print(f"[RegimeNarrative] LLM call failed: {e}")
        return None

    record = {
        "date":          date.today().isoformat(),
        "regime":        regime,
        "narrative":     narrative,
        "macro_context": macro_moves,
    }

    NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NARRATIVE_PATH, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[RegimeNarrative] {regime.upper()}: {narrative}")
    return narrative


def load_narrative() -> str:
    """Load today's cached narrative, or generate if missing."""
    if _already_generated_today():
        try:
            rec = json.loads(NARRATIVE_PATH.read_text())
            return rec.get("narrative", "")
        except Exception:
            pass
    return generate_regime_narrative() or ""


if __name__ == "__main__":
    result = generate_regime_narrative(force=True)
    if result:
        print(f"\nNarrative: {result}")
    else:
        print("Failed to generate narrative")
