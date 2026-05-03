"""ascent/data/ingest/analyst.py

Fetch analyst upgrade/downgrade revision data from Yahoo Finance.
Applies a 1-business-day signal lag so the action is only visible the
trading day after it's published (avoids look-ahead on intraday timing).

Score mapping:
    up / upgraded / upgrade              → +1
    down / downgraded / downgrade        → -1
    init with bullish grade              → +1
    init with bearish grade              → -1
    reit / main / maintained / reiterate →  0  (no new directional info)
"""
from __future__ import annotations
import logging
import time
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Grade strings → direction for initiation actions (case-insensitive substrings)
_BULLISH_GRADES = {"buy", "outperform", "overweight", "strong buy", "positive",
                   "accumulate", "add", "conviction buy"}
_BEARISH_GRADES = {"sell", "underperform", "underweight", "strong sell", "negative",
                   "reduce", "avoid", "conviction sell"}


def _grade_score(grade: str) -> int:
    """Map a grade string to +1 / -1 / 0."""
    if not grade:
        return 0
    g = str(grade).lower()
    if any(b in g for b in _BULLISH_GRADES):
        return 1
    if any(b in g for b in _BEARISH_GRADES):
        return -1
    return 0


def _action_score(action: str, to_grade: str = "") -> int:
    """Map an analyst action + optional to_grade to a numeric score."""
    a = str(action).lower()
    if any(k in a for k in ("up", "upgraded", "upgrade", "raised")):
        return 1
    if any(k in a for k in ("down", "downgraded", "downgrade", "lowered", "lower")):
        return -1
    if any(k in a for k in ("init", "initiat")):
        return _grade_score(to_grade)
    return 0  # reit, main, maintained, etc.


def _fetch_one(sym: str) -> pd.DataFrame:
    """
    Fetch analyst upgrade/downgrade history for a single symbol.

    Returns long-format DataFrame with columns:
        symbol, signal_date, score

    signal_date = action_date + 1 business day (avoid look-ahead).
    score is +1 (bullish), -1 (bearish), or 0 (neutral).
    Rows with score == 0 are kept so the panel captures absence of activity.
    """
    try:
        t = yf.Ticker(sym)
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            return pd.DataFrame()

        ud = ud.copy().reset_index()

        # Normalise column names across yfinance versions
        ud.columns = [c.strip() for c in ud.columns]
        col_map = {c: c.lower().replace(" ", "_") for c in ud.columns}
        ud = ud.rename(columns=col_map)

        # Find date column
        date_col = next(
            (c for c in ud.columns if "date" in c or "time" in c),
            None,
        )
        if date_col is None:
            return pd.DataFrame()

        # Find action and grade columns
        action_col  = next((c for c in ud.columns if "action" in c), None)
        to_grade_col = next((c for c in ud.columns if "tograde" in c or "to_grade" in c), None)

        if action_col is None and to_grade_col is None:
            return pd.DataFrame()

        ud[date_col] = pd.to_datetime(ud[date_col])
        if ud[date_col].dt.tz is not None:
            ud[date_col] = ud[date_col].dt.tz_localize(None)

        rows = []
        for _, row in ud.iterrows():
            action   = str(row.get(action_col, "")) if action_col else ""
            to_grade = str(row.get(to_grade_col, "")) if to_grade_col else ""
            score    = _action_score(action, to_grade)
            signal_date = row[date_col] + pd.offsets.BDay(1)
            rows.append({
                "symbol":      sym,
                "signal_date": signal_date,
                "score":       score,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        return df

    except Exception as e:
        log.debug("analyst fetch failed for %s: %s", sym, e)
        return pd.DataFrame()


def fetch_analyst(symbols: list, delay_s: float = 0.3) -> pd.DataFrame:
    """
    Fetch analyst revision data for a list of symbols.
    Returns long-format DataFrame with columns: symbol, signal_date, score.
    """
    frames = []
    for i, sym in enumerate(symbols):
        df = _fetch_one(sym)
        if not df.empty:
            frames.append(df)
        if i > 0 and i % 25 == 0:
            log.info("analyst: %d/%d symbols fetched", i, len(symbols))
        if delay_s > 0:
            time.sleep(delay_s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_analyst(df: pd.DataFrame) -> None:
    """Append new analyst rows, dedup by (symbol, signal_date). Never overwrites history."""
    from ascent.data.store.parquet import save_parquet, load_parquet, has_data
    if df is None or df.empty:
        return
    existing = pd.DataFrame()
    if has_data("analyst_revisions"):
        try:
            existing = load_parquet("analyst_revisions")
        except Exception:
            pass
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
        combined["signal_date"] = pd.to_datetime(combined["signal_date"]).dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["symbol", "signal_date"], keep="last")
        combined = combined.sort_values(["symbol", "signal_date"])
    else:
        combined = df.copy()
        combined["signal_date"] = pd.to_datetime(combined["signal_date"]).dt.tz_localize(None)
    save_parquet(combined, "analyst_revisions")
    log.info(
        "analyst: saved %d rows, %d symbols",
        len(combined),
        combined["symbol"].nunique() if not combined.empty else 0,
    )


def load_analyst() -> pd.DataFrame:
    from ascent.data.store.parquet import load_parquet, has_data
    if not has_data("analyst_revisions"):
        return pd.DataFrame()
    return load_parquet("analyst_revisions")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ascent.config.settings import get_config
    cfg = get_config()
    symbols = list(cfg.universe.symbols)
    print(f"Fetching analyst revisions for {len(symbols)} symbols...")
    df = fetch_analyst(symbols, delay_s=0.3)
    if not df.empty:
        save_analyst(df)
        print(f"Done. {len(df)} rows, {df['symbol'].nunique()} symbols saved.")
    else:
        print("No analyst data returned.")
