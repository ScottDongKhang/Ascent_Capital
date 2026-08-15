#!/usr/bin/env python
"""
Regenerate the live-book and data-integrity sections of CURRENT_VERIFIED_NUMBERS.md.

Why this exists
---------------
CURRENT_VERIFIED_NUMBERS.md declares itself the single source of truth and wins
any disagreement with other docs. It was hand-maintained, went five weeks stale,
and ended up losing to CLAUDE.md on its own tiebreak rule. Meanwhile six
different max-drawdown values for one book were in circulation at once.

A source of truth that requires discipline to stay true is not one. This script
recomputes the volatile sections from artifacts and rewrites them in place
between generated-block markers, so hand-written caveats survive.

    .venv/bin/python scripts/reconcile_numbers.py --check   # drift check, no write
    .venv/bin/python scripts/reconcile_numbers.py --write   # rewrite the doc
    .venv/bin/python scripts/reconcile_numbers.py           # print, no write

Rules this script follows, and you should too:
- Every figure carries its method and its source artifact.
- A figure that cannot be computed is reported as NOT COMPUTABLE with the
  reason. It is never omitted and never estimated.
- Where two methods disagree, both are shown. Silently picking one is how the
  drawdown got published six different ways.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def _markers(tag: str) -> tuple[str, str]:
    return (f"<!-- BEGIN GENERATED {tag}: reconcile_numbers.py -->",
            f"<!-- END GENERATED {tag} -->")


BEGIN_BOOK, END_BOOK = _markers("live-book")
BEGIN_DATA, END_DATA = _markers("data-integrity")

LIVE_START = date(2026, 4, 1)
HOLDINGS = ROOT / "logs" / "holdings_log.jsonl"
PRICES = ROOT / "data_cache" / "prices_live.parquet"


# ------------------------------------------------------------------ utils ----

def _fmt(v, pct=False, sign=False, nd=2):
    if v is None:
        return "NOT COMPUTABLE"
    s = f"{v * 100:.{nd}f}%" if pct else f"{v:,.{nd}f}"
    if sign and v > 0:
        s = "+" + s
    return s


def _load_holdings():
    if not HOLDINGS.exists():
        return []
    rows = []
    for line in HOLDINGS.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


# ------------------------------------------------------- live book block ----

def live_book() -> dict:
    """Recompute the live paper book from holdings_log.jsonl."""
    rows = _load_holdings()
    out = {"source": "logs/holdings_log.jsonl", "notes": []}
    if not rows:
        out["error"] = "holdings_log.jsonl missing or empty"
        return out

    live = [r for r in rows
            if r.get("date") and date.fromisoformat(r["date"]) >= LIVE_START
            and r.get("equity")]
    if len(live) < 2:
        out["error"] = f"only {len(live)} rows at/after {LIVE_START}"
        return out

    first, last = live[0], live[-1]
    eq = [float(r["equity"]) for r in live]

    out["window"] = f"{first['date']} -> {last['date']}"
    out["n_rows"] = len(live)
    out["equity"] = float(last["equity"])
    out["anchor_equity"] = float(first["equity"])
    out["anchor_date"] = first["date"]
    out["total_return"] = eq[-1] / eq[0] - 1

    # Max drawdown, equity-based. This is the number to quote; it is the only
    # one computed from the settled equity series rather than a daily-return
    # column known to contain fake zeros.
    peak = eq[0]
    dd = 0.0
    peak_date = first["date"]
    trough_date = first["date"]
    cur_peak_date = first["date"]
    for r, e in zip(live, eq):
        if e > peak:
            peak, cur_peak_date = e, r["date"]
        d = e / peak - 1
        if d < dd:
            dd, peak_date, trough_date = d, cur_peak_date, r["date"]
    out["max_drawdown"] = dd
    out["dd_peak_date"] = peak_date
    out["dd_trough_date"] = trough_date
    out["dd_peak_equity"] = peak

    out["n_positions"] = last.get("n_positions")

    # SPY over the same window, from the price cache rather than the log's
    # spy_return column: that column has gaps from the same late-settlement
    # problem that fakes 0.0 day returns, so its cumulative product understates
    # the index badly.
    out["spy_return"] = None
    try:
        import pandas as pd
        # Date keying is delegated to the store's own _calendar_day_key. The
        # `date` column holds a date-only value at UTC midnight, tz-localized to
        # New York, so bars are stamped 20:00 the PREVIOUS evening. Converting
        # to the New York calendar date shifts every bar back a day: it produces
        # zero Fridays and 307 phantom Sundays, and silently disagrees with the
        # dates the live pipeline reads.
        from ascent.data.store.parquet import _calendar_day_key
        df = pd.read_parquet(PRICES, columns=["date", "symbol", "close"])
        spy = df[df["symbol"] == "SPY"].copy()
        spy["cal"] = _calendar_day_key(spy["date"]).astype(str).str.slice(0, 10)
        spy = spy.dropna(subset=["cal"]).drop_duplicates("cal", keep="last").sort_values("cal")
        lo = first["date"]
        hi = last["date"]
        win = spy[(spy["cal"] >= lo) & (spy["cal"] <= hi)]
        if len(win) >= 2:
            out["spy_return"] = float(win["close"].iloc[-1] / win["close"].iloc[0] - 1)
            out["spy_window"] = f"{win['cal'].iloc[0]} -> {win['cal'].iloc[-1]}"
            out["spy_n"] = len(win)
        else:
            out["notes"].append("SPY: fewer than 2 cache rows in the window")
    except Exception as exc:
        out["notes"].append(f"SPY from price cache failed: {type(exc).__name__}: {exc}")

    if out["spy_return"] is not None:
        out["excess"] = out["total_return"] - out["spy_return"]

    # The log's own SPY column, shown alongside so the disagreement is visible
    # rather than resolved silently.
    try:
        cum = 1.0
        n = 0
        for r in live:
            sr = r.get("spy_return")
            if sr is not None:
                cum *= (1 + float(sr))
                n += 1
        out["spy_return_from_log"] = cum - 1
        out["spy_log_n"] = n
    except Exception:
        pass

    # Sharpe is deliberately NOT reported as a figure. At this sample the
    # standard error swamps the estimate, and the day_return column is known
    # unreliable. State that, do not print a number that will be quoted.
    fake_zeros = sum(1 for r in live if r.get("day_return") == 0.0)
    out["fake_zero_day_returns"] = fake_zeros
    return out


def render_live_book(d: dict) -> str:
    if d.get("error"):
        return (f"### 2a. Live paper book — NOT COMPUTABLE\n\n"
                f"`{d['source']}`: {d['error']}\n")
    L = []
    L.append("| Metric | Value | Method | Source |")
    L.append("|---|---|---|---|")
    L.append(f"| Window | {d['window']} ({d['n_rows']} logged rows) | rows at/after "
             f"{LIVE_START.isoformat()} | `{d['source']}` |")
    L.append(f"| Current equity | ${d['equity']:,.2f} | last logged row | `{d['source']}` |")
    L.append(f"| Total return | {_fmt(d['total_return'], pct=True, sign=True)} | "
             f"equity {d['anchor_equity']:,.0f} ({d['anchor_date']}) -> {d['equity']:,.0f} "
             f"| `{d['source']}` |")
    if d.get("spy_return") is not None:
        L.append(f"| SPY, same window | {_fmt(d['spy_return'], pct=True, sign=True)} | "
                 f"close-to-close, {d.get('spy_n')} bars, {d.get('spy_window')} | "
                 f"`data_cache/prices_live.parquet` |")
        L.append(f"| Book vs SPY | {_fmt(d['excess'], pct=True, sign=True)} | "
                 f"difference of the two above | derived |")
    else:
        L.append("| SPY, same window | NOT COMPUTABLE | see notes | - |")
    if d.get("spy_return_from_log") is not None:
        L.append(f"| SPY per the log's own column | "
                 f"{_fmt(d['spy_return_from_log'], pct=True, sign=True)} | "
                 f"cumulative `spy_return` over {d.get('spy_log_n')} rows | "
                 f"`{d['source']}` |")
    L.append(f"| Max drawdown | {_fmt(d['max_drawdown'], pct=True)} | equity-based, peak "
             f"{d['dd_peak_equity']:,.0f} ({d['dd_peak_date']}) -> trough "
             f"({d['dd_trough_date']}) | `{d['source']}` |")
    L.append(f"| Open positions | {d.get('n_positions', 'NOT COMPUTABLE')} | last logged row "
             f"| `{d['source']}` |")
    L.append("| Annualized Sharpe | NOT COMPUTABLE | see note below | - |")

    body = "\n".join(L)
    note = (
        f"\n**Sharpe is deliberately absent.** At {d['n_rows']} sessions the standard error "
        f"swamps the estimate, and {d['fake_zero_day_returns']} of those rows carry a "
        f"`day_return` of exactly 0.0 — the Alpaca late-settlement artifact, not flat days. "
        f"Any Sharpe computed from that column is meaningless. Do not reintroduce one here.\n"
    )
    if d.get("spy_return_from_log") is not None and d.get("spy_return") is not None:
        note += (
            f"\n**The two SPY figures disagree** ({_fmt(d['spy_return'], pct=True, sign=True)} "
            f"from the price cache vs {_fmt(d['spy_return_from_log'], pct=True, sign=True)} "
            f"from the log column). The cache figure is the one to quote: the log's "
            f"`spy_return` column inherits the same missing-day problem as `day_return`, so "
            f"its cumulative product understates the index. Both are shown so the gap stays "
            f"visible.\n"
        )
    if d.get("notes"):
        note += "\nComputation notes:\n" + "".join(f"- {n}\n" for n in d["notes"])
    return body + "\n" + note


# -------------------------------------------------- data integrity block ----

def data_integrity() -> dict:
    out = {"path": "data_cache/prices_live.parquet"}
    if not PRICES.exists():
        out["error"] = "prices_live.parquet not found"
        return out
    try:
        import pandas as pd
        df = pd.read_parquet(PRICES, columns=["date", "symbol", "source"]) \
            if "source" in pd.read_parquet(PRICES).columns else \
            pd.read_parquet(PRICES, columns=["date", "symbol"])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    from ascent.data.store.parquet import _calendar_day_key
    from datetime import time as _time

    out["rows"] = len(df)
    out["symbols"] = int(df["symbol"].nunique())
    # Same convention as the live read path -- see the note in live_book().
    cal = _calendar_day_key(df["date"]).astype(str).str.slice(0, 10)
    valid = cal[cal.notna() & (cal != "")]
    out["date_min"] = str(valid.min()) if len(valid) else None
    out["date_max"] = str(valid.max()) if len(valid) else None

    key = pd.DataFrame({"symbol": df["symbol"].values, "cal": cal.values})
    dup_mask = key.duplicated(keep="first")
    out["dup_rows"] = int(dup_mask.sum())
    out["dup_pct"] = out["dup_rows"] / max(1, out["rows"])
    out["dup_symbols"] = int(key[key.duplicated(keep=False)]["symbol"].nunique())

    # Phantom-row corruption signature (2026-08-15 audit): a bar written with
    # a non-midnight time-of-day slips past the calendar-day-collapsing dedup
    # above (that's exactly why this check reads `dup_rows: 0` even on a
    # fully phantom-corrupted cache -- the dedup key already collapses the
    # corruption away before counting). Mirrors the check in
    # ascent/data/store/parquet.py::validate_cache: handle a tz-aware column
    # by localizing to naive before comparing .dt.time, and gate on
    # is_datetime64_any_dtype so a mixed-dtype object column doesn't crash.
    non_midnight_rows = 0
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        naive_dates = dates
        if isinstance(naive_dates.dtype, pd.DatetimeTZDtype):
            naive_dates = naive_dates.dt.tz_localize(None)
        non_midnight_rows = int((naive_dates.dt.time != _time(0, 0)).sum())
    out["non_midnight_rows"] = non_midnight_rows
    out["non_midnight_pct"] = non_midnight_rows / max(1, out["rows"])
    if "source" in df.columns:
        out["sources"] = {str(k): int(v) for k, v in
                          df["source"].value_counts().head(6).items()}
    return out


def render_data_integrity(d: dict) -> str:
    if d.get("error"):
        return f"`{d['path']}`: **{d['error']}**\n"
    L = [f"`{d['path']}` as measured now:", ""]
    L.append("| Property | Value |")
    L.append("|---|---|")
    L.append(f"| Rows | {d['rows']:,} |")
    L.append(f"| Distinct symbols | {d['symbols']:,} |")
    L.append(f"| Date range | {d['date_min']} -> {d['date_max']} |")
    L.append(f"| Duplicate (symbol, market-calendar-day) rows | "
             f"**{d['dup_rows']:,}** ({d['dup_pct'] * 100:.1f}% of rows), "
             f"across {d['dup_symbols']:,} symbols |")
    if "non_midnight_rows" in d:
        L.append(f"| Non-midnight time-of-day rows (phantom-row signature) | "
                 f"**{d['non_midnight_rows']:,}** ({d['non_midnight_pct'] * 100:.1f}% of rows) |")
    if d.get("sources"):
        srcs = ", ".join(f"`{k}` {v:,}" for k, v in d["sources"].items())
        L.append(f"| Source generations present | {srcs} |")
    L.append("")
    if d["dup_rows"] > 0:
        L.append(
            f"**Duplicates are present.** The ingest dedup in `save_parquet` and the "
            f"read-side `~index.duplicated(keep=\"last\")` in `ascent/main.py` are not "
            f"keeping this cache clean; a past one-time collapse to zero duplicates has not "
            f"held. The live pipeline dedupes on read and is shielded, but the walk-forward "
            f"framework does **not**, so any fresh backtest on this cache is suspect until "
            f"this is zero.")
    else:
        L.append("**No duplicates.** Any backtest run on this cache is clean on that axis.")
    if d.get("sources") and len(d["sources"]) > 1:
        L.append("")
        L.append("More than one source generation is blended in this cache. Mixed adjustment "
                 "bases (split-only vs total-return) across generations produce fake jumps, "
                 "which is what drove a 70% alpha sleeve to zero once already.")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------- assembly ----

_STAMP = ("*Regenerated by `scripts/reconcile_numbers.py`. Do not hand-edit between the "
          "markers; edits are overwritten. Last regenerated: {today}.*")


def build_book_block() -> str:
    from ascent.reporting.verified_numbers import canonical_wf, MissingArtifact

    parts = [BEGIN_BOOK, "", _STAMP.format(today=date.today().isoformat()), ""]
    parts.append("## 2. Live paper-trading book (SYSTEM 1 + SYSTEM 2 blended actual account)")
    parts.append("")
    parts.append(render_live_book(live_book()))
    parts.append("")
    parts.append("### Walk-forward artifact cross-check")
    parts.append("")
    try:
        wf = canonical_wf()
        parts.append(f"Section 1 above must match `{wf.artifact}`:")
        parts.append("")
        parts.append(f"    {wf.citation()}")
        parts.append("")
        for c in wf.caveats:
            parts.append(f"- {c}")
    except MissingArtifact as exc:
        parts.append(f"**Canonical walk-forward artifact unreadable: {exc}**")
    parts.append("")
    parts.append(END_BOOK)
    return "\n".join(parts)


def build_data_block() -> str:
    parts = [BEGIN_DATA, "", _STAMP.format(today=date.today().isoformat()), ""]
    parts.append("## 4. Data integrity status")
    parts.append("")
    parts.append(render_data_integrity(data_integrity()))
    parts.append(END_DATA)
    return "\n".join(parts)


def _splice(txt: str, begin: str, end: str, block: str,
            legacy_from: str, legacy_to: str) -> str:
    """Replace an existing generated block, or the legacy section on first run.

    Bounded by the NEXT section heading so neighbouring hand-written sections
    (section 3, the AI-native layer) are never swallowed.
    """
    if begin in txt and end in txt:
        return txt[: txt.index(begin)] + block + txt[txt.index(end) + len(end):]
    if legacy_from in txt and legacy_to in txt:
        i, j = txt.index(legacy_from), txt.index(legacy_to)
        if i < j:
            return txt[:i] + block + "\n\n---\n" + txt[j:]
    return txt.rstrip() + "\n\n---\n\n" + block + "\n"


def render_doc(txt: str) -> str:
    txt = _splice(txt, BEGIN_BOOK, END_BOOK, build_book_block(),
                  "\n## 2. ", "\n## 3. ")
    txt = _splice(txt, BEGIN_DATA, END_DATA, build_data_block(),
                  "\n## 4. ", "\n## 5. ")
    return txt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite CURRENT_VERIFIED_NUMBERS.md")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the doc's generated blocks are out of date")
    args = ap.parse_args()

    doc = ROOT / "CURRENT_VERIFIED_NUMBERS.md"
    txt = doc.read_text()
    new = render_doc(txt)

    if args.check:
        if new != txt:
            print("CURRENT_VERIFIED_NUMBERS.md is out of date. Run with --write.")
            return 1
        print("CURRENT_VERIFIED_NUMBERS.md is current.")
        return 0

    if args.write:
        doc.write_text(new)
        print(f"rewrote {doc.relative_to(ROOT)}")
        return 0

    print(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
