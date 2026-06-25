# ascent/strategy/falsifier_registry.py
"""
Falsifier enforcement layer — the AI layer's own "what would prove me wrong"
conditions, checked daily and acted on.

The system already collects falsifiable conditions everywhere and acts on none
of them: prethesis `what_would_change_my_mind` per conviction name, causal
graph falsification conditions (Gate 4 flags them into a log nobody reads),
judge 10-day predictions, and the AI PM pre-mortem. This module:

  1. build_registry()      — on rebalance day, structures those conditions into
                             data_cache/active_falsifiers.json. Price/macro
                             conditions are structured by ONE Haiku call;
                             anything unparseable becomes a news watch.
  2. add_judge_falsifier() — registers the judge's applied position change as a
                             relative-performance condition (symbol vs SPY).
  3. check_all()           — daily: price/macro conditions evaluated IN CODE
                             from the parquet caches (no LLM); news watches
                             evaluated with ONE Haiku call against the day's
                             headlines; causal early-exit flags folded in.

Fired falsifiers are returned to run_all_agents.py, which executes a bounded
25% trim (floor 4%) through the mini-rebalance path, records a
`falsifier_trim` intervention scored at 10 days, and shares the 5-trading-day
mini-rebalance cooldown. One trim per symbol per holding period.

Every external call is failure-safe: no LLM → news watches simply don't fire;
no prices → price conditions don't fire. Never raises.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_REPO         = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = _REPO / "data_cache" / "active_falsifiers.json"
_EXPIRY_DAYS  = 14  # calendar ≈ one 10-business-day holding period

# Relative-performance threshold for judge predictions: position must not lag
# SPY by more than this since the intervention date.
_JUDGE_REL_THRESHOLD = -0.03


def _parse_json_objects(text: str) -> list:
    """Tolerantly extract top-level JSON objects from an LLM response.

    Both Haiku calls here used json.loads(text[start:end]) on the whole array, so
    a single missing comma OR a response truncated at max_tokens discarded EVERY
    object and silently dropped the registry to news watches (2026-06-24 run:
    "Expecting ',' delimiter: line 99 column 73"). This scans for balanced
    top-level {...} spans and parses each independently: a malformed object is
    skipped, a truncated trailing object is dropped, and the well-formed ones
    survive. The surrounding code already keys results by "i", so per-item loss
    degrades gracefully (missing index → its own fallback).
    """
    objs: list = []
    depth = 0
    start: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objs.append(json.loads(text[start:i + 1]))
                    except Exception:
                        pass
                    start = None
    return objs


# ── Registry I/O ───────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {"as_of": None, "falsifiers": []}


def _save_registry(reg: dict) -> None:
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2))
        tmp.rename(REGISTRY_PATH)
    except Exception as exc:
        log.warning("[Falsifier] Registry save failed: %s", exc)


# ── Build (rebalance day) ──────────────────────────────────────────────────────

def build_registry(
    today: date,
    prethesis_raw: Optional[dict] = None,
    thesis: Optional[dict] = None,
) -> int:
    """
    Build a fresh registry from the rebalance's falsifiable conditions.
    Replaces the previous registry (old holding period's falsifiers expire
    with their rebalance). Returns the number of registered falsifiers.
    """
    entries: list = []
    expires = (today + timedelta(days=_EXPIRY_DAYS)).isoformat()

    raw_items: list = []  # (symbol, text, source)
    for n in (prethesis_raw or {}).get("high_conviction_names", []) or []:
        sym, txt = n.get("symbol", ""), n.get("what_would_change_my_mind", "")
        if sym and txt:
            raw_items.append((sym, txt, "prethesis"))

    pre_mortem = (thesis or {}).get("pre_mortem", "")
    if pre_mortem:
        raw_items.append(("__PORTFOLIO__", str(pre_mortem)[:600], "pre_mortem"))

    structured = _structure_with_haiku(raw_items, today) if raw_items else []
    for s in structured:
        s["expires"] = expires
        s["fired"] = False
        s["trimmed"] = False
        entries.append(s)

    reg = {"as_of": today.isoformat(), "falsifiers": entries}
    _save_registry(reg)
    log.info("[Falsifier] Registry built: %d falsifiers (%d sources)",
             len(entries), len(raw_items))
    return len(entries)


def add_judge_falsifier(symbol: str, prediction: str, today: date) -> None:
    """
    Register the judge's applied position change as a relative-performance
    falsifier: if the symbol lags SPY by >3pp since the intervention, the
    judge's thesis for holding it at the reduced weight is in question.
    """
    try:
        reg = _load_registry()
        reg["falsifiers"].append({
            "id": f"judge-{symbol}-{today.isoformat()}",
            "symbol": symbol,
            "source": "judge",
            "kind": "relative_price",
            "condition": {"vs": "SPY", "op": "<", "value": _JUDGE_REL_THRESHOLD,
                          "since": today.isoformat()},
            "raw_text": (prediction or "")[:300],
            "expires": (today + timedelta(days=_EXPIRY_DAYS)).isoformat(),
            "fired": False,
            "trimmed": False,
        })
        _save_registry(reg)
    except Exception as exc:
        log.warning("[Falsifier] add_judge_falsifier failed: %s", exc)


def _structure_with_haiku(raw_items: list, today: date) -> list:
    """
    ONE Haiku call: convert freeform falsifier sentences into structured
    conditions. Unparseable / non-market-observable items become news watches.
    On any LLM failure, everything falls back to a news watch (still useful —
    checked daily against headlines).
    """
    fallback = [
        {"id": f"{src}-{sym}-{i}", "symbol": sym, "source": src,
         "kind": "news", "condition": {"keywords": []}, "raw_text": txt[:300]}
        for i, (sym, txt, src) in enumerate(raw_items)
    ]
    try:
        from ascent.llm.client import HAIKU_MODEL
        import anthropic
        client = anthropic.Anthropic()

        items_str = "\n".join(
            f"{i}. [{sym}] {txt[:250]}" for i, (sym, txt, _) in enumerate(raw_items)
        )
        prompt = f"""Convert each falsifier sentence into a structured, checkable condition.
Today is {today.isoformat()}.

FALSIFIERS:
{items_str}

For each item return one JSON object:
- Price-checkable (e.g. "stock drops 8%"; convert price levels to a % return from today):
  {{"i": <index>, "kind": "price", "metric": "ret_since_rebalance", "op": "<", "value": -0.08}}
- Macro-checkable (VIX level, 10Y yield, HY spread):
  {{"i": <index>, "kind": "macro", "metric": "vix|t10y|hy_spread", "op": ">", "value": 30}}
- Anything else (earnings miss, guidance cut, contract loss, narrative events):
  {{"i": <index>, "kind": "news", "keywords": ["2-4 short keywords"]}}

Return ONLY a JSON array, no other text."""

        resp = client.messages.create(
            model=HAIKU_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip() if resp.content else "[]"
        parsed = _parse_json_objects(text)

        by_index = {int(p["i"]): p for p in parsed if isinstance(p, dict) and "i" in p}
        out = []
        for i, (sym, txt, src) in enumerate(raw_items):
            p = by_index.get(i)
            entry = {"id": f"{src}-{sym}-{i}", "symbol": sym, "source": src,
                     "raw_text": txt[:300]}
            if p and p.get("kind") == "price" and isinstance(p.get("value"), (int, float)):
                entry["kind"] = "price"
                entry["condition"] = {"metric": "ret_since_rebalance",
                                      "op": str(p.get("op", "<")),
                                      "value": float(p["value"]),
                                      "since": today.isoformat()}
            elif p and p.get("kind") == "macro" and isinstance(p.get("value"), (int, float)):
                entry["kind"] = "macro"
                entry["condition"] = {"metric": str(p.get("metric", "vix")),
                                      "op": str(p.get("op", ">")),
                                      "value": float(p["value"])}
            else:
                entry["kind"] = "news"
                entry["condition"] = {"keywords": (p or {}).get("keywords", [])[:4]}
            out.append(entry)
        return out
    except Exception as exc:
        log.warning("[Falsifier] Haiku structuring failed (%s) — using news fallback", exc)
        return fallback


# ── Daily evaluation ───────────────────────────────────────────────────────────

def _load_close_panel():
    try:
        import pandas as pd
        p = _REPO / "data_cache" / "prices_live.parquet"
        if not p.exists():
            return None
        raw = pd.read_parquet(p)
        if "symbol" in raw.columns and "close" in raw.columns:
            panel = raw.pivot_table(index="date", columns="symbol", values="close",
                                    aggfunc="last")
            panel.index = pd.to_datetime(panel.index)
            if getattr(panel.index, "tz", None) is not None:
                panel.index = panel.index.tz_localize(None)
            return panel.sort_index()
        return raw
    except Exception:
        return None


def _ret_since(panel, symbol: str, since: str) -> Optional[float]:
    try:
        import pandas as pd
        if panel is None or symbol not in panel.columns:
            return None
        col = panel[symbol].dropna()
        window = col[col.index >= pd.Timestamp(since)]
        if len(window) < 2:
            return None
        return float(window.iloc[-1] / window.iloc[0] - 1)
    except Exception:
        return None


def _latest_macro(metric: str) -> Optional[float]:
    _SERIES = {"vix": ["VIXCLS", "vix"], "t10y": ["DGS10", "treasury_10y"],
               "hy_spread": ["BAMLH0A0HYM2"], "t10y2y": ["T10Y2Y"]}
    try:
        import pandas as pd
        p = _REPO / "data_cache" / "macro_live.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        sub = df[df["series_id"].isin(_SERIES.get(metric, [metric]))]
        if sub.empty:
            return None
        return float(sub.sort_values("date")["value"].dropna().iloc[-1])
    except Exception:
        return None


def _compare(value: float, op: str, threshold: float) -> bool:
    return value < threshold if op == "<" else value > threshold


def check_all(today: date, news_context: Optional[dict] = None) -> list:
    """
    Evaluate all active falsifiers. Returns the list of NEWLY fired entries
    (fired this call, not previously fired, not yet trimmed). Persists fired
    flags so a falsifier fires at most once.
    """
    reg = _load_registry()
    active = [f for f in reg.get("falsifiers", [])
              if not f.get("fired") and f.get("expires", "9999") >= today.isoformat()]
    if not active and not _causal_early_exits():
        return []

    panel = _load_close_panel()
    fired: list = []
    news_batch: list = []

    for f in active:
        kind, cond = f.get("kind"), f.get("condition", {})
        try:
            if kind == "price" and f.get("symbol") not in ("", "__PORTFOLIO__"):
                r = _ret_since(panel, f["symbol"], cond.get("since", reg.get("as_of", "")))
                if r is not None and _compare(r, cond.get("op", "<"), cond.get("value", -1)):
                    f["fired_value"] = round(r, 4)
                    fired.append(f)
            elif kind == "relative_price":
                r_sym = _ret_since(panel, f["symbol"], cond.get("since", ""))
                r_spy = _ret_since(panel, cond.get("vs", "SPY"), cond.get("since", ""))
                if r_sym is not None and r_spy is not None:
                    rel = r_sym - r_spy
                    if _compare(rel, cond.get("op", "<"), cond.get("value", -1)):
                        f["fired_value"] = round(rel, 4)
                        fired.append(f)
            elif kind == "macro":
                v = _latest_macro(cond.get("metric", "vix"))
                if v is not None and _compare(v, cond.get("op", ">"), cond.get("value", 1e9)):
                    f["fired_value"] = v
                    fired.append(f)
            elif kind == "news" and news_context:
                news_batch.append(f)
        except Exception:
            continue

    if news_batch and news_context:
        for f in _check_news_batch(news_batch, news_context):
            fired.append(f)

    # Causal early exits (Gate 4) fold in as falsifiers
    for sym in _causal_early_exits():
        already = any(f.get("symbol") == sym and f.get("source") == "causal"
                      for f in reg.get("falsifiers", []))
        entry = {"id": f"causal-{sym}-{today.isoformat()}", "symbol": sym,
                 "source": "causal", "kind": "causal",
                 "raw_text": "causal mechanism broken (early-exit threshold breached)",
                 "expires": (today + timedelta(days=_EXPIRY_DAYS)).isoformat(),
                 "fired": False, "trimmed": False}
        if not already:
            reg["falsifiers"].append(entry)
            fired.append(entry)

    for f in fired:
        f["fired"] = True
        f["fired_date"] = today.isoformat()

    _save_registry(reg)
    if fired:
        log.info("[Falsifier] %d falsifier(s) fired: %s",
                 len(fired), [(f.get("symbol"), f.get("source")) for f in fired])
    return fired


def _causal_early_exits() -> list:
    try:
        from ascent.causal.tracker import check_early_exits
        return check_early_exits() or []
    except Exception:
        return []


def _check_news_batch(batch: list, news_context: dict) -> list:
    """ONE Haiku call: did today's headlines satisfy any news falsifier?"""
    try:
        from ascent.llm.client import HAIKU_MODEL
        import anthropic
        client = anthropic.Anthropic()

        items = []
        for i, f in enumerate(batch):
            sym = f.get("symbol", "")
            heads = news_context.get(sym, []) if sym != "__PORTFOLIO__" else [
                h for hs in news_context.values() for h in hs][:15]
            if not heads:
                continue
            items.append((i, f, heads))
        if not items:
            return []

        block = "\n\n".join(
            f"{i}. CONDITION [{f.get('symbol')}]: {f.get('raw_text','')[:200]}\n"
            "   HEADLINES:\n" + "\n".join(f"   - {h[:120]}" for h in heads[:6])
            for i, f, heads in items
        )
        prompt = (
            "For each numbered condition, decide if today's headlines clearly satisfy it. "
            "Be conservative: only YES if a headline directly confirms the condition.\n\n"
            f"{block}\n\n"
            'Return ONLY a JSON array like [{"i": 0, "fired": true, "evidence": "headline snippet"}] '
            "containing every index."
        )
        resp = client.messages.create(
            model=HAIKU_MODEL, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip() if resp.content else "[]"
        parsed = _parse_json_objects(text)
        fired = []
        idx_map = {i: f for i, f, _ in items}
        for p in parsed:
            if isinstance(p, dict) and p.get("fired") and int(p.get("i", -1)) in idx_map:
                f = idx_map[int(p["i"])]
                f["fired_evidence"] = str(p.get("evidence", ""))[:200]
                fired.append(f)
        return fired
    except Exception as exc:
        log.debug("[Falsifier] News batch check failed: %s", exc)
        return []


def mark_trimmed(falsifier_id: str) -> None:
    """Record that a trim was executed for this falsifier (one trim per entry)."""
    reg = _load_registry()
    for f in reg.get("falsifiers", []):
        if f.get("id") == falsifier_id:
            f["trimmed"] = True
    _save_registry(reg)
