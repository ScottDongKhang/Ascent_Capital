# Plan A — Monitoring & Attribution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every daily run logs actual Alpaca holdings, portfolio return vs SPY, and position-level attribution — the foundation everything else depends on.

**Architecture:** Three additions to `forward_pnl_tracker.py` and `run_all_agents.py`. No new modules except `ascent/monitoring/attribution.py`. Outcome tracker's NAV source is fixed. Holdings log already exists (added Apr 16) — extend it with SPY and benchmark comparison.

**Tech Stack:** yfinance, pandas, existing `ascent/execution/alpaca_broker.py`, existing `logs/` structure.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `ascent/monitoring/forward_pnl_tracker.py` | Add SPY benchmark return to every PnL entry; fix US equities routing |
| Create | `ascent/monitoring/attribution.py` | Daily position-level P&L attribution + factor exposure summary |
| Modify | `run_all_agents.py` | Call `run_attribution()` after `_log_holdings()` |
| Modify | `logs/holdings_log.jsonl` schema | Add `spy_return`, `alpha_vs_spy` fields (written by `_log_holdings`) |

---

## Task A1: Add SPY benchmark to every PnL entry

**Problem:** `forward_pnl_tracker.py` batch-fetches agent symbols but never fetches SPY. The holdings log writes `day_return` but not `spy_return`. Can't track alpha without a benchmark.

**Files:**
- Modify: `ascent/monitoring/forward_pnl_tracker.py:_fetch_latest_returns`
- Modify: `run_all_agents.py:_log_holdings`

- [ ] **Step 1: Write failing test**

```python
# tests/test_plan_a.py
import json
from pathlib import Path
from datetime import date
from unittest.mock import patch

def test_pnl_entry_includes_spy():
    """Every PnL log entry must include spy_return and alpha fields."""
    from ascent.monitoring.forward_pnl_tracker import _fetch_latest_returns

    mock_returns = {"AAPL": 0.01, "MSFT": 0.02, "SPY": 0.005}
    with patch("yfinance.download") as mock_dl:
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2026-04-14", periods=2, freq="B")
        data = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 204.0], "SPY": [500.0, 502.5]},
            index=dates,
        )
        mock_dl.return_value = pd.concat({"Close": data}, axis=1)
        result = _fetch_latest_returns(["AAPL", "MSFT", "SPY"])

    assert "SPY" in result, "SPY must be in returned dict"
    assert abs(result["SPY"] - 0.005) < 0.0001

def test_holdings_log_has_benchmark(tmp_path, monkeypatch):
    """_log_holdings must write spy_return and alpha_vs_spy."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    from unittest.mock import MagicMock, patch
    import pandas as pd

    mock_pos = pd.DataFrame({
        "symbol": ["AAPL"], "qty": [10.0],
        "market_value": [1000.0], "current_price": [100.0], "weight": [1.0]
    })
    mock_acct = {"equity": "101000", "last_equity": "100000", "cash": "0"}

    with patch("ascent.execution.alpaca_broker.get_positions", return_value=mock_pos), \
         patch("ascent.execution.alpaca_broker.get_account", return_value=mock_acct), \
         patch("yfinance.download") as mock_dl:

        dates = pd.date_range("2026-04-14", periods=2, freq="B")
        spy_data = pd.DataFrame({"SPY": [500.0, 502.5]}, index=dates)
        mock_dl.return_value = pd.concat({"Close": spy_data}, axis=1)

        from run_all_agents import _log_holdings
        _log_holdings(date(2026, 4, 16))

    entry = json.loads((tmp_path / "logs" / "holdings_log.jsonl").read_text().strip())
    assert "spy_return" in entry, "holdings_log must have spy_return"
    assert "alpha_vs_spy" in entry, "holdings_log must have alpha_vs_spy"
    assert abs(entry["spy_return"] - 0.005) < 0.001
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_plan_a.py::test_pnl_entry_includes_spy tests/test_plan_a.py::test_holdings_log_has_benchmark -v
```
Expected: FAIL — `SPY` not in result, `spy_return` not in holdings entry.

- [ ] **Step 3: Add SPY to batch fetch in `forward_pnl_tracker.py`**

In `run_forward_pnl_cycle`, add SPY to the batch symbols list:

```python
# In run_forward_pnl_cycle, after building all_symbols:
all_symbols.add("SPY")  # always fetch SPY for benchmark

batch_returns = {}
if all_symbols:
    batch_returns = _fetch_latest_returns(list(all_symbols))
    spy_return = batch_returns.get("SPY", 0.0)
    print(f"[ForwardPnL] SPY benchmark: {spy_return:+.2%}")
```

Also return `spy_return` from the function:

```python
# At end of run_forward_pnl_cycle, change return to:
return {"agent_returns": results, "spy_return": batch_returns.get("SPY", 0.0)}
```

- [ ] **Step 4: Add `spy_return` and `alpha_vs_spy` to `_log_holdings` in `run_all_agents.py`**

```python
def _log_holdings(today):
    log_path = Path("logs/holdings_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ascent.execution.alpaca_broker import get_positions, get_account
        pos = get_positions()
        acct = get_account()
        equity    = float(acct.get("equity", 0))
        last_eq   = float(acct.get("last_equity", equity) or equity)
        day_ret   = (equity / last_eq - 1) if last_eq else 0.0

        # Fetch SPY benchmark
        spy_ret = 0.0
        try:
            import yfinance as yf
            spy_data = yf.download("SPY", period="2d", interval="1d",
                                   progress=False, auto_adjust=True)
            if len(spy_data) >= 2:
                spy_ret = float(spy_data["Close"].pct_change().iloc[-1])
        except Exception:
            pass

        positions = []
        if not pos.empty:
            for _, row in pos.sort_values("market_value", ascending=False).iterrows():
                positions.append({
                    "symbol":        row["symbol"],
                    "qty":           round(float(row["qty"]), 4),
                    "market_value":  round(float(row["market_value"]), 2),
                    "current_price": round(float(row["current_price"]), 4),
                    "weight":        round(float(row["weight"]), 4),
                })

        entry = {
            "date":           today.isoformat(),
            "timestamp":      datetime.now().isoformat(),
            "equity":         round(equity, 2),
            "cash":           round(float(acct.get("cash", 0)), 2),
            "day_return":     round(day_ret, 6),
            "spy_return":     round(spy_ret, 6),
            "alpha_vs_spy":   round(day_ret - spy_ret, 6),
            "n_positions":    len(positions),
            "positions":      positions,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        sign = "+" if day_ret >= spy_ret else "-"
        print(f"[Runner] Holdings logged — equity ${equity:,.2f} | "
              f"portfolio {day_ret:+.2%} vs SPY {spy_ret:+.2%} ({sign})")
    except Exception as e:
        print(f"[Runner] Holdings log skipped ({e})")
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_plan_a.py::test_pnl_entry_includes_spy tests/test_plan_a.py::test_holdings_log_has_benchmark -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ascent/monitoring/forward_pnl_tracker.py run_all_agents.py tests/test_plan_a.py
git commit -m "feat(monitoring): add SPY benchmark to PnL log and holdings snapshot"
```

---

## Task A2: Fix US equities PnL routing

**Problem:** `PNL_LOGS["us_equities"]` points to `logs/eod_log.jsonl`, which has a different schema (old `portfolio_value` key, mixed run events). The skill tracker can't compute a clean Sharpe from it. US equities needs its own clean log: `logs/us_equities_pnl.jsonl`.

**Files:**
- Modify: `ascent/monitoring/forward_pnl_tracker.py:PNL_LOGS`
- Modify: `ascent/monitoring/skill_tracker.py` (if it reads `eod_log.jsonl` directly)

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_a.py
def test_us_equities_pnl_has_own_log(tmp_path, monkeypatch):
    """US equities PnL must go to us_equities_pnl.jsonl, not eod_log.jsonl."""
    from ascent.monitoring.forward_pnl_tracker import PNL_LOGS
    assert str(PNL_LOGS["us_equities"]) == "logs/us_equities_pnl.jsonl", \
        "us_equities must use its own PnL log, not eod_log.jsonl"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_plan_a.py::test_us_equities_pnl_has_own_log -v
```
Expected: FAIL — currently points to `logs/eod_log.jsonl`.

- [ ] **Step 3: Update `PNL_LOGS` in `forward_pnl_tracker.py`**

```python
PNL_LOGS = {
    "us_equities":   Path("logs/us_equities_pnl.jsonl"),  # was eod_log.jsonl
    "macro":         Path("logs/macro_pnl.jsonl"),
    "international": Path("logs/international_pnl.jsonl"),
    "alternatives":  Path("logs/alternatives_pnl.jsonl"),
}
```

- [ ] **Step 4: Run test**

```bash
.venv/bin/pytest tests/test_plan_a.py::test_us_equities_pnl_has_own_log -v
```
Expected: PASS

- [ ] **Step 5: Verify skill tracker reads all pnl files, not just eod_log**

```bash
grep -n "eod_log\|us_equities_pnl" ascent/monitoring/skill_tracker.py
```

If `skill_tracker.py` still reads `eod_log.jsonl` for us_equities, update it to read `PNL_LOGS` from `forward_pnl_tracker`:

```python
# In skill_tracker.py, replace hardcoded paths with:
from ascent.monitoring.forward_pnl_tracker import PNL_LOGS
# Then use PNL_LOGS[agent_id] instead of hardcoded paths
```

- [ ] **Step 6: Commit**

```bash
git add ascent/monitoring/forward_pnl_tracker.py ascent/monitoring/skill_tracker.py tests/test_plan_a.py
git commit -m "fix(monitoring): route us_equities PnL to its own log file"
```

---

## Task A3: Daily attribution report

**Problem:** No daily breakdown of what drove portfolio P&L. Running a fund blind.

**Files:**
- Create: `ascent/monitoring/attribution.py`
- Modify: `run_all_agents.py` — call `run_attribution()` at end of `_log_holdings`
- New log: `logs/attribution_log.jsonl`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_a.py
def test_attribution_produces_report(tmp_path, monkeypatch):
    """attribution report must return top contributors, drags, and alpha vs SPY."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    from unittest.mock import patch
    import pandas as pd
    from datetime import date

    positions = [
        {"symbol": "MPWR", "weight": 0.064, "market_value": 6222.0, "current_price": 1402.0, "qty": 4.4},
        {"symbol": "EWY",  "weight": 0.112, "market_value": 10968.0, "current_price": 147.0, "qty": 74.4},
        {"symbol": "MRK",  "weight": 0.024, "market_value": 2385.0,  "current_price": 115.0, "qty": 20.6},
    ]
    mock_returns = {"MPWR": 0.037, "EWY": 0.018, "MRK": -0.021, "SPY": 0.003}

    with patch("yfinance.download") as mock_dl:
        dates = pd.date_range("2026-04-14", periods=2, freq="B")
        price_data = {s: [100.0, 100.0*(1+r)] for s, r in mock_returns.items()}
        df = pd.DataFrame(price_data, index=dates)
        mock_dl.return_value = pd.concat({"Close": df}, axis=1)

        from ascent.monitoring.attribution import run_attribution
        report = run_attribution(positions, date(2026, 4, 16))

    assert "portfolio_return" in report
    assert "spy_return" in report
    assert "alpha_vs_spy" in report
    assert "top_contributors" in report
    assert "top_drags" in report
    assert len(report["top_contributors"]) >= 1
    assert len(report["top_drags"]) >= 1
    # MPWR should be top contributor
    assert report["top_contributors"][0]["symbol"] == "MPWR"
    # MRK should be top drag
    assert report["top_drags"][0]["symbol"] == "MRK"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_plan_a.py::test_attribution_produces_report -v
```
Expected: FAIL — `ascent/monitoring/attribution.py` doesn't exist.

- [ ] **Step 3: Create `ascent/monitoring/attribution.py`**

```python
"""
ascent/monitoring/attribution.py
Daily position-level P&L attribution.

Computes:
  - Per-position contribution (weight × return)
  - Portfolio total return
  - SPY benchmark return
  - Alpha vs SPY
  - Top 5 contributors and drags

Called by run_all_agents._log_holdings() after each run.
Output appended to logs/attribution_log.jsonl.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict

ATTRIBUTION_LOG = Path("logs/attribution_log.jsonl")


def _fetch_returns(symbols: List[str]) -> Dict[str, float]:
    """Fetch latest close-to-close returns for given symbols + SPY."""
    syms = list(set(symbols) | {"SPY"})
    try:
        import yfinance as yf
        data = yf.download(syms, period="2d", interval="1d",
                           progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            return {}
        closes = data["Close"] if hasattr(data.columns, "levels") else data
        rets = closes.pct_change().iloc[-1]
        return {s: float(rets[s]) for s in syms if s in rets and not __import__("math").isnan(rets[s])}
    except Exception as e:
        print(f"[Attribution] Price fetch failed: {e}")
        return {}


def run_attribution(positions: List[Dict], run_date: date) -> Dict:
    """
    Compute daily attribution from a list of position dicts.

    Each position dict must have: symbol, weight, market_value.

    Returns attribution dict and appends to logs/attribution_log.jsonl.
    """
    symbols = [p["symbol"] for p in positions]
    returns = _fetch_returns(symbols)

    if not returns:
        print("[Attribution] No returns fetched — skipping")
        return {}

    spy_ret = returns.get("SPY", 0.0)

    # Per-position attribution
    contribs = []
    for pos in positions:
        sym = pos["symbol"]
        w   = float(pos.get("weight", 0))
        r   = returns.get(sym, 0.0)
        contrib = w * r
        contribs.append({
            "symbol":      sym,
            "weight":      round(w, 4),
            "return":      round(r, 6),
            "contribution": round(contrib, 6),
        })

    port_ret = sum(c["contribution"] for c in contribs)
    alpha    = port_ret - spy_ret

    contribs_sorted = sorted(contribs, key=lambda x: -x["contribution"])
    top_contributors = [c for c in contribs_sorted if c["contribution"] > 0][:5]
    top_drags        = [c for c in reversed(contribs_sorted) if c["contribution"] < 0][:5]

    report = {
        "date":              run_date.isoformat(),
        "timestamp":         datetime.now().isoformat(),
        "portfolio_return":  round(port_ret, 6),
        "spy_return":        round(spy_ret, 6),
        "alpha_vs_spy":      round(alpha, 6),
        "n_positions":       len(positions),
        "top_contributors":  top_contributors,
        "top_drags":         top_drags,
        "all_positions":     contribs_sorted,
    }

    ATTRIBUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTRIBUTION_LOG, "a") as f:
        f.write(json.dumps(report) + "\n")

    # Print summary
    sign = "▲" if alpha >= 0 else "▼"
    print(f"[Attribution] {run_date} | portfolio {port_ret:+.2%} vs SPY {spy_ret:+.2%} "
          f"({sign} {alpha:+.2%} alpha)")
    if top_contributors:
        best = top_contributors[0]
        print(f"  Best: {best['symbol']} +{best['contribution']:.3%} "
              f"(wt {best['weight']:.1%} × ret {best['return']:+.1%})")
    if top_drags:
        worst = top_drags[0]
        print(f"  Worst: {worst['symbol']} {worst['contribution']:.3%} "
              f"(wt {worst['weight']:.1%} × ret {worst['return']:+.1%})")

    return report
```

- [ ] **Step 4: Wire into `run_all_agents._log_holdings`**

At the end of `_log_holdings` in `run_all_agents.py`, after writing the entry:

```python
        # Run attribution report
        if positions:
            try:
                from ascent.monitoring.attribution import run_attribution
                run_attribution(positions, today)
            except Exception as e:
                print(f"[Runner] Attribution failed ({e})")
```

- [ ] **Step 5: Run test**

```bash
.venv/bin/pytest tests/test_plan_a.py::test_attribution_produces_report -v
```
Expected: PASS

- [ ] **Step 6: Run all Plan A tests**

```bash
.venv/bin/pytest tests/test_plan_a.py -v
```
Expected: All PASS

- [ ] **Step 7: Smoke test end-to-end**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from datetime import date
from run_all_agents import _log_holdings
_log_holdings(date.today())
import json
e = json.loads(open('logs/holdings_log.jsonl').readlines()[-1])
print('spy_return:', e.get('spy_return'))
print('alpha_vs_spy:', e.get('alpha_vs_spy'))
a = json.loads(open('logs/attribution_log.jsonl').readlines()[-1])
print('top contributor:', a['top_contributors'][0] if a['top_contributors'] else 'none')
print('top drag:', a['top_drags'][0] if a['top_drags'] else 'none')
"
```

- [ ] **Step 8: Commit**

```bash
git add ascent/monitoring/attribution.py run_all_agents.py tests/test_plan_a.py
git commit -m "feat(monitoring): daily attribution report with position-level P&L and SPY alpha"
```
