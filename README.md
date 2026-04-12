# Ascent Capital

**AI-Driven Systematic Quant Research & Trading Platform**

A portfolio-native quantitative research platform built with institutional-grade practices: point-in-time data safety, walk-forward evaluation, realistic transaction costs, and modular architecture.

## Quick Start

```bash
# Install dependencies (use venv if system pip has permission issues)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run full pipeline with simulated data (no API keys needed)
.venv/bin/python -m ascent.main

# Run with live data (requires API keys)
export POLYGON_API_KEY=your_key
export FRED_API_KEY=your_key
.venv/bin/python -m ascent.main --live

# Run tests
.venv/bin/pytest tests/ -v

# Customized backtest
.venv/bin/python -m ascent.main --start 2021-01-01 --end 2024-12-31 --top-n 15 --rebalance 21
```

## Architecture

```
ascent/
├── config/          # Settings, API keys, parameters
├── data/
│   ├── ingest/      # Polygon, Tiingo, FRED, simulated data
│   ├── normalize/   # Cleaning, schema validation
│   └── store/       # Parquet storage, point-in-time joins
├── features/        # Feature engineering (momentum, vol, macro)
├── research/        # Walk-forward splits, evaluation metrics
├── alpha/           # Signal generation (trend, mean-rev, stack)
├── risk/            # Covariance, VaR/CVaR, stress tests
├── portfolio/       # Optimizer, constraints, rebalancing
├── backtest/        # Engine, cost model, reporting
└── main.py          # End-to-end pipeline runner
```

## Core Design Principles

1. **No look-ahead bias** — Features only use past data. Point-in-time joins enforce this.
2. **Portfolio-native** — Thinks in weights across the universe, not individual BUY/SELL.
3. **Realistic costs** — Market impact scales with √(participation). Spread + impact modeled.
4. **Walk-forward evaluation** — Train on past, test on future, with purge gaps.
5. **Deterministic** — Same data + config = same results.
6. **Immutable raw data** — Append-only storage. Never silently overwrite.

## Pipeline Flow

```
DATA → FEATURES → ALPHA SCORES → RANKS → TARGET WEIGHTS → BACKTEST → REPORT
```

- Signal computed at date t close
- Trade executed at t+1 open (1-day delay)
- Costs modeled per rebalance (spread + market impact)
- Walk-forward splits with purge gap prevent leakage

## Data Sources

| Source | Type | Status |
|--------|------|--------|
| Simulated | Prices + Macro | ✅ Built-in |
| Polygon.io | OHLCV | ✅ Implemented |
| Tiingo | EOD backup | ✅ Implemented |
| FRED | Macro (rates, VIX, CPI) | ✅ Implemented |
| FMP | Fundamentals | 🔲 Stub |
| GDELT | News/events | 🔲 Stub |

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/test_core.py -k leak   # Leakage tests only
pytest tests/test_core.py -k split  # Split tests only
```

Tests cover: data integrity, leakage detection, walk-forward splits, cost model, portfolio constraints, and full integration.
