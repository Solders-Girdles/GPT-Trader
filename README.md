# GPT-Trader (v1 scaffold)

Architecture-first scaffold for an equities-only trading bot with:
- Adapter-based data layer (free now, swappable to real-time later)
- Backtest engine + demo MA strategy
- Config via `.env` and Pydantic
- Lint/format/type checks with Ruff, Black, Mypy
- Tests with Pytest
- Pre-commit hooks

## Quick start

```bash
# From your project root (where pyproject.toml lives)
poetry install
pre-commit install

# Copy .env.example to .env and fill values (Alpaca keys optional for v1)
cp .env.example .env

# Run a demo backtest
poetry run python -m bot.cli backtest --strategy demo_ma --symbol AAPL --start 2023-01-01 --end 2023-06-30
```

## Commands

```bash
# Backtest
poetry run python -m src.bot.cli backtest --strategy demo_ma --symbol AAPL --start 2023-01-01 --end 2023-06-30

# (placeholder) Paper trading - wired later
poetry run python -m src.bot.cli paper --strategy demo_ma
```

## Layout

```
src/bot/
  config.py        # Pydantic settings
  logging.py       # Logger setup
  cli.py           # Entrypoints: backtest, paper (stub), live (stub)

  dataflow/        # Data adapters
    base.py
    sources/yfinance_source.py

  strategy/        # Strategies
    base.py
    demo_ma.py

  risk/            # Risk rules
    basic.py

  backtest/        # Backtest engine
    engine.py

  exec/            # Broker adapters (stubs for now)
    base.py
    alpaca_paper.py

  monitor/         # Alerts/health (stubs)
    alerts.py
```

## Notes

- We start with `yfinance` for historical bars; later we can plug in Polygon/IEX/etc.
- SQLite not required for v1; the backtest emits CSV into `./data/backtests`. We'll add DB once needed.
- PyCharm CE is sufficient.
