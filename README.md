# GPT-Trader

Coinbase trading system built with AI coding assistants.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## About

GPT-Trader is a Coinbase spot trading bot with vertical slice architecture. The name reflects how AI assistants (Claude, GPT, Gemini) collaborate in developing this codebase—the trading strategies themselves use technical analysis, not LLM inference.

**Current focus**: Spot trading on Coinbase Advanced Trade. Perpetual futures support exists but requires INTX program access.

## Quick Start

```bash
# Install
uv sync

# Run (dev profile with mock broker)
uv run coinbase-trader run --profile dev --dev-fast

# Launch the TUI (mode selector; add --mode demo/paper/live to skip the prompt)
uv run gpt-trader tui

# Run tests
uv run pytest tests/unit -q
```

Use `uv run gpt-trader tui` instead of `python -m ...` to keep uv-locked dependencies and the CLI's env/logging initialization.

## Configuration

The bot uses profiles for different environments:

| Profile | Broker | Use Case |
|---------|--------|----------|
| `dev` | DeterministicBroker (mock) | Local development |
| `canary` | Real (tiny limits) | Production testing |
| `prod` | Real | Production |

Environment variables control credentials and risk limits. See `config/environments/.env.template`.

## Project Structure

```
src/gpt_trader/
├── cli/              # Command-line interface
├── features/         # Vertical slices
│   ├── brokerages/   # Coinbase integration
│   ├── data/         # Market data
│   ├── live_trade/   # Production trading
│   ├── optimize/     # Parameter optimization
│   └── strategy_tools/
├── orchestration/    # Trading bot core
└── monitoring/       # Telemetry and metrics
```

## Agent Tools

AI assistants can use these commands for context:

```bash
uv run agent-check      # Quality gate
uv run agent-impact     # Change impact analysis
uv run agent-tests      # Test discovery
uv run agent-naming     # Naming convention check
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design
- [Contributing](CONTRIBUTING.md) - Development workflow

## Development

```bash
# Format and lint
uv run ruff check . --fix
uv run black .

# Type check
uv run mypy src/gpt_trader

# Full quality gate
pre-commit run --all-files
```
