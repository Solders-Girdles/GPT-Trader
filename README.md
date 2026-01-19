# GPT-Trader

A production-ready Coinbase trading system built collaboratively with AI coding assistants.

[![CI](https://github.com/Solders-Girdles/GPT-Trader/actions/workflows/ci.yml/badge.svg)](https://github.com/Solders-Girdles/GPT-Trader/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

GPT-Trader is a Coinbase trading bot featuring a Terminal User Interface, vertical slice architecture, and comprehensive risk management. The name reflects how AI assistants (Claude, GPT, Gemini) collaborate in developing this codebase—the trading strategies themselves use technical analysis, not LLM inference.

### Trading Capabilities

| Mode | Status | Description |
|------|--------|-------------|
| **Spot Trading** | Active | BTC-USD, ETH-USD, and top-10 USD pairs |
| **CFM Futures** | Available | US-regulated futures via Coinbase Financial Markets |
| **INTX Perpetuals** | Code Ready | Requires international INTX account access |

## Quick Start

```bash
# Install dependencies
uv sync

# Launch the Terminal UI (recommended)
uv run gpt-trader tui

# Or run the trading bot directly
uv run gpt-trader run --profile dev
```

## Terminal User Interface

The TUI provides real-time monitoring and control with a modern dashboard experience:

```bash
uv run gpt-trader tui              # Mode selector
uv run gpt-trader tui --mode demo  # Demo mode (no credentials needed)
uv run gpt-trader tui --mode paper # Paper trading
uv run gpt-trader tui --mode live  # Live trading
```

**Features:**
- Real-time market data and portfolio display
- Account, position, and order management
- Risk monitoring and alerts
- Demo mode for safe exploration
- Keyboard-first navigation

## Configuration

### Trading Profiles

| Profile | Broker | Use Case |
|---------|--------|----------|
| `dev` | DeterministicBroker (mock) | Local development |
| `canary` | Real (tiny limits) | Production validation |
| `prod` | Real | Full production trading |

### Environment Setup

Copy the template and configure your credentials:

```bash
cp config/environments/.env.template .env
```

Key variables:
- `COINBASE_CREDENTIALS_FILE` or `COINBASE_CDP_API_KEY` + `COINBASE_CDP_PRIVATE_KEY` - JWT credentials
- `GPT_TRADER_PROFILE` - Trading profile (dev/canary/prod)

See [config/environments/.env.template](config/environments/.env.template) for all options.

## Project Structure

```
src/gpt_trader/
├── app/                  # Modern DI container (ApplicationContainer)
├── backtesting/           # Backtesting framework (canonical)
├── cli/                  # Command-line interface
├── features/             # Vertical feature slices
│   ├── brokerages/       # Coinbase REST/WebSocket integration
│   ├── data/             # Market data acquisition
│   ├── intelligence/     # Strategy intelligence, Kelly sizing
│   ├── live_trade/       # Production trading engine & risk
│   ├── optimize/         # Parameter optimization
│   └── strategy_tools/   # Shared strategy helpers
├── monitoring/           # Runtime guards, metrics, telemetry
├── persistence/          # Event/order persistence
├── security/             # Secrets management, input sanitization
├── tui/                  # Terminal User Interface (Textual)
└── validation/           # Declarative validators
```

## Development

### Scaffold a New Slice

```bash
make scaffold-slice name=<slice> flags="--with-tests --with-readme"
```

Or run directly:

```bash
uv run python scripts/maintenance/feature_slice_scaffold.py --name <slice> --dry-run
```

### Quality Gates

```bash
# Linting and formatting
uv run ruff check . --fix
uv run black .

# Type checking
uv run mypy src/gpt_trader

# Run all pre-commit hooks
pre-commit run --all-files

# Check naming conventions
uv run agent-naming
```

### Testing

```bash
# Unit tests (fast, default)
uv run pytest tests/unit -q

# With coverage
uv run pytest tests/unit --cov=src/gpt_trader -q

# Property-based tests
uv run pytest tests/property -q

# TUI snapshot tests
uv run pytest tests/unit/gpt_trader/tui/test_snapshots_*.py -q
```

### Agent Tools

Commands for AI-assisted development:

```bash
uv run agent-check      # Quality gate with JSON output
uv run agent-impact     # Analyze change impact
uv run agent-map        # Generate dependency graph
uv run agent-naming     # Check naming conventions
uv run agent-risk       # Query risk configuration
```

## Documentation

| Document | Purpose |
|----------|---------|
| [Architecture](docs/ARCHITECTURE.md) | System design and vertical slices |
| [Reliability](docs/RELIABILITY.md) | Guard stack, degradation, chaos testing |
| [Monitoring](docs/MONITORING_PLAYBOOK.md) | Metrics, alerting, dashboards |
| [Production](docs/guides/production.md) | Deployment and operations |
| [Contributing](CONTRIBUTING.md) | Development workflow |
| [TUI Style Guide](docs/TUI_STYLE_GUIDE.md) | Terminal UI conventions |

Full documentation index: [docs/README.md](docs/README.md)

## Architecture Notes

This project uses a modern **Dependency Injection** pattern via `ApplicationContainer` in `src/gpt_trader/app/`. The legacy `orchestration/` layer was removed during the DI migration; prefer `app/` and `features/` paths.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## License

MIT
