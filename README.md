# GPT-Trader V2

**Coinbase Spot Trading Stack with Future-Ready Perps Support**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-orange.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ML-driven Coinbase trading system with market regime detection, built on a clean vertical slice architecture optimized for AI development.

> **Update (INTX required)**: Coinbase now gates perpetual futures behind the INTX program. The bot runs **spot trading by default** and keeps perpetual logic in place for future activation. Enable derivatives only if your account has INTX access and you set `COINBASE_ENABLE_DERIVATIVES=1` alongside CDP credentials.

## 🔎 What's Active Today

- Coinbase trading stack (spot-first): `src/bot_v2/orchestration/perps_bot.py`
  - Run spot mode: `poetry run perps-bot --profile dev --dev-fast`
    - The dev profile uses the built-in `MockBroker` for safety; enable real spot execution with `SPOT_FORCE_LIVE=1` plus Coinbase API keys.
    - Default universe: top-ten USD spot markets by Coinbase volume (`BTC`, `ETH`, `SOL`, `XRP`, `LTC`, `ADA`, `DOGE`, `BCH`, `AVAX`, `LINK`).
  - Optional perps (future-ready): requires INTX + `COINBASE_ENABLE_DERIVATIVES=1`
  - Adapter: `src/bot_v2/features/brokerages/coinbase/`
  - Account telemetry: `poetry run perps-bot --account-snapshot` (fees, limits, permissions)
  - Treasury helpers: `poetry run perps-bot --convert USD:USDC:1000` and `--move-funds pf-a:pf-b:50`
  - Order tooling: preview without executing `poetry run perps-bot --preview-order --order-symbol BTC-PERP --order-side buy --order-type market --order-qty 0.1`
    and apply edits with `poetry run perps-bot --apply-order-edit ORDER_ID:PREVIEW_ID`
- Coinbase adapter: `src/bot_v2/features/brokerages/coinbase/`
  - Tests: `pytest tests/unit/bot_v2/features/brokerages/coinbase/test_*.py -q`
- CLI entrypoint: `src/bot_v2/cli.py`
  - Run: `poetry run perps-bot ...` or `poetry run gpt-trader ...`
- Stage 3 runner: `scripts/stage3_runner.py` (delegates to `perps-bot` for legacy workflows)

Experimental (kept for demos, not in the perps critical path):
- `features/backtest/*`, `features/ml_strategy/*`, `features/market_regime/*`, `monitoring/monitoring_dashboard.py`
  - These modules are tagged with `__experimental__ = True` and their heavy deps are now optional extras.
  - Install extras as needed, e.g.: `poetry install -E ml -E research -E api`

## 🚀 Quick Start

```bash
# Install dependencies
poetry install

# Run Coinbase bot in spot mode (dev profile)
poetry run perps-bot --profile dev --dev-fast

# Run with derivatives (requires COINBASE_ENABLE_DERIVATIVES=1 + CDP creds)
# COINBASE_ENABLE_DERIVATIVES=1 poetry run perps-bot --profile canary

# Run tests (full spot suite)
poetry run pytest -q

# Stage 3 multi-asset runner
poetry run python scripts/stage3_runner.py --duration-minutes 60
```

> Spot risk defaults (per-symbol caps and slippage guards) are loaded automatically from `config/risk/spot_top10.json` for dev/demo/spot profiles; adjust that file if you need different limits.

## 📊 Current State

### ✅ What's Working
- **Spot Trading**: BTC-USD, ETH-USD via Coinbase Advanced Trade with mock broker support for dev/canary
- **Perps Support (dormant)**: Code paths remain ready; enable only with INTX credentials and derivatives flag
- **Vertical Architecture**: Feature slices under `src/bot_v2/features/` with per-slice tests
- **Risk Management**: Daily loss guard, liquidation buffers, volatility circuit breakers, correlation checks
- **Operational Telemetry**: Account snapshots, cycle metrics, Prometheus exporter
- **Test Coverage**: 445 active tests selected during collection (`poetry run pytest --collect-only -q`)

### 🚨 Production vs Sandbox

| Environment | Products | API | Authentication |
|------------|----------|-----|----------------|
| **Production (default)** | Spot (BTC-USD, ETH-USD, …) | Advanced Trade v3 (HMAC) | API key/secret |
| **Production (perps)** | Perpetuals (INTX-gated) | Advanced Trade v3 | CDP (JWT) + `COINBASE_ENABLE_DERIVATIVES=1` |
| **Sandbox** | Not used by bot (API shape diverges) | — | Use only with `PERPS_PAPER=1` |

**Note:** Sandbox does **not** support perpetuals and the bot will refuse to run live with `COINBASE_SANDBOX=1`. Use production canary profile for perps or stay in spot mode.

## 🦺 Trading Profiles

### Development (`--profile dev`)
- Mock broker with deterministic fills
- Tiny positions for testing
- Extensive logging

### Canary (`--profile canary`)
- Ultra-safe production testing
- 0.01 BTC max positions
- $10 daily loss limit
- Multiple circuit breakers

### Production (`--profile prod`)
- Full position sizing
- Production risk limits
- Real-time monitoring

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get running in 5 minutes
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and capabilities
- **[Perpetuals Trading Logic](docs/reference/trading_logic_perps.md)** - Future-ready INTX implementation details
- **[AI Agent Guide](docs/guides/agents.md)** - For AI development
- **[Production Guide](docs/guides/production.md)** - Deployment guide
- **[Monitoring Guide](docs/guides/monitoring.md)** - Exporter, Prometheus/Grafana, and alerting setup
- **Monitoring Exporter**: `poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json`
  - Serves `/metrics` (Prometheus) and `/metrics.json` (raw snapshot); requires the optional `flask` extra (`poetry install -E monitoring`).
  - Sample stack: `scripts/monitoring/docker-compose.yml.example` (Prometheus, Grafana, Loki, Promtail)

## 🏗️ Architecture

```
src/bot_v2/
├── cli.py                    # Main CLI entry point
├── features/                 # Vertical slices (11 total)
│   ├── live_trade/          # Production trading
│   ├── ml_strategy/         # ML-driven selection
│   ├── market_regime/       # Regime detection
│   ├── position_sizing/     # Kelly Criterion
│   └── brokerages/
│       └── coinbase/        # API integration
└── orchestration/
    └── perps_bot.py         # Main orchestrator
```

### Workspace highlights
- `config/environments/` – versioned environment templates and guidance
- `docs/agents/` – shared playbooks for CLAUDE, Gemini, and the agent roster
- `var/` – local runtime artifacts (logs, metrics, event store); created automatically and ignored by git

## 🧪 Test Status

- **Active Code**: 445 tests collected after deselection ✅
- **Legacy/archived**: Additional tests skipped/deselected by markers
- **Command**: `poetry run pytest --collect-only` (run `poetry install` first for new deps like `pyotp`)

## 🔧 Environment Setup

```bash
# Copy template
cp config/environments/.env.template .env

# Required for perpetuals (production)
COINBASE_PROD_CDP_API_KEY=your_key
COINBASE_PROD_CDP_PRIVATE_KEY=your_private_key

# Optional (paper mode only)
# PERPS_PAPER=1 enables mock trading without touching production
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and [AGENTS.md](docs/agents/Agents.md) for AI-specific guidelines.

## 📈 Performance Metrics

- **Backtest Speed**: 100 symbol-days/second
- **Memory Usage**: <50MB typical
- **WebSocket Latency**: <100ms
- **Order Execution**: <500ms round-trip

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*For detailed documentation, see [docs/README.md](docs/README.md)*
