# Agent Quick Reference for GPT-Trader

This is a condensed reference for AI agents working with the GPT-Trader repository. Focus on the current spot-first architecture and avoid legacy confusion.

## 🎯 Current State (2025-10)

**Active System**: Spot trading bot with Coinbase Advanced Trade
**Primary CLI**: `poetry run perps-bot`
**Architecture**: Vertical slices under `src/bot_v2/`
**Perpetuals**: ⚠️ Code exists but requires INTX access + `COINBASE_ENABLE_DERIVATIVES=1`

## 📁 Directory Navigation

### ✅ Active Areas
```
src/bot_v2/                    # Main codebase (161 test files)
├── cli/                       # CLI commands (run, account, orders, treasury)
├── orchestration/             # Coordinators and service management
├── features/                  # Vertical feature slices
│   ├── live_trade/           # Production trading engine
│   ├── brokerages/coinbase/  # Coinbase adapter
│   └── position_sizing/      # Risk management
├── monitoring/               # Runtime guards and metrics
└── validation/               # Input validation framework

config/                       # Profile-specific configurations
tests/unit/bot_v2/           # Active test suite (1484 collected / 1483 selected / 1 deselected)
docs/agents/                 # Agent-specific guides
```

### ⚠️ Legacy Artifacts
- `var/legacy/legacy_bundle_<timestamp>.tar.gz` contains archived experimental slices and the retired PoC CLI (see `docs/archive/legacy_recovery.md`)
- `docs/archive/` stores historical documentation snapshots

## 🚀 Essential Commands

### Development
```bash
# Install dependencies
poetry install

# Run bot in dev mode (mock broker)
poetry run perps-bot run --profile dev --dev-fast

# Run tests
poetry run pytest --collect-only  # Check test count
poetry run pytest -q              # Run active suite

# Account verification
poetry run perps-bot account snapshot
```

### Trading Operations
```bash
# Treasury operations
poetry run perps-bot treasury convert --from USD --to USDC --amount 1000
poetry run perps-bot treasury move --from-portfolio a --to-portfolio b --amount 50

# Order preview (no execution)
poetry run perps-bot orders preview --symbol BTC-USD --side buy --type market --quantity 0.1
```

### Monitoring
```bash
# Export metrics
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json
```

## 🔧 Configuration

### Profiles
- **dev**: Mock broker, tiny positions, verbose logging
- **canary**: Ultra-safe production (0.01 BTC max, $10 daily loss)
- **prod**: Full production trading

### Environment Variables
```bash
# Spot trading (default)
COINBASE_API_KEY=your_hmac_key
COINBASE_API_SECRET=your_hmac_secret
COINBASE_ENABLE_DERIVATIVES=0

# Perpetuals (requires INTX access)
# COINBASE_ENABLE_DERIVATIVES=1
# COINBASE_PROD_CDP_API_KEY=organizations/{org}/apiKeys/{key_id}
# COINBASE_PROD_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----..."

# Debug flags
PERPS_DEBUG=1                 # Enable debug logging
DRY_RUN=1                     # Dry run mode
```

## 🏗️ Architecture Patterns

### Coordinator Pattern (New)
```python
# Services are managed by coordinators
from bot_v2.orchestration.coordinators import (
    RuntimeCoordinator,
    ExecutionCoordinator,
    StrategyCoordinator,
    TelemetryCoordinator
)

# Access through PerpsBot instance
bot.runtime_coordinator
bot.execution_coordinator
bot.strategy_coordinator
bot.telemetry_coordinator
```

### Feature Slices
```python
# Import from active slices
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.brokerages.coinbase import CoinbaseClient
from bot_v2.features.position_sizing import calculate_position_size
```

### Configuration
```python
# Load configuration
from bot_v2.orchestration.configuration import BotConfig
config = BotConfig.from_profile("dev")

# Service registry
from bot_v2.orchestration.service_registry import ServiceRegistry
registry = ServiceRegistry.from_config(config)
```

## 🧪 Testing Guidelines

### Test Structure
```bash
tests/unit/bot_v2/            # Active tests
├── features/                 # Feature-specific tests
├── orchestration/            # Core orchestration tests
├── cli/                      # CLI tests
└── monitoring/               # Monitoring tests
```

### Test Commands
```bash
# Run all active tests
poetry run pytest -q

# Run specific module
poetry run pytest tests/unit/bot_v2/features/live_trade/ -q

# Check test discovery
poetry run pytest --collect-only | grep "test session starts"
```

### Expected Results
- **Collected**: 1484 tests
- **Selected**: 1483 tests (active)
- **Deselected**: 1 legacy placeholder

## 🚨 Common Mistakes to Avoid

### ❌ Don't Do This
```python
# Legacy imports
from src.bot.paper_trading import ml_paper_trader
from archived.experimental.features.backtest import BacktestEngine

# Assume perps are enabled
if config.derivatives_enabled:  # Usually False
    trade_perpetuals()

# Use outdated config settings
data_provider = config.data_provider.default  # Contains legacy settings
```

### ✅ Do This Instead
```python
# Active imports
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.orchestration.live_execution import LiveExecutionEngine

# Check derivatives gate explicitly
if config.derivatives_enabled and config.intx_access:
    trade_perpetuals()

# Use profile-specific configuration
profile_config = config.load_profile_config(config.profile)
```

## 📋 Agent Workflow Checklist

### Before Starting
- [ ] Run `poetry install` to ensure dependencies
- [ ] Check test discovery: `poetry run pytest --collect-only`
- [ ] Verify current working directory structure
- [ ] Check if task involves spot or perps features

### During Development
- [ ] Use `src/bot_v2/` imports only
- [ ] Test with dev profile first
- [ ] Add tests to `tests/unit/bot_v2/`
- [ ] Run `poetry run pytest -q` regularly

### Before Finishing
- [ ] Update relevant documentation
- [ ] Sync agent guides if architecture changed
- [ ] Note INTX gating for perps-related work
- [ ] Remove references to archived components

## 🔍 Debugging Tips

### Enable Debug Logging
```bash
export PERPS_DEBUG=1
export LOG_LEVEL=DEBUG
poetry run perps-bot run --profile dev --dev-fast
```

### Check System State
```bash
# Account status
poetry run perps-bot account snapshot

# Metrics
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/dev/metrics.json

# Logs
tail -f var/logs/perps_bot.log
```

### Common Issues
- **Import errors**: Check if using legacy paths
- **Configuration missing**: Verify profile exists in `config/`
- **Test failures**: Check if using deselected legacy tests
- **Perps not working**: Verify INTX access and `COINBASE_ENABLE_DERIVATIVES=1`

## 📚 Key Documentation

- **Agent Guide**: `docs/agents/Agents.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Development**: `docs/DEVELOPMENT_GUIDELINES.md`
- **Testing**: `docs/guides/testing.md`
- **Production**: `docs/guides/production.md`

## 🆘 Getting Help

1. Check `docs/AGENT_CONFUSION_POINTS.md` for known issues
2. Run `poetry run pytest --collect-only` to verify test state
3. Use dev profile with `--dev-fast` for safe testing
4. Check file modification dates to verify recency
5. Look at `docs/agents/Agents.md` for agent-specific guidance

---

*Last updated: 2025-10-18 | For the latest confusion points, see `docs/AGENT_CONFUSION_POINTS.md`*
