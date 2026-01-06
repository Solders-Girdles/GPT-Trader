# AI Agent Guide

This is the consolidated reference for AI agents working with the GPT-Trader repository.

## Quick Navigation

- Codebase map: `docs/agents/CODEBASE_MAP.md`
- Migration status (legacy → modern wiring): `docs/MIGRATION_STATUS.md`

## Current State

**Active System**: Spot trading bot with Coinbase Advanced Trade
**Primary CLI**: `uv run coinbase-trader`
**Architecture**: Vertical slices under `src/gpt_trader/`
**Perpetuals**: Code exists but requires INTX access + `COINBASE_ENABLE_DERIVATIVES=1`

### Test Status
```bash
uv run pytest --collect-only  # Verify test discovery
```

## Directory Navigation

### Active Areas
```
src/gpt_trader/                    # Main codebase
├── cli/                           # CLI commands (run, account, orders, treasury)
├── orchestration/                 # Coordinators and service management
│   └── trading_bot/bot.py         # Core orchestrator
├── features/                      # Vertical feature slices
│   ├── live_trade/                # Production trading engine
│   ├── brokerages/coinbase/       # Coinbase adapter
│   └── intelligence/sizing/       # Position sizing (Kelly criterion)
├── monitoring/                    # Runtime guards and metrics
└── validation/                    # Input validation framework

config/                            # Profile-specific configurations
tests/unit/gpt_trader/             # Active test suite
```

### Legacy Artifacts
- `var/legacy/legacy_bundle_*.tar.gz` - Archived experimental slices
- Do NOT use legacy imports or paths

## Essential Commands

### Development
```bash
# Install dependencies
uv sync

# Run bot in dev mode (mock broker)
uv run coinbase-trader run --profile dev --dev-fast

# Run tests
uv run pytest --collect-only  # Check test count
uv run pytest -q              # Run active suite

# Account verification
uv run coinbase-trader account snapshot
```

### Trading Operations
```bash
# Treasury operations
uv run coinbase-trader treasury convert --from USD --to USDC --amount 1000
uv run coinbase-trader treasury move --from-portfolio a --to-portfolio b --amount 50

# Order preview (no execution)
uv run coinbase-trader orders preview --symbol BTC-USD --side buy --type market --quantity 0.1
```

### Monitoring
```bash
# Export metrics
uv run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/coinbase_trader/prod/metrics.json
```

## Configuration

### Profiles
| Profile | Environment | Use Case |
|---------|------------|----------|
| **dev** | Mock broker | Development and testing |
| **canary** | Production | Ultra-safe validation (tiny positions) |
| **prod** | Production | Full trading |

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
COINBASE_TRADER_DEBUG=1
DRY_RUN=1
```

## Architecture Patterns

### Coordinator Pattern
```python
from gpt_trader.orchestration.coordinators import (
    RuntimeCoordinator,
    ExecutionCoordinator,
    StrategyCoordinator,
    TelemetryCoordinator
)

# Access through TradingBot instance
bot.runtime_coordinator
bot.execution_coordinator
```

### Feature Imports
```python
# Active imports
from gpt_trader.features.live_trade.risk import LiveRiskManager
from gpt_trader.features.brokerages.coinbase import CoinbaseClient
from gpt_trader.features.intelligence.sizing import PositionSizer
```

## Common Confusion Points

### 1. Spot vs Perpetuals
- **Spot trading**: Active (BTC-USD, ETH-USD)
- **Perpetuals**: Requires INTX access + `COINBASE_ENABLE_DERIVATIVES=1`
- Default behavior is spot-only

### 2. Legacy vs Active Code
- **Use**: `src/gpt_trader/**` only
- **Avoid**: Legacy paths, archived experimental slices
- Legacy imports will cause errors

### 3. Documentation Trust
- Always verify file modification dates
- Cross-reference with actual code
- Prefer recent docs over older references

### Common Mistakes
```python
# DON'T: Legacy imports
from src.bot.paper_trading import ml_paper_trader

# DO: Active imports
from gpt_trader.features.live_trade.risk import LiveRiskManager

# DON'T: Assume perps are enabled
if config.derivatives_enabled:
    trade_perpetuals()

# DO: Check derivatives gate explicitly
if config.derivatives_enabled and config.intx_access:
    trade_perpetuals()
```

## Agent Workflow

### Before Starting
- [ ] Run `uv sync`
- [ ] Check test discovery: `uv run pytest --collect-only`
- [ ] Quick smoke test: `uv run coinbase-trader run --profile dev --dev-fast`
- [ ] Check if task involves spot or perps features

### During Development
- [ ] Use `src/gpt_trader/` imports only
- [ ] Test with dev profile first
- [ ] Add tests to `tests/unit/gpt_trader/`
- [ ] Run `uv run pytest -q` regularly

### Before Finishing
- [ ] Update relevant documentation
- [ ] Note INTX gating for perps work
- [ ] Remove references to archived components
- [ ] Verify test counts remain stable

## Debugging

```bash
# Enable debug logging
export PERPS_DEBUG=1
export LOG_LEVEL=DEBUG
uv run coinbase-trader run --profile dev --dev-fast

# Check system state
uv run coinbase-trader account snapshot

# Tail logs
tail -f var/logs/coinbase_trader.log
```

## Agent-Specific Tips

### Claude
- Use planning tool for multi-slice edits
- Surface risk-impact summaries in responses
- Include testing commands for reviewers

### Gemini
- Keep responses concise with numbered follow-ups
- Include exact commands with `rg`/`fd` snippets
- Call out environment prerequisites

## Key Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview and quick start |
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/guides/testing.md` | Testing guide |
| `docs/guides/production.md` | Production deployment |
| `docs/guides/agent-tools.md` | **Agent tools reference** |
| `docs/reference/coinbase_complete.md` | Coinbase API reference |

## Source of Truth Checklist

After making changes:
- [ ] README reflects new instructions
- [ ] Architecture doc matches live system
- [ ] Tests pass or document dependency gaps
- [ ] Note INTX gate for perps-related changes
