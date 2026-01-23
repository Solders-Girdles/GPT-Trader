# AI Agent Guide

This is the consolidated reference for AI agents working with the GPT-Trader repository.

## Quick Navigation

- Codebase map: `docs/agents/CODEBASE_MAP.md`
- Agent docs index: `docs/agents/README.md`
- Migration status (legacy → modern wiring): `docs/MIGRATION_STATUS.md`

## Current State

**Active System**: Spot trading bot with Coinbase Advanced Trade
**Primary CLI**: `uv run gpt-trader`
**Architecture**: Vertical slices under `src/gpt_trader/`
**Perpetuals**: INTX perps require INTX access + `COINBASE_ENABLE_INTX_PERPS=1`

### Test Status
```bash
uv run pytest --collect-only  # Verify test discovery
```

## Directory Navigation

### Active Areas
```
src/gpt_trader/                    # Main codebase
├── app/                           # DI container + runtime wiring
├── cli/                           # CLI commands (run, account, orders, treasury)
├── features/                      # Vertical feature slices
│   ├── live_trade/                # Production trading engine
│   ├── brokerages/coinbase/       # Coinbase adapter
│   └── intelligence/sizing/       # Position sizing helpers
├── monitoring/                    # Runtime guards and metrics
└── validation/                    # Input validation framework

config/                            # Profile-specific configurations
tests/unit/gpt_trader/             # Active test suite
```

### Legacy Artifacts
- Legacy artifacts are not tracked in the repo; avoid legacy imports or paths.

## Essential Commands

### Development
```bash
# Install dependencies
uv sync

# Run bot in dev mode (mock broker)
uv run gpt-trader run --profile dev --dev-fast

# Run tests
uv run pytest --collect-only  # Check test count
uv run pytest -q              # Run active suite

# Account verification
uv run gpt-trader account snapshot
```

### Agent Tooling
```bash
make agent-health  # Alias for agent-health-full
make agent-health-fast
make agent-health-fast AGENT_HEALTH_FAST_QUALITY_CHECKS=none  # CI: skip lint/format/types
make agent-impact  # Uses importer + file-only suggestions (no integration)
make agent-impact-full  # Includes integration tests
make agent-map
make agent-tests
```

### Trading Operations
```bash
# Treasury operations
uv run gpt-trader treasury convert --from USD --to USDC --amount 1000
uv run gpt-trader treasury move --from-portfolio a --to-portfolio b --amount 50

# Order preview (no execution)
uv run gpt-trader orders preview --symbol BTC-USD --side buy --type market --quantity 0.1
```

### Monitoring
```bash
# Export metrics
uv run python scripts/monitoring/export_metrics.py \
  --profile prod --runtime-root .
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
# Spot trading (default; JWT)
COINBASE_CREDENTIALS_FILE=/path/to/cdp_key.json
# or set both:
# COINBASE_CDP_API_KEY=organizations/{org}/apiKeys/{key_id}
# COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----..."
TRADING_MODES=spot
CFM_ENABLED=0
COINBASE_ENABLE_INTX_PERPS=0

# INTX perps (requires INTX access)
# COINBASE_ENABLE_INTX_PERPS=1
# COINBASE_PROD_CDP_API_KEY=organizations/{org}/apiKeys/{key_id}
# COINBASE_PROD_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----..."

# Legacy fallback (still supported; avoid in new configs)
# COINBASE_API_KEY_NAME=organizations/{org}/apiKeys/{key_id}
# COINBASE_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----..."

# Debug flags
COINBASE_TRADER_DEBUG=1
DRY_RUN=1
```

## Architecture Patterns

### Bot Pattern
```python
from gpt_trader.app.container import ApplicationContainer
from gpt_trader.app.bootstrap import build_bot

# Create bot via ApplicationContainer
container = ApplicationContainer(config)
bot = container.create_bot()

# Or use convenience function
bot = build_bot(config)

# Access services through container
bot.container.broker
bot.container.risk_manager
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
- **CFM futures (US)**: Opt-in via `TRADING_MODES=cfm` + `CFM_ENABLED=1`
- **INTX perps**: Requires INTX access + `COINBASE_ENABLE_INTX_PERPS=1`
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
from gpt_trader.orchestration import live_execution  # removed

# DO: Active imports
from gpt_trader.features.live_trade.engines import TradingEngine
from gpt_trader.features.live_trade.risk import LiveRiskManager

# DON'T: Assume perps are enabled
if config.derivatives_enabled:
    trade_perpetuals()

# DO: Check derivatives gate explicitly
if config.derivatives_enabled and config.coinbase_intx_perpetuals_enabled:
    trade_perpetuals()
```

## Agent Workflow

### Before Starting
- [ ] Run `uv sync`
- [ ] Check test discovery: `uv run pytest --collect-only`
- [ ] Quick smoke test: `uv run gpt-trader run --profile dev --dev-fast`
- [ ] Check if task involves spot or perps features

### During Development
- [ ] Use `src/gpt_trader/` imports only
- [ ] Test with dev profile first
- [ ] Scaffold new slices via `scripts/maintenance/feature_slice_scaffold.py --name <slice>`
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
uv run gpt-trader run --profile dev --dev-fast

# Check system state
uv run gpt-trader account snapshot

# Tail logs
tail -f ${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log
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
