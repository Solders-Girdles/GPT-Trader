# Perpetuals Trading System - Operations Runbook

## Overview

This runbook covers the perpetuals trading system from Phase 1-7, including product discovery, execution, risk management, and strategy integration.

## Quick Start

### 1. Development Mode (Mocked)
```bash
# Run with mock broker and dry-run
python scripts/run_perps_bot.py --profile dev --dry-run

# Single cycle for testing (exits after one iteration)
python scripts/run_perps_bot.py --profile dev --dry-run --dev-fast

# With specific symbols
python scripts/run_perps_bot.py --profile dev --symbols BTC-PERP ETH-PERP

# Using CLI interface
python -m src.bot_v2.simple_cli perps --profile dev --dev-fast
```

### 2. Demo Mode (Tiny Positions)
```bash
# Small position sizes for testing
python scripts/run_perps_bot.py --profile demo --symbols BTC-PERP

# With reduced leverage
python scripts/run_perps_bot.py --profile demo --leverage 1
```

### 3. Production Mode
```bash
# Full trading (requires authentication)
python scripts/run_perps_bot.py --profile prod

# Reduce-only mode (exit only)
python scripts/run_perps_bot.py --profile prod --reduce-only

# Single cycle validation
python scripts/run_perps_bot.py --profile prod --dev-fast --dry-run
```

## Environment Variables

The system uses several environment variables for configuration and testing:

### Core Variables
- **`EVENT_STORE_ROOT`** - Data storage location (default: `./data`)
  - All health files, state files, and events are stored under this path
  - Format: `$EVENT_STORE_ROOT/perps_bot/{profile}/`
  - Example: `EVENT_STORE_ROOT=/tmp/perps_test python scripts/run_perps_bot.py --profile dev --dev-fast`

- **`COINBASE_SANDBOX`** - Enable sandbox mode for demo/test profiles
  - Set to `"1"` to force sandbox mode
  - Required for demo profile in production environments

- **`PERPS_FORCE_MOCK`** - Override broker selection to use mock
  - Set to `"1"` to force mock broker regardless of profile
  - Useful for CI testing of demo/prod profiles without credentials

- **`LOG_LEVEL`** - Logging verbosity (DEBUG, INFO, WARNING, ERROR)

- **`COINBASE_ENABLE_DERIVATIVES`** - Enable derivatives/perpetuals trading
  - Set to `"1"` to enable CFM endpoints for perpetuals
  - Required for accessing BTC/ETH/SOL/XRP perps

- **`RUN_SANDBOX_VALIDATIONS`** - Enable sandbox validation scripts
  - Set to `"1"` to run Week 1 validation scripts
  - Prevents accidental live API calls during testing

### Usage Examples
```bash
# Local testing with custom data directory
EVENT_STORE_ROOT=/tmp/perps_test python scripts/run_perps_bot.py --profile dev --dev-fast

# CI testing of demo profile (mocked)
COINBASE_SANDBOX=1 PERPS_FORCE_MOCK=1 python scripts/run_perps_bot.py --profile demo --dev-fast

# Debug logging
LOG_LEVEL=DEBUG python scripts/run_perps_bot.py --profile dev
```

## Week 1 Enhancements

### New Order Types & TIF Support
```bash
# Market orders (default)
python scripts/run_perps_bot.py --profile demo --order-type market

# Limit orders with time-in-force
python scripts/run_perps_bot.py --profile demo --order-type limit --tif GTC
python scripts/run_perps_bot.py --profile demo --order-type limit --tif IOC

# CLI interface with new flags
python -m src.bot_v2.simple_cli perps --profile demo --order-type limit --tif GTC
```

### Sandbox Testing
```bash
# Coinbase sandbox environment
COINBASE_SANDBOX=1 COINBASE_ENABLE_DERIVATIVES=1 python scripts/run_perps_bot.py --profile demo

# Validation scripts (requires sandbox credentials)
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_perps_client_week1.py
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_ws_week1.py
```

### Dynamic Symbol Support
- **Automatic Discovery**: BTC-PERP, ETH-PERP, SOL-PERP, XRP-PERP detected dynamically
- **No Hardcoding**: Product enumeration via Coinbase API
- **Funding Rates**: Real-time funding rate tracking for perpetuals

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Points                         │
├─────────────────────────────────────────────────────────┤
│  run_perps_bot.py │ simple_cli.py │ Jupyter Notebooks  │
└────────────┬──────┴───────┬────────┴────────┬──────────┘
             │              │                  │
┌────────────▼──────────────▼──────────────────▼──────────┐
│                  Strategy Layer                          │
├───────────────────────────────────────────────────────────┤
│  BaselinePerpsStrategy                                   │
│  • MA Crossover Signals                                  │
│  • Trailing Stop Logic                                   │
│  • Position Sizing                                       │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│                   Risk Layer                             │
├───────────────────────────────────────────────────────────┤
│  LiveRiskManager                                         │
│  • Leverage Limits                                       │
│  • Loss Controls                                         │
│  • Exposure Management                                   │
│  • Reduce-Only Mode                                      │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│                Execution Layer                           │
├───────────────────────────────────────────────────────────┤
│  ExecutionEngine                                         │
│  • Order Placement                                       │
│  • Fill Simulation                                       │
│  • Quantization Rules                                    │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│                 Brokerage Layer                          │
├───────────────────────────────────────────────────────────┤
│  CoinbasePerpetualsClient                               │
│  • Product Discovery                                     │
│  • Order Management                                      │
│  • Position Tracking                                     │
│  • Mark Price Feeds                                      │
└───────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Product Catalog (Phase 2)

**Location**: `src/bot_v2/features/brokerages/coinbase/product_catalog.py`

**Key Functions**:
- `load_from_brokerage()` - Fetch product metadata
- `get_perpetual_products()` - Filter for perps only
- `enforce_perp_rules()` - Apply quantization

**Usage**:
```python
from src.bot_v2.features.brokerages.coinbase.product_catalog import ProductCatalog
from src.bot_v2.features.brokerages.coinbase.utils import enforce_perp_rules

# Load products
catalog = ProductCatalog()
await catalog.load_from_brokerage(client)

# Get perps
perps = catalog.get_perpetual_products()

# Quantize order
qty, price = enforce_perp_rules(product, qty=Decimal("1.234"), price=Decimal("50000"))
```

### 2. Execution Engine (Phase 3)

**Location**: `src/bot_v2/features/live_trade/execution.py`

**Key Features**:
- Order placement with retries
- Fill simulation for testing
- Cancel/modify order support
- Event logging

**Usage**:
```python
from src.bot_v2.features.live_trade.execution import ExecutionEngine

engine = ExecutionEngine(config=exec_config, event_store=event_store)

# Place order
order = await engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    qty=Decimal("0.1"),
    order_type="market"
)

# Cancel order
await engine.cancel_order(order.id)
```

### 3. Position Tracking (Phase 4)

**Location**: `src/bot_v2/features/live_trade/position_tracker.py`

**Key Features**:
- Real-time position updates
- P&L calculation
- Fee tracking
- Mark-to-market

**Usage**:
```python
from src.bot_v2.features.live_trade.position_tracker import PositionTracker

tracker = PositionTracker(event_store=event_store)

# Update position
tracker.update_position(symbol="BTC-PERP", qty=Decimal("0.1"), price=Decimal("50000"))

# Get current position
position = tracker.get_position("BTC-PERP")
print(f"Unrealized P&L: {position.unrealized_pnl}")
```

### 4. Risk Manager (Phase 5)

**Location**: `src/bot_v2/features/live_trade/risk.py`

**Key Features**:
- Pre-trade validation
- Post-trade checks
- Kill switch
- Daily loss limits

**Configuration**:
```python
risk_config = RiskConfig(
    leverage_max_global=5,
    max_daily_loss_pct=0.05,
    max_exposure_pct=0.8,
    reduce_only_mode=False,
    kill_switch_enabled=False
)
```

### 5. Strategy (Phase 6)

**Location**: `src/bot_v2/features/live_trade/strategies/perps_baseline.py`

**Key Features**:
- MA crossover signals
- Trailing stops
- Position sizing
- Feature flags

**Configuration**:
```python
strategy_config = StrategyConfig(
    short_ma_period=5,
    long_ma_period=20,
    target_leverage=2,
    trailing_stop_pct=0.01,
    enable_shorts=False
)
```

## Operational Procedures

### Starting the Bot

1. **Check prerequisites**:
   ```bash
   # Verify environment
   python -c "import src.bot_v2; print('OK')"
   
   # Check API credentials (if prod)
   echo $COINBASE_API_KEY
   ```

2. **Start in dry-run first**:
   ```bash
   python scripts/run_perps_bot.py --profile demo --dry-run
   ```

3. **Monitor initial cycles**:
   - Check for MA signal generation
   - Verify risk checks pass
   - Confirm quantization applied

4. **Enable live trading**:
   ```bash
   python scripts/run_perps_bot.py --profile demo
   ```

### Monitoring

**Log Locations** (under `EVENT_STORE_ROOT`, default: `./data`):
- Bot state: `${EVENT_STORE_ROOT:-data}/perps_bot/{profile}/last_state.json`
- Health status: `${EVENT_STORE_ROOT:-data}/perps_bot/{profile}/health.json`
- Event logs: `${EVENT_STORE_ROOT:-data}/perps_bot/{profile}/events/`
- Risk events: `${EVENT_STORE_ROOT:-data}/perps_bot/{profile}/risk_events.json`

**Key Metrics**:
```
- Current positions and P&L
- Daily loss vs limit
- Leverage vs limits
- Signal generation rate
- Order success rate
```

**Health Checks**:
```bash
# Check bot is running
ps aux | grep run_perps_bot

# Check latest state (respects EVENT_STORE_ROOT)
cat ${EVENT_STORE_ROOT:-data}/perps_bot/demo/last_state.json

# Check health status
cat ${EVENT_STORE_ROOT:-data}/perps_bot/demo/health.json

# Check for errors
grep ERROR logs/perps_bot.log
```

### Health File Format

The `health.json` file provides quick bot status for monitoring:

```json
{
  "timestamp": "2025-08-30T02:37:35.326508",
  "ok": true,
  "profile": "dev", 
  "symbols": ["BTC-PERP"],
  "last_decisions": {
    "BTC-PERP": "hold"
  },
  "message": "",
  "error": ""
}
```

**Key Fields**:
- `ok`: Overall health status (true/false)
- `profile`: Active trading profile (dev/demo/prod)
- `symbols`: Symbols being tracked
- `last_decisions`: Most recent strategy decisions per symbol
- `message`: Optional status message
- `error`: Error details if ok=false

### Emergency Procedures

#### 1. Kill Switch Activation

**Immediate stop**:
```bash
# Send interrupt signal
kill -INT <bot_pid>

# Or use reduce-only mode
python scripts/run_perps_bot.py --profile prod --reduce-only
```

**Via Risk Manager**:
```python
risk_manager.activate_kill_switch("Manual intervention required")
```

#### 2. Position Unwinding

**Close all positions**:
```python
from scripts.emergency_close_all import close_all_positions

# Close with market orders
await close_all_positions(broker_client, symbols=["BTC-PERP", "ETH-PERP"])
```

#### 3. Risk Limit Breach

**Symptoms**:
- "Risk check failed" in logs
- Orders rejected
- reduce_only_mode activated

**Resolution**:
1. Check current exposure
2. Reduce position sizes
3. Adjust risk config
4. Clear daily loss counter (if needed)

### Testing Procedures

#### 1. Unit Tests
```bash
# Test strategy
python -m pytest tests/unit/bot_v2/features/live_trade/test_derivatives_phase6.py -v

# Test quantization
python tests/unit/bot_v2/features/live_trade/test_quantization_simple.py
```

#### 2. Integration Tests
```bash
# Run validation scripts
python scripts/validate_derivatives_phase6_strategy.py
python scripts/validate_derivatives_phase7_e2e.py
```

#### 3. Week 1 Validation Tests

**Mock Mode (Default - recommended for CI/testing):**
```bash
# Perpetuals client validation
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_perps_client_week1.py

# WebSocket market data validation  
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_ws_week1.py

# Runner smoke test with mock broker
PERPS_FORCE_MOCK=1 python scripts/run_perps_bot.py --profile dev --dry-run --dev-fast --symbols BTC-PERP
```

**Real Sandbox Mode (for live API validation):**
```bash
# Set sandbox credentials
export COINBASE_SANDBOX=1
export COINBASE_API_KEY=<sandbox_key>
export COINBASE_API_SECRET=<sandbox_secret> 
export COINBASE_PASSPHRASE=<sandbox_passphrase>
export COINBASE_ENABLE_DERIVATIVES=1

# Run with real Coinbase adapter
USE_REAL_ADAPTER=1 RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_perps_client_week1.py
USE_REAL_ADAPTER=1 RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_ws_week1.py

# Real sandbox runner test (uses actual Coinbase sandbox)
python scripts/run_perps_bot.py --profile dev --dry-run --dev-fast --symbols BTC-PERP
```

**Available Perpetuals**: BTC-PERP, ETH-PERP, SOL-PERP, XRP-PERP

#### 3. Mock Trading
```bash
# Full cycle with mocks
python scripts/run_perps_bot.py --profile dev --interval 1
```

## Configuration Reference

### Profile Settings

| Profile | Max Position | Max Leverage | Enable Shorts | Mock Broker |
|---------|-------------|--------------|---------------|-------------|
| dev     | $10,000     | 3x           | No            | Yes         |
| demo    | $100        | 1x           | No            | No          |
| prod    | $50,000     | 3x           | Yes           | No          |

### Strategy Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| short_ma_period | 5 | 2-50 | Fast MA period |
| long_ma_period | 20 | 10-200 | Slow MA period |
| target_leverage | 2 | 1-5 | Position leverage |
| trailing_stop_pct | 0.01 | 0.005-0.05 | Stop distance |

### Risk Limits

| Limit | Default | Description |
|-------|---------|-------------|
| leverage_max_global | 5x | Global leverage cap |
| max_daily_loss_pct | 5% | Daily loss limit |
| max_exposure_pct | 80% | Max portfolio exposure |
| max_position_pct_per_symbol | 20% | Per-symbol limit |

## Troubleshooting

### Common Issues

#### 1. "No mark price for symbol"
**Cause**: Market data feed issue
**Fix**: 
- Check broker connection
- Verify symbol is correct
- Restart with fresh mark cache

#### 2. "Risk check failed: Leverage exceeds limit"
**Cause**: Position too large for equity
**Fix**:
- Reduce target_leverage
- Increase equity
- Check current positions

#### 3. "Quantization failed: Below minimum"
**Cause**: Order size too small
**Fix**:
- Check min_notional for product
- Increase position size
- Verify decimal precision

#### 4. "Strategy generates no signals"
**Cause**: Insufficient price data
**Fix**:
- Wait for MA periods to fill
- Check mark price updates
- Verify MA parameters

### Debug Commands

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/run_perps_bot.py --profile dev

# Test specific symbol
python -c "
from scripts.test_single_symbol import test_symbol
test_symbol('BTC-PERP')
"

# Dump current state
python -c "
import json, os
path = os.environ.get('EVENT_STORE_ROOT', 'data') + '/perps_bot/dev/last_state.json'
with open(path) as f:
    print(json.dumps(json.load(f), indent=2))
"
```

## Performance Tuning

### Optimization Tips

1. **Reduce update interval for less frequent trading**:
   ```bash
   --interval 30  # 30 second updates
   ```

2. **Limit symbols for focused trading**:
   ```bash
   --symbols BTC-PERP  # Single symbol
   ```

3. **Adjust MA periods for market conditions**:
   - Volatile: Shorter periods (3/10)
   - Trending: Standard (5/20)
   - Ranging: Longer (10/50)

4. **Use appropriate leverage**:
   - Testing: 1x
   - Conservative: 2x
   - Aggressive: 3-5x

## Maintenance

### Daily Tasks
- Review P&L and positions
- Check for errors in logs
- Verify risk limits
- Monitor strategy performance

### Weekly Tasks
- Analyze trade history
- Adjust strategy parameters
- Review risk events
- Update mark windows cache

### Monthly Tasks
- Full system backup
- Performance analysis
- Strategy optimization
- Risk limit review

## Support

### Resources
- Source: `src/bot_v2/features/live_trade/`
- Tests: `tests/unit/bot_v2/features/live_trade/`
- Scripts: `scripts/`
- Docs: `docs/`

### Contacts
- Issues: GitHub Issues
- Dev Team: [Team Contact]
- Risk Team: [Risk Contact]

## Appendix

### A. Event Types

| Event | Description | Location |
|-------|-------------|----------|
| trade.executed | Order filled | execution.py |
| risk.rejected | Trade blocked | risk.py |
| position.updated | Position changed | position_tracker.py |
| strategy.signal | Signal generated | perps_baseline.py |

### B. File Structure

```
GPT-Trader/
├── src/bot_v2/features/
│   ├── live_trade/
│   │   ├── strategies/
│   │   │   └── perps_baseline.py
│   │   ├── execution.py
│   │   ├── risk.py
│   │   └── position_tracker.py
│   └── brokerages/
│       └── coinbase/
│           ├── perpetuals_client.py
│           └── product_catalog.py
├── scripts/
│   ├── run_perps_bot.py
│   └── validate_derivatives_*.py
├── tests/
│   └── unit/bot_v2/features/live_trade/
│       └── test_derivatives_*.py
└── docs/
    └── RUNBOOK_PERPS.md
```

### C. Safety Checklist

- [ ] API credentials secured
- [ ] Risk limits configured
- [ ] Dry-run tested
- [ ] Monitoring enabled
- [ ] Kill switch ready
- [ ] Backups configured
- [ ] Team notified

---

*Last Updated: Phase 7 - End-to-End Testing*