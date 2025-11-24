# Coinbase Perpetuals Trading Logic (Future-Ready)

> **Status:** Coinbase currently gates perpetual futures behind the INTX program. GPT-Trader ships with spot trading enabled by default and keeps the perps logic described here in a **dormant, future-ready** state. Enable it only after your account is approved for INTX and `COINBASE_ENABLE_DERIVATIVES=1` is set.
>
> 📘 **Trust reminder:** Confirm module references against `docs/agents/Document_Verification_Matrix.md` and `docs/ARCHITECTURE.md` before acting on this guide.

## Executive Summary

This document captures the architecture and trading flow for the Coinbase Perpetuals (Perps) implementation that remains in the repository. While the code paths compile and are exercised by tests, live execution is disabled until INTX access is granted. Use this reference when preparing to activate derivatives or when maintaining the dormant perps modules.

## System Architecture Overview

### Core Components

1. **TradingBot** (`src/gpt_trader/orchestration/trading_bot/bot.py`)
   - Main orchestration layer shared with spot trading
   - Detects `COINBASE_ENABLE_DERIVATIVES` + INTX context before enabling perps
   - Coordinates between strategy, risk, and execution layers via the coordinator pattern

2. **CoinbaseBrokerage Adapter** (`src/gpt_trader/features/brokerages/coinbase/adapter.py`)
   - Implements broker-agnostic interface
   - Handles REST API and WebSocket connections
   - Manages product discovery and order routing

3. **Strategy Layer**
   - **Active:** `src/gpt_trader/features/live_trade/strategies/perps_baseline/strategy.py` (instantiated by `StrategyOrchestrator`)
     - Baseline MA crossover with optional confirmation windows
     - Maintains trailing stops and per-symbol position counters
     - Delegates all exposure checks to the risk engine
   - **Experimental:** `src/gpt_trader/features/live_trade/strategies/perps_baseline_enhanced/strategy.py`
     - Adds liquidity filters, dedicated state management, and additional guards; kept for future iteration but not wired into the bot by default

4. **Risk Management** (`src/gpt_trader/features/live_trade/risk/`)
   - `manager.py` orchestrates leverage limits, liquidation buffers, and exposure controls (defaults mirror spot until INTX is approved)
   - `pre_trade_checks.py` enforces synchronous validation before orders leave the process
   - `runtime_monitoring.py` and `state_management.py` maintain guardrails and reduce-only transitions
   - Daily loss limits and kill-switch support remain available regardless of product type

5. **Execution Engine** (`src/gpt_trader/orchestration/live_execution.py`)
   - Market and limit order support with side-aware quantization
   - TIF support (GTC, IOC)
   - Slippage guard rails based on snapshot depth
   - Optional order preview plumbing

## Trading Logic Flow

### 1. Initialization Phase
```
1. Load configuration (dev/demo/prod profiles)
2. Initialize broker connection (REST + WebSocket)
3. Discover available perpetual products dynamically
4. Set up risk manager with configured limits
5. Initialize strategy with parameters
6. Create execution engine with order configuration
7. Reconcile state with exchange on startup
```

### 2. Market Data Collection
```
1. REST-first: fetches quotes every cycle (fallback when streaming is disabled)
2. Optional WebSocket threads for trade/orderbook feeds (INTX-ready, off by default)
3. Maintain mark price windows for technical indicators
4. Track funding rates for perpetuals via catalog refresh
5. Monitor market depth and spread when WS snapshots are available
6. Detect stale data conditions through risk manager timestamps
```

### 3. Signal Generation

#### Baseline MA Crossover Strategy
- **Entry Signals:**
  - Bullish: Short MA crosses above Long MA
  - Bearish: Short MA crosses below Long MA (if shorts enabled)
  - Epsilon tolerance for robust crossover detection
  - Optional confirmation bars requirement

- **Market Condition Filters (optional via enhanced strategy):**
  - Max spread: 10 bps (configurable)
  - Min L1 depth: $50,000
  - Min L10 depth: $200,000
  - Min 1m volume: $100,000
  - RSI confirmation (oversold < 30, overbought > 70)

- **Risk Guards (baseline relies on risk engine; enhanced strategy layers on extra checks):**
  - Min liquidation buffer: 20%
  - Max slippage impact: 15 bps

- **Exit Signals:**
  - Opposite MA crossover
  - Trailing stop hit (1% default)
  - Risk manager intervention

### 4. Order Execution

#### Pre-Trade Validation
```python
1. Check kill switch status
2. Verify reduce-only mode compliance
3. Validate leverage limits (max 3x default)
4. Check liquidation distance
5. Verify exposure limits (20% per symbol)
6. Confirm daily loss limits not breached
```

#### Order Placement
```python
1. Calculate position size based on:
   - Account equity
   - Target leverage
   - Product constraints (min size, increments)

2. Quantize order parameters:
   - Round size to step_size
   - Round price to price_increment
   - Ensure min_notional compliance

3. Generate client order ID (UUID)

4. Submit order with:
   - Market or limit type
   - Time-in-force (GTC/IOC)
   - Reduce-only flag if applicable
   - Post-only flag for maker orders
```

#### Order Management
- Track open orders in local store
- Monitor fills via WebSocket user channel
- Handle partial fills and rejections
- Implement cancel/replace flows
- Maintain order audit trail

### 5. Position Management

#### Position Tracking
```python
- Real-time position updates from exchange
- Track entry price, size, side
- Calculate unrealized PnL with funding
- Monitor margin requirements
- Update trailing stop levels
```

#### Risk Monitoring
```python
- Leverage ratio checks every cycle
- Liquidation price distance monitoring
- Exposure concentration limits
- Daily PnL tracking against limits
- Runtime guards for anomaly detection
```

## Configuration Profiles

### Development Profile
```python
BotConfig(
    profile=Profile.DEV,
    mock_broker=True,  # toggles the DeterministicBroker stub
    mock_fills=True,
    dry_run=True,
    max_position_size=Decimal("10000"),
    target_leverage=1,
    reduce_only_mode=True,
)
```

### Demo Profile (spot-only by default)
```python
BotConfig(
    profile=Profile.DEMO,
    max_position_size=Decimal("100"),
    max_leverage=1,
    enable_shorts=False,
    mock_broker=True,  # toggles the DeterministicBroker stub
)
```

### Production Profile (derivatives enabled only after INTX approval)
```python
BotConfig(
    profile=Profile.PROD,
    max_position_size=Decimal("5000"),
    max_leverage=3,
    enable_shorts=True,
    reduce_only_mode=False,
)
```

Setting `mock_broker=True` in any profile now routes through the
`DeterministicBroker` safety stub. This keeps development and demo runs free of
real orders while exercising the same orchestration paths used in production.

### Canary Profile (Ultra-Safe Production)
```yaml
trading:
  symbols: ["BTC-PERP"]
  mode: reduce_only
  position_sizing:
    max_notional_value: 500
risk_management:
  max_leverage: 1
  daily_loss_limit: 10
```

## Error Handling & Edge Cases

### Connection Management
- Automatic WebSocket reconnection with exponential backoff
- Sequence gap detection for missed messages
- Staleness detection (30s timeout)
- Fallback to reduce-only on stale data

### Order Failures
- Insufficient funds → Log and skip
- Min size violations → Adjust to minimum
- Rate limiting → Exponential backoff
- Network errors → Retry with idempotent IDs

### Risk Interventions
- Daily loss limit hit → Enter reduce-only mode
- Leverage exceeded → Force position reduction
- Kill switch activated → Cancel all orders, halt trading
- High error rate → Emergency shutdown

## Validation & Testing

### Week 1 Validations
1. **Product Discovery** - Dynamic enumeration of BTC/ETH/SOL/XRP perps
2. **Order Placement** - Market/limit orders with proper quantization
3. **TIF Support** - GTC/IOC mapping and enforcement
4. **WebSocket Streaming** - Real-time market data and user updates

### Integration Tests
- End-to-end trading cycle simulation
- Mock broker for deterministic testing
- Sandbox environment validation
- Performance benchmarking

### Runtime Validations
```bash
# Sandbox validation (archived helper — recover from git history if needed)
# RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_perps_client_week1.py

# WebSocket validation (archived helper)
# RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_ws_week1.py

# Full cycle test
poetry run coinbase-trader run --profile dev --dry-run --dev-fast
```

## Performance Metrics

### Rejection Tracking
```python
rejection_counts = {
    'filter_spread': 0,      # Spread too wide
    'filter_depth': 0,       # Insufficient liquidity
    'filter_volume': 0,      # Low volume
    'filter_rsi': 0,         # RSI not confirming
    'guard_liquidation': 0,  # Too close to liquidation
    'guard_slippage': 0,     # High impact cost
    'stale_data': 0,         # Market data stale
    'entries_accepted': 0    # Successful entries
}
```

### Key Performance Indicators
- Order acceptance rate
- Average fill latency
- Slippage statistics
- WebSocket message throughput
- Risk limit utilization

## Security Considerations

### Authentication
- HMAC signing for legacy Exchange API
- JWT for CDP (Coinbase Developer Platform)
- API key rotation support
- Passphrase protection

### Data Protection
- No credentials in logs
- Secure environment variable handling
- Encrypted event store
- Audit trail maintenance

## Deployment Recommendations

### Production Checklist
1. ✅ Set appropriate API credentials
2. ✅ Configure risk limits for account size
3. ✅ Enable monitoring and alerts
4. ✅ Set up health check endpoints
5. ✅ Configure data persistence paths
6. ✅ Test emergency shutdown procedures
7. ✅ Validate network connectivity
8. ✅ Review position size calculations

### Monitoring Setup
`EVENT_STORE_ROOT` should point to the parent runtime directory (for example `/srv/gpt-trader-runtime`). The bot automatically creates `coinbase_trader/{profile}` underneath it (and continues writing to the legacy `trading_bot/{profile}` path during the migration window), so do not append the profile when exporting the variable.
```bash
# Health status location
$EVENT_STORE_ROOT/coinbase_trader/{profile}/health.json

# Event logs
$EVENT_STORE_ROOT/coinbase_trader/{profile}/events/

# Order tracking
$EVENT_STORE_ROOT/coinbase_trader/{profile}/orders/
```

### Operational Commands
```bash
# Start production trading
poetry run coinbase-trader run --profile prod

# Emergency reduce-only mode
poetry run coinbase-trader run --profile prod --reduce-only

# Canary deployment
poetry run coinbase-trader run --profile canary

# Single cycle validation
poetry run coinbase-trader run --profile prod --dev-fast --dry-run
```

## Future Enhancements

### Planned Improvements
1. **Advanced Strategies**
   - Multi-timeframe analysis
   - Volume profile integration
   - Order flow analysis
   - Machine learning signals

2. **Risk Management**
   - Portfolio-level risk metrics
   - Correlation-based position limits
   - Dynamic leverage adjustment
   - Options hedging integration

3. **Execution Optimization**
   - Smart order routing
   - Iceberg orders
   - TWAP/VWAP algorithms
   - Cross-exchange arbitrage

4. **Monitoring & Analytics**
   - Real-time P&L dashboard
   - Strategy performance attribution
   - Risk exposure heatmaps
   - Automated reporting

## Conclusion

The Coinbase Perpetuals trading system is a robust, production-ready implementation with comprehensive risk management, advanced order types, and sophisticated market analysis capabilities. The modular architecture allows for easy extension and customization while maintaining safety through multiple layers of validation and risk controls.

### Key Strengths
- ✅ Dynamic product discovery without hardcoding
- ✅ Robust error handling and recovery
- ✅ Comprehensive risk management
- ✅ Real-time WebSocket integration
- ✅ Multiple deployment profiles
- ✅ Extensive validation coverage

### Risk Factors
- ⚠️ Market data staleness in low liquidity
- ⚠️ Funding rate impact on long-term positions
- ⚠️ Network latency in volatile markets
- ⚠️ Slippage during large moves

### Recommendation
The system is ready for production deployment with appropriate risk limits and monitoring. Start with the canary profile for initial production testing, then gradually scale up position sizes and complexity as confidence builds.

---

*Report Generated: December 2024*
*Version: 2.0 (Week 3 Enhanced)*
*Author: GPT-Trader Development Team*
