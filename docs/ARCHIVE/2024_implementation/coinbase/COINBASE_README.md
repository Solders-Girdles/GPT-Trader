# Coinbase Brokerage Integration

## Overview
This module provides integration with Coinbase Advanced Trade API for cryptocurrency trading, including perpetual futures support.

## Features
- Full Advanced Trade API v3 support
- WebSocket real-time data feeds
- Rate limiting and throttling
- Automatic retries with exponential backoff
- Connection pooling and keep-alive
- Product catalog with trading rules
- Order placement and management
- Portfolio tracking
- **Derivatives/Perpetuals support (Phase 1)** - See [Perpetuals Trading Logic Report](PERPS_TRADING_LOGIC_REPORT.md)

## Performance Optimizations

### Connection Reuse (Keep-Alive)
The client automatically reuses HTTP connections to reduce latency:

```python
# Keep-alive is enabled by default
client = CoinbaseClient(
    base_url="https://api.coinbase.com",
    enable_keep_alive=True  # Default
)

# Disable if needed for debugging
client = CoinbaseClient(
    base_url="https://api.coinbase.com",
    enable_keep_alive=False
)
```

**Benefits:**
- Reduces TCP handshake overhead
- Improves request latency by 20-40ms per request
- Reduces server load

**Debugging Note:**
If you experience connection issues with proxies or corporate firewalls, try disabling keep-alive:
```python
# For debugging proxy/firewall issues
client = CoinbaseClient(
    base_url="https://api.coinbase.com",
    enable_keep_alive=False  # Disable connection reuse
)

# Or via environment variable
COINBASE_ENABLE_KEEP_ALIVE=0
```
Common scenarios where disabling might help:
- Corporate proxies that don't support persistent connections
- Load balancers with aggressive connection timeouts
- Debugging "connection reset" errors

### Backoff with Jitter
Implements exponential backoff with deterministic jitter to prevent thundering herd:

```python
# Configure in system config
{
    "system": {
        "max_retries": 3,
        "retry_delay": 1.0,
        "jitter_factor": 0.1  # 10% jitter
    }
}

# Or disable jitter for testing
{
    "system": {
        "jitter_factor": 0  # No jitter
    }
}
```

**Jitter calculation:**
- Deterministic based on attempt number (for reproducible tests)
- Formula: `delay = base_delay * 2^attempt + (jitter_factor * delay * (attempt % 10) / 10)`
- Example: Attempt 1 with 1s base and 0.1 jitter = 1.01s
- Example: Attempt 2 with 1s base and 0.1 jitter = 2.04s

### Rate Limiting
Built-in rate limiting with sliding window:

```python
client = CoinbaseClient(
    base_url="https://api.coinbase.com",
    rate_limit_per_minute=100,  # Default
    enable_throttle=True  # Auto-throttle when approaching limit
)
```

**Features:**
- Sliding window tracking
- Warning at 80% of limit
- Automatic throttling at limit
- Per-minute request counting

## Derivatives Prerequisites

### Requirements
- **CDP JWT Authentication**: Required for derivatives access
- **Derivatives Permissions**: Must be enabled on your API key
- **Advanced Mode Only**: Set `COINBASE_API_MODE=advanced`
- **Feature Flag**: Set `COINBASE_ENABLE_DERIVATIVES=1`
- **KYC/Compliance**: Account must be eligible for derivatives trading

### Phase 1: Derivatives API Enablement ✅
- ✅ Read CFM positions via `cfm_positions()`
- ✅ Close positions via `close_position()`
- ✅ Leverage and reduce_only support in orders
- ✅ Balance summary and margin window queries

### Phase 2: Product Catalog & Metadata ✅
- ✅ Extended Product model with perps fields (contract_size, funding_rate, next_funding_time)
- ✅ Enhanced `to_product()` mapper for derivatives metadata
- ✅ ProductCatalog with `refresh()` for perps products
- ✅ `get_funding()` helper for funding rate/time queries
- ✅ `enforce_perp_rules()` for order validation and quantization

### Phase 3: WebSocket Integration ✅
- ✅ Market data streaming for perpetuals (trades, ticker, level2)
- ✅ Authenticated user events with JWT support
- ✅ Automatic reconnection with resubscription
- ✅ SequenceGuard for gap detection and reset on reconnect
- ✅ Message normalization with Decimal prices/sizes
- ✅ Liveness monitoring with configurable timeout

### Phase 4: PnL and Funding Accrual ✅
- ✅ **MarkCache**: TTL-based mark price caching for perpetuals
- ✅ **FundingCalculator**: Discrete funding accrual at scheduled times
- ✅ **PositionState**: Position tracking with realized/unrealized PnL
- ✅ **Streaming Integration**: Mark price updates from trades/ticker
- ✅ **Fill Processing**: Automatic PnL updates from user events
- ✅ **Event Persistence**: Metrics and funding events to EventStore
- ✅ **Portfolio Aggregation**: Total PnL across all positions

### Phase 5: Risk Engine ✅
- ✅ **RiskConfig**: Comprehensive risk configuration
- ✅ **Leverage Caps**: Global and per-symbol leverage limits
- ✅ **Liquidation Buffer**: Minimum buffer requirement from liquidation
- ✅ **Exposure Limits**: Per-symbol and total portfolio exposure caps
- ✅ **Daily Loss Guard**: Automatic reduce-only on daily loss breach
- ✅ **Slippage Guard**: Reject orders with excessive expected slippage
- ✅ **Kill Switch**: Emergency halt for all trading
- ✅ **Runtime Monitoring**: Continuous risk checks with event logging

### Phase 6: Strategy Integration ✅
- ✅ **BaselinePerpsStrategy**: MA crossover with trailing stops
- ✅ **Position Sizing**: Leverage-based with RiskManager constraints
- ✅ **Exit Management**: Reduce-only orders for all exits
- ✅ **Feature Flags**: enable_shorts, max_adds, disable_new_entries
- ✅ **Runner Integration**: run_strategy() in live_trade.py

### Upcoming Phases
- ⏳ Phase 7: E2E testing and validation

## Configuration

### API Modes
```python
# Advanced Trade API (default)
config = APIConfig(
    api_key="your_key",
    api_secret="your_secret",
    api_mode="advanced"
)

# Legacy Exchange API
config = APIConfig(
    api_key="your_key",
    api_secret="your_secret",
    passphrase="your_passphrase",
    api_mode="exchange"
)
```

### Sandbox Mode
```python
config = APIConfig(
    api_key="your_key",
    api_secret="your_secret",
    base_url="https://api-sandbox.coinbase.com",
    sandbox=True
)
```

## Testing

### Unit Tests
The module includes comprehensive unit tests with mocked transports:

```bash
# Run all Coinbase tests
pytest tests/unit/bot_v2/features/brokerages/coinbase/

# Run performance tests
pytest tests/unit/bot_v2/features/brokerages/coinbase/test_performance.py

# Run with coverage
pytest tests/unit/bot_v2/features/brokerages/coinbase/ --cov=src.bot_v2.features.brokerages.coinbase
```

### Test Hooks
For deterministic testing, the client provides several hooks:

1. **Transport Override**: Replace network layer
```python
client.set_transport_for_testing(mock_transport)
```

2. **Time Control**: Mock time.sleep for retry testing
```python
with patch('time.sleep') as mock_sleep:
    # Test retry delays
```

3. **Jitter Control**: Deterministic jitter based on attempt number

## Error Handling

### Exception Hierarchy
```
BrokerageError
├── AuthenticationError     # Invalid API credentials
├── InsufficientFunds      # Not enough balance
├── InvalidSymbol          # Unknown trading pair
├── InvalidOrder           # Order validation failed
├── OrderNotFound          # Order ID not found
├── RateLimitError         # Rate limit exceeded
├── InvalidRequestError    # Bad request format
└── BrokerageAPIError      # General API errors
```

### Retry Logic
Automatic retries for transient errors:
- Status 429: Rate limited (uses Retry-After header)
- Status 502-504: Gateway errors
- Network timeouts

## Risk Controls (Perpetuals)

### Configuration

Risk controls are configured via `RiskConfig` with environment variables or JSON:

```python
from src.bot_v2.config.live_trade_config import RiskConfig

# Load from environment
config = RiskConfig.from_env()

# Or create with specific limits
config = RiskConfig(
    leverage_max_global=10,
    leverage_max_per_symbol={"BTC-PERP": 5, "ETH-PERP": 8},
    min_liquidation_buffer_pct=0.15,  # 15% buffer
    max_daily_loss_pct=0.02,           # 2% daily loss limit
    max_exposure_pct=0.8,              # 80% max portfolio exposure
    max_position_pct_per_symbol=0.2,   # 20% max per symbol
    slippage_guard_bps=50,             # 50 bps slippage limit
    kill_switch_enabled=False,
    reduce_only_mode=False
)
```

### Pre-Trade Validation

All orders are validated before placement:

```python
from src.bot_v2.features.live_trade.risk import LiveRiskManager

risk_manager = LiveRiskManager(config=config)

# Validate order (raises ValidationError if rejected)
risk_manager.pre_trade_validate(
    symbol="BTC-PERP",
    side="buy",
    qty=Decimal("1.0"),
    price=Decimal("50000"),
    product=product,
    equity=Decimal("100000"),
    current_positions=positions
)
```

### Runtime Guards

Monitor and enforce risk limits during trading:

```python
# Track daily PnL (triggers reduce-only if limit breached)
if risk_manager.track_daily_pnl(equity, positions_pnl):
    print("Daily loss limit breached - reduce-only mode enabled")
    # Cancel all open orders
    
# Check liquidation buffers
for symbol, position in positions.items():
    if risk_manager.check_liquidation_buffer(symbol, position, equity):
        print(f"Low buffer for {symbol} - reduce-only enabled")

# Check mark staleness
if risk_manager.check_mark_staleness("BTC-PERP"):
    print("Stale mark price - halting new orders")
```

### Reduce-Only Mode

When reduce-only mode is active (manually or triggered by risk breach):

- **New positions blocked**: Cannot open new positions
- **Increases blocked**: Cannot increase existing positions  
- **Reductions allowed**: Can only reduce or close positions
- **Orders forced reduce-only**: All orders automatically set `reduce_only=True`

Example:
```python
# After daily loss breach or manual activation
risk_manager.config.reduce_only_mode = True

# This order will be rejected
risk_manager.pre_trade_validate(
    symbol="BTC-PERP",
    side="buy",          # Would increase/open long
    qty=Decimal("1.0"),
    ...
)  # Raises: "Reduce-only mode active"

# This order is allowed
risk_manager.pre_trade_validate(
    symbol="BTC-PERP", 
    side="sell",         # Reduces existing long
    qty=Decimal("0.5"),
    ...
    current_positions={"BTC-PERP": {"side": "long", "qty": Decimal("1.0")}}
)  # Passes validation
```

### Risk Events

All risk events are logged to EventStore for audit:

```python
# Risk events are automatically logged:
# - daily_loss_breach: Daily loss limit exceeded
# - liquidation_buffer_breach: Buffer below minimum
# - stale_mark_price: Mark price too old
# - order_rejected: Pre-trade validation failure

# Query risk events
events = event_store.tail(
    bot_id="risk_engine",
    limit=100,
    types=["metric"]
)

risk_events = [e for e in events if e.get("type") == "risk_event"]
```

## PnL and Funding (Perpetuals)

### Mark Price Tracking

The adapter maintains a mark price cache for perpetuals with configurable TTL:

```python
# Mark prices are automatically updated from streaming
for trade in adapter.stream_trades(["BTC-USD-PERP"]):
    # Mark cache is updated internally
    mark = adapter._mark_cache.get_mark("BTC-USD-PERP")
    print(f"Current mark: {mark}")
```

### Position Tracking

Positions are automatically tracked when fills are processed:

```python
# Fills from user events update position state
for event in adapter.stream_user_events():
    if event.get('type') == 'fill':
        # Position is updated internally
        pnl = adapter.get_position_pnl(event['product_id'])
        print(f"Position PnL: {pnl}")
```

### PnL Retrieval

Get PnL for individual positions or entire portfolio:

```python
# Single position PnL
pnl = adapter.get_position_pnl("BTC-USD-PERP")
# Returns: {
#     'symbol': 'BTC-USD-PERP',
#     'qty': Decimal('1.0'),
#     'side': 'long',
#     'entry': Decimal('50000'),
#     'mark': Decimal('51000'),
#     'unrealized_pnl': Decimal('1000'),
#     'realized_pnl': Decimal('500'),
#     'funding_accrued': Decimal('-5')
# }

# Portfolio aggregation
portfolio = adapter.get_portfolio_pnl()
# Returns: {
#     'total_unrealized': Decimal('1500'),
#     'total_realized': Decimal('300'),
#     'total_funding': Decimal('-10'),
#     'positions': {...}
# }
```

### Funding Accrual

Funding is automatically calculated at scheduled times:

```python
# Funding is accrued when position metrics are updated
# Convention: Positive funding rate means longs pay shorts

# The adapter tracks:
# - Funding rate from product metadata
# - Next funding time
# - Position size and side
# - Mark price at funding time

# Funding events are persisted to EventStore for audit
```

### Event Persistence

All PnL and funding events are persisted to the EventStore:

```python
# Events are written to results/managed/events.jsonl
# Event types:
# - 'position': Position snapshots with PnL
# - 'metric' with type='funding': Funding accruals

# Query recent events
events = adapter._event_store.tail(
    bot_id="coinbase_perps",
    limit=100,
    types=["position", "metric"]
)
```

## WebSocket Support

### WebSocket Channels

| Channel | Description | Notes |
|---------|-------------|-------|
| `market_trades` | Real-time trade prints | Includes perpetuals; normalized with Decimal prices/sizes |
| `ticker` | Best bid/ask and last price (L1) | Used for level=1 orderbook streaming |
| `level2` | Full orderbook updates (L2) | Used for level>=2 orderbook streaming |
| `user` | Authenticated orders/fills | Requires JWT auth for derivatives; includes sequence gap detection |

### Market Data Streaming

```python
# Stream trades for perpetuals
for trade in adapter.stream_trades(["BTC-USD-PERP", "ETH-USD-PERP"]):
    print(f"Trade: {trade['product_id']} @ {trade['price']}")  # price is Decimal

# Stream orderbook (L1 ticker or L2 full book)
for update in adapter.stream_orderbook(["BTC-USD-PERP"], level=2):
    print(f"Book update: {update}")
```

### Authenticated User Events

```python
# Stream user events with auth (orders/fills)
for event in adapter.stream_user_events(["BTC-USD-PERP"]):
    if event.get("gap_detected"):
        print(f"Sequence gap detected! Last: {event['last_seq']}")
    print(f"Event: {event}")
```

### WebSocket Authentication

For the `user` channel with derivatives enabled:
- CDP JWT authentication is automatically handled when `auth_type="JWT"`
- The adapter creates a `ws_auth_provider` that generates JWT tokens
- Auth data is injected into the subscribe payload as `{"jwt": token}`

Example configuration:
```python
config = APIConfig(
    api_key="your_key",
    api_secret="your_secret",
    auth_type="JWT",
    cdp_api_key="your_cdp_key",
    cdp_private_key="your_private_key",
    enable_derivatives=True
)
```

### Reconnection & Reliability

- **Automatic Reconnection**: On disconnect, the WebSocket automatically reconnects with exponential backoff
- **Resubscription**: After reconnect, all previous subscriptions are restored
- **SequenceGuard**: Detects message gaps in `user` channel; resets on reconnection to avoid false positives
- **Liveness Monitoring**: Optional timeout detection if no messages received (configurable)
- **Message Normalization**: All market data messages have prices/sizes converted to `Decimal` for precision

## Best Practices

1. **Use Product Catalog**: Always check trading rules before placing orders
2. **Handle Rate Limits**: Monitor request counts and use throttling
3. **Implement Reconnection**: WebSocket connections may drop
4. **Validate Orders**: Use preview endpoint before placing orders
5. **Monitor Positions**: Track portfolio state locally
6. **Log Everything**: Enable debug logging for troubleshooting

## Troubleshooting

### Common Issues

**Authentication Errors:**
- Verify API key has correct permissions
- Check if using sandbox vs production credentials
- Ensure system time is synchronized (for HMAC)

**Rate Limiting:**
- Enable throttling: `enable_throttle=True`
- Reduce `rate_limit_per_minute` if needed
- Implement caching for frequently accessed data

**Connection Issues:**
- Check network connectivity
- Verify firewall allows HTTPS to api.coinbase.com
- Try disabling keep-alive if experiencing connection resets

**Order Failures:**
- Check product catalog for min/max sizes
- Verify account has sufficient balance
- Use correct order types for the product

## Strategy Integration (Perpetuals)

### Baseline Strategy

The `BaselinePerpsStrategy` provides a simple, production-safe strategy for perpetuals:

```python
from src.bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy, StrategyConfig, create_baseline_strategy
)

# Configure strategy
config = StrategyConfig(
    short_ma_period=5,
    long_ma_period=20,
    target_leverage=2,
    trailing_stop_pct=0.01,  # 1% trailing stop
    enable_shorts=False,      # Long-only by default
    max_adds=1,              # Prevent pyramiding
    disable_new_entries=False
)

# Create with risk manager
strategy = create_baseline_strategy(
    config=config.__dict__,
    risk_manager=risk_manager
)

# Generate decision
decision = strategy.decide(
    symbol="BTC-PERP",
    current_mark=Decimal("50000"),
    position_state=position_state,  # Current position or None
    recent_marks=mark_window,        # List of recent prices
    equity=Decimal("100000"),
    product=product
)

# Decision contains:
# - action: BUY/SELL/HOLD/CLOSE
# - target_notional: Position size in USD
# - leverage: Target leverage
# - reduce_only: True for exits
# - reason: Explanation string
```

### Strategy Behavior

**Entry Signals:**
- Long entry: Short MA crosses above long MA (bullish crossover)
- Short entry: Short MA crosses below long MA (bearish crossover, if `enable_shorts=True`)

**Exit Signals:**
- Opposing MA crossover (bearish for longs, bullish for shorts)
- Trailing stop hit (1% default, configurable)
- Reduce-only mode active (forced exit)

**Position Sizing:**
- Uses `target_leverage` * equity for notional
- Respects RiskManager leverage caps
- Applies ProductCatalog quantization rules

### Feature Flags

- `enable_shorts`: Allow short positions (default: False)
- `max_adds`: Maximum position additions per side (default: 1)
- `disable_new_entries`: Exit-only mode for validation (default: False)

### Safety Features

- All exits use `reduce_only=True` flag
- Respects global reduce-only mode from RiskManager
- Handles ValidationError from risk checks gracefully
- No persistent state beyond mark price windows
- Clear decision reasons for audit trail

## Support

For issues or questions:
1. Check the unit tests for usage examples
2. Review error messages and logs
3. Consult Coinbase API documentation
4. File an issue in the project repository