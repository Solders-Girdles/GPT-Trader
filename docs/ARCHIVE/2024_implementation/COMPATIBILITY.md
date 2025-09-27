# Exchange Compatibility Guide

## Coinbase Advanced Trade API Support

### Fully Supported Order Types

| Order Type | API Support | Implementation Status | Notes |
|------------|-------------|----------------------|-------|
| Market | ✅ Full | ✅ Implemented | Standard market orders |
| Limit | ✅ Full | ✅ Implemented | With post-only option |
| Stop Loss | ✅ Full | ✅ Implemented | Stop market orders |
| Stop Limit | ✅ Full | ✅ Implemented | Stop orders with limit price |

### Time-In-Force (TIF) Support

| TIF | API Support | Implementation Status | Notes |
|-----|-------------|----------------------|-------|
| GTC (Good Till Cancelled) | ✅ Full | ✅ Implemented | Default for most orders |
| IOC (Immediate or Cancel) | ✅ Full | ✅ Implemented | For immediate fills |
| GTD (Good Till Date) | ✅ Full | ⚠️ Not implemented | Planned for future |
| FOK (Fill or Kill) | ❌ Not supported | 🚫 Gated | Not available on Coinbase AT |

### Advanced Features

| Feature | API Support | Implementation Status | Notes |
|---------|-------------|----------------------|-------|
| Post-Only | ✅ Full | ✅ Implemented | Maker-only for limit orders |
| Reduce-Only | ✅ Full | ✅ Implemented | For closing positions |
| Client Order ID | ✅ Full | ✅ Implemented | Idempotency support |
| Leverage | ✅ Perpetuals | ✅ Implemented | For perpetual contracts |

### Perpetual Contracts

| Symbol | Status | Features | Notes |
|--------|--------|----------|-------|
| BTC-PERP | ✅ Active | Funding, leverage | Bitcoin perpetual |
| ETH-PERP | ✅ Active | Funding, leverage | Ethereum perpetual |
| SOL-PERP | ✅ Active | Funding, leverage | Solana perpetual |
| XRP-PERP | ✅ Active | Funding, leverage | Ripple perpetual |

### Funding Mechanics

- **Interval**: 8 hours (00:00, 08:00, 16:00 UTC)
- **Rate Convention**: 
  - Positive rate: Longs pay shorts
  - Negative rate: Shorts pay longs
- **Calculation**: `Notional Value × Funding Rate`
- **Implementation**: ✅ Full support with accrual tracking

## Implementation Details

### Order Type Handling

```python
# Market orders - always supported
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.01"),
    order_type="market"
)

# Limit orders with post-only
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.01"),
    order_type="limit",
    limit_price=Decimal("49990"),
    post_only=True  # Reject if would cross
)

# Stop orders (stop-loss)
order = engine.place_order(
    symbol="BTC-PERP",
    side="sell",
    quantity=Decimal("0.01"),
    order_type="stop",
    stop_price=Decimal("48000")
)

# Stop-limit orders
order = engine.place_order(
    symbol="BTC-PERP",
    side="sell",
    quantity=Decimal("0.01"),
    order_type="stop_limit",
    stop_price=Decimal("48000"),
    limit_price=Decimal("47900")
)
```

### Unsupported Features Handling

When an unsupported feature is requested, the engine handles it gracefully:

1. **FOK Orders**: 
   - Returns `None` with clear log message
   - Increments `rejected` counter
   - Message: "FOK order type is gated and not yet supported"

2. **GTD Orders**:
   - Currently not implemented
   - Would require date/time parameter
   - Planned for future release

### Local Emulation

For features not directly supported by the exchange, we provide local emulation:

1. **Stop Trigger Tracking**:
   - Engine maintains `StopTrigger` objects
   - `check_stop_triggers()` monitors price crosses
   - Automatically places market/limit when triggered

2. **Impact-Aware Sizing**:
   - Local calculation based on L1/L10 depth
   - Three modes: Conservative, Strict, Aggressive
   - Prevents excessive slippage

## WebSocket Data

### Supported Channels

| Channel | Purpose | Implementation |
|---------|---------|----------------|
| ticker | Price updates | ✅ Normalized |
| matches | Trade data | ✅ Volume tracking |
| level2 | Order book | ✅ Depth calculation |

### Schema Normalization

The adapter includes schema normalization to handle variations:

```python
# Handles multiple channel name variants
'ticker': ['ticker', 'tickers', 'ticker_batch']
'match': ['match', 'matches', 'trade', 'trades']
'level2': ['level2', 'l2', 'l2update', 'orderbook']

# Field normalization
'price' → 'last_price', 'last', 'close'
'size' → 'quantity', 'qty', 'amount'
```

## Environment Variables

### Required for Production

```bash
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_PASSPHRASE=your_passphrase  # V2 API
```

### Optional Configuration

```bash
# Force sandbox mode
COINBASE_SANDBOX=1

# Enable derivatives
COINBASE_ENABLE_DERIVATIVES=1

# Force mock broker (for testing)
PERPS_FORCE_MOCK=1

# Enable sandbox validations
RUN_SANDBOX_VALIDATIONS=1

# Use real adapter (not mock)
USE_REAL_ADAPTER=1
```

## Testing Compatibility

### Mock Broker

For development and testing without API calls:

```python
from src.bot_v2.orchestration.mock_broker import MockBroker

broker = MockBroker()
# Provides realistic market data and order simulation
```

### Sandbox Testing

For integration testing with real API:

```bash
# Run with sandbox enabled
COINBASE_SANDBOX=1 python scripts/run_perps_bot_v2.py

# Run validation suite
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_week3_orders.py
```

## Migration Path

### From V2 to V3 API

1. **Authentication**: V3 uses JWT instead of HMAC
2. **Endpoints**: Different URL structure
3. **Order Schema**: New `order_configuration` format
4. **WebSocket**: Different subscription format

Current implementation uses V2 (Advanced Trade) which is stable and recommended.

## Known Limitations

1. **FOK Orders**: Not supported by Coinbase AT
2. **GTD Orders**: Not yet implemented
3. **OCO Orders**: Not supported by exchange
4. **Iceberg Orders**: Not available
5. **Bracket Orders**: Must be emulated locally

## Troubleshooting

### Common Issues

1. **"Order type not supported"**
   - Check COMPATIBILITY.md for supported types
   - Ensure correct parameter names
   - Verify TIF is valid

2. **"Post-only would cross"**
   - Limit price would immediately match
   - Adjust price further from spread
   - Check current bid/ask

3. **"Insufficient depth for sizing"**
   - Market lacks liquidity
   - Reduce position size
   - Switch to CONSERVATIVE mode

4. **"Stop not triggering"**
   - Verify stop direction (buy stop > price, sell stop < price)
   - Check price feed is updating
   - Ensure trigger tracking is active

## Support

For exchange-specific issues:
- Coinbase Advanced Trade Docs: https://docs.cloud.coinbase.com/advanced-trade-api/
- API Status: https://status.coinbase.com/

For implementation issues:
- Check logs in `logs/` directory
- Enable debug logging: `--log-level DEBUG`
- Run validation scripts in `scripts/`