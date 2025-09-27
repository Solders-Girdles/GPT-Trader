# Advanced Order Types Guide

## Overview

Week 3 introduces production-grade order management with advanced order types, impact-aware sizing, and comprehensive execution controls.

## Supported Order Types

### Market Orders
- **Type**: `market`
- **TIF**: GTC, IOC
- **Features**: Immediate execution at best available price
- **Use Case**: Quick entries/exits when price is less important than execution

### Limit Orders
- **Type**: `limit`
- **TIF**: GTC, IOC
- **Features**:
  - Price quantization to exchange increments
  - Post-only maker protection
  - Automatic offset calculation from bid/ask
- **Use Case**: Better price execution, maker rebates

### Stop Orders
- **Type**: `stop`
- **TIF**: GTC
- **Features**:
  - Trigger tracking
  - Automatic trigger detection
- **Use Case**: Stop-loss, breakout entries

### Stop-Limit Orders
- **Type**: `stop_limit`
- **TIF**: GTC
- **Features**:
  - Dual price validation
  - Quantized stop and limit prices
- **Use Case**: Controlled stop-loss with price protection

## Time-In-Force (TIF) Options

| TIF | Description | Supported | Notes |
|-----|-------------|-----------|-------|
| GTC | Good Till Cancelled | ✅ | Default for most orders |
| IOC | Immediate or Cancel | ✅ | Fills immediately or cancels |
| FOK | Fill or Kill | ❌ | Gated - not yet supported |

## Post-Only Protection

For limit orders, post-only ensures you're always a maker:

```python
# Will reject if limit price would cross
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.01"),
    order_type="limit",
    limit_price=Decimal("49990"),
    post_only=True
)
```

**Rejection Scenarios**:
- Buy limit >= ask price
- Sell limit <= bid price

## Impact-Aware Sizing

Three modes for managing market impact:

### Conservative Mode (Default)
```python
config = OrderConfig(
    sizing_mode=SizingMode.CONSERVATIVE,
    max_impact_bps=Decimal("15")
)
```
- Automatically reduces order size to fit within impact limit
- Logs "sized_down" when adjustment occurs
- Best for minimizing slippage

### Strict Mode
```python
config = OrderConfig(
    sizing_mode=SizingMode.STRICT,
    max_impact_bps=Decimal("10")
)
```
- Rejects orders that would exceed impact limit
- Returns zero size if cannot fit
- Best for precise execution requirements

### Aggressive Mode
```python
config = OrderConfig(
    sizing_mode=SizingMode.AGGRESSIVE,
    max_impact_bps=Decimal("15")
)
```
- Allows up to 2x the configured impact limit
- Prioritizes execution over price
- Best for urgent trades

## Cancel and Replace

Atomic cancel/replace with idempotent client IDs:

```python
# Original order
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.01"),
    order_type="limit",
    limit_price=Decimal("49000"),
    client_id="original_123"
)

# Cancel and replace with new price
new_order = engine.cancel_and_replace(
    order_id=order.id,
    new_price=Decimal("49500"),
    new_size=Decimal("0.02")
)
```

**Features**:
- Exponential backoff retry logic
- Idempotent client IDs prevent duplicates
- Atomic operation (cancel then place)

## CLI Examples

### Basic Market Order
```bash
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP \
    --order-type market
```

### Limit Order with Post-Only
```bash
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP \
    --order-type limit \
    --limit-offset-bps 5 \
    --post-only
```

### Stop-Loss Order
```bash
python scripts/run_perps_bot_v2.py \
    --symbols ETH-PERP \
    --order-type stop \
    --stop-pct 2
```

### Conservative Sizing
```bash
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP \
    --sizing-mode conservative \
    --max-impact-bps 10
```

## Validation

Run the Week 3 validation script to test all features:

```bash
# Mock tests only
python scripts/validate_week3_orders.py

# Include sandbox tests
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_week3_orders.py
```

## Integration with Runner

The runner (`scripts/run_perps_bot_v2.py`) should be updated to use the `AdvancedExecutionEngine`:

```python
from src.bot_v2.features.live_trade.execution_v3 import (
    AdvancedExecutionEngine, OrderConfig, SizingMode
)

# Configure advanced orders
order_config = OrderConfig(
    enable_limit_orders=True,
    enable_stop_orders=True,
    enable_post_only=True,
    sizing_mode=SizingMode.CONSERVATIVE,
    max_impact_bps=Decimal(args.max_impact_bps)
)

# Initialize engine
exec_engine = AdvancedExecutionEngine(
    broker=broker,
    config=order_config
)

# Use impact-aware sizing
adjusted_size, impact = exec_engine.calculate_impact_aware_size(
    target_notional=target_notional,
    market_snapshot=market_snapshot
)

# Place order with advanced features
order = exec_engine.place_order(
    symbol=symbol,
    side=side,
    quantity=adjusted_size / current_mark,
    order_type=args.order_type,
    limit_price=limit_price if args.order_type == "limit" else None,
    stop_price=stop_price if "stop" in args.order_type else None,
    time_in_force=args.tif,
    post_only=args.post_only,
    reduce_only=is_exit
)
```

## Metrics and Monitoring

The engine tracks comprehensive metrics:

```python
metrics = exec_engine.get_metrics()
# {
#     'orders': {
#         'placed': 10,
#         'filled': 8,
#         'cancelled': 1,
#         'rejected': 1,
#         'post_only_rejected': 2,
#         'stop_triggered': 3
#     },
#     'pending_count': 2,
#     'stop_triggers': 3,
#     'active_stops': 1
# }
```

## Safety Features

1. **Price Quantization**: All prices rounded to exchange increments
2. **Duplicate Prevention**: Idempotent client IDs prevent double orders
3. **Cross Protection**: Post-only orders rejected if they would cross
4. **Impact Limits**: Automatic sizing to prevent excessive slippage
5. **Validation**: Stop/limit price relationships validated

## Troubleshooting

### Post-Only Rejections
- Check bid/ask spread
- Ensure limit price won't cross
- Use larger offset from mid

### Stop Not Triggering
- Verify trigger price is set correctly
- Check that price updates are flowing
- Ensure stop direction matches intent

### Impact Sizing Issues
- Verify market depth data is available
- Check L1/L10 depths are in USD notional
- Adjust max_impact_bps if too restrictive

## Capability Detection

Before deploying, verify broker support for advanced features:

```bash
# Test against mock broker
python scripts/probe_capabilities.py

# Test against sandbox
COINBASE_SANDBOX=1 python scripts/probe_capabilities.py --live

# Test specific symbol
python scripts/probe_capabilities.py --symbol ETH-PERP
```

The probe tests all order types and features, reporting:
- ✅ Supported and verified
- ❌ Not supported or failed
- 🚫 Gated (like FOK orders)

## Next Steps

1. **Capability Verification**: Run probe before production
2. **Sandbox Testing**: Validate all features work as expected
3. **Production Deployment**: Enable in live trading with conservative limits