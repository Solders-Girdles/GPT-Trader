# Phase 2: Demo Profile Execution Guide

## Pre-Flight Checklist

### 1. Environment Setup
```bash
# Required environment variables
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_CDP_API_KEY='your_key'
export COINBASE_CDP_PRIVATE_KEY='your_private_key'
export EVENT_STORE_ROOT=/tmp/week3_eventstore

# Disable mocks
unset PERPS_FORCE_MOCK
unset COINBASE_SANDBOX
```

### 2. Run Pre-Flight Check
```bash
bash scripts/phase2_preflight_check.sh
```

Expected output:
- ✅ Credentials set
- ✅ System clock synchronized
- ✅ DNS resolution OK
- ✅ HTTPS connectivity OK
- ✅ Broker connection successful
- ✅ Products listed
- ✅ Quote received

## Phase 2 Execution Steps

### Step 1: Dry-Run Pulse (1 cycle)
Test connectivity and filters without placing orders:

```bash
python scripts/run_perps_bot_v2_week3.py \
  --profile demo \
  --symbols BTC-PERP \
  --dry-run \
  --run-once \
  --log-level INFO
```

**Verify**:
- WebSocket snapshots received
- Market data (spread, depth, volume) logged
- Filters working (RSI, spread, depth checks)
- No errors in connection

### Step 2: Post-Only Limit Test
Place a non-crossing post-only limit order:

```bash
python scripts/phase2_demo_runner.py
```

**Features**:
- $25-100 tiny positions
- Post-only limits (10 bps below bid)
- Auto-cancel after 30 seconds if unfilled
- Pre-funding quiet period (30 min)
- Daily loss limit ($100)
- Kill switch ready (Ctrl+C)

**Monitor** (in separate terminal):
```bash
python scripts/phase2_metrics_monitor.py
```

### Step 3: Market Order Test (Optional)
Test tiny market order with reduce-only exit:

```bash
python scripts/run_perps_bot_v2_week3.py \
  --profile demo \
  --symbols BTC-PERP \
  --order-type market \
  --sizing-mode conservative \
  --max-impact-bps 10 \
  --target-notional 50 \
  --run-once
```

### Step 4: Stop Order Test
Place far stop-limit and cancel immediately:

```bash
python -c "
import sys
sys.path.insert(0, '.')
from decimal import Decimal
from src.bot_v2.orchestration.broker_factory import create_brokerage
from src.bot_v2.features.live_trade.execution_v3 import AdvancedExecutionEngine

broker = create_brokerage()
broker.connect()
engine = AdvancedExecutionEngine(broker)

# Place far stop
order = engine.place_order(
    symbol='BTC-PERP',
    side='sell',
    quantity=Decimal('0.001'),
    order_type='stop',
    stop_price=Decimal('40000'),  # Far below market
    client_id='test_stop_demo'
)

if order:
    print(f'Stop placed: {order.id}')
    # Cancel immediately
    if broker.cancel_order(order.id):
        print('Stop cancelled')
"
```

## Metrics to Monitor

### 1. Engine Metrics
- `orders_placed` - Total attempted
- `orders_filled` - Successfully executed
- `orders_cancelled` - Manually cancelled
- `orders_rejected` - Broker rejections
- `post_only_rejected` - Crossing rejections

### 2. Strategy Metrics
- Acceptance rate (> 20% good)
- Rejection breakdown:
  - `spread_too_wide` - Spread > max_spread_bps
  - `insufficient_depth` - L1/L10 below minimums
  - `low_volume` - 1m/5m volume below minimums
  - `rsi_not_confirmed` - RSI signal not met
  - `stale_data` - WebSocket data too old

### 3. PnL Metrics
- Realized PnL per trade
- Unrealized PnL on positions
- Funding paid/received
- Daily metrics:
  - Win rate
  - Max drawdown
  - Sharpe ratio

### 4. Latency Metrics
- `broker.place_order` latency
- WebSocket staleness
- Network errors
- Retry/backoff counts

## Safety Guardrails

### Kill Switch
```bash
# Immediate stop
bash scripts/emergency_kill_switch.sh

# Or reduce-only mode
kill -USR1 <pid>  # Switches to reduce-only
```

### Position Limits
- Max position size: 0.01 BTC
- Max notional: $100 per order
- Leverage: 1x (no leverage)
- Daily loss limit: $100

### Rate Limits
- Max 10 orders per minute
- Use post-only to avoid taker fees
- Exponential backoff on errors

## Success Criteria

Phase 2 is successful if:

| Metric | Target | Actual |
|--------|--------|--------|
| Connection stable | 1 hour | __ |
| Orders placed | > 5 | __ |
| Fill rate | > 20% | __ |
| Post-only working | Yes | __ |
| Cancel/replace working | Yes | __ |
| Stop orders logged | Yes | __ |
| PnL tracking accurate | Yes | __ |
| No unexpected errors | Yes | __ |

## Troubleshooting

### JWT Authentication Issues
- Verify system clock is synchronized
- Check private key format (PEM)
- Ensure CDP API key is valid

### WebSocket Staleness
- Increase staleness threshold to 15s
- Monitor reconnection logs
- Check network stability

### Post-Only Rejections
- Increase offset to 15-20 bps
- Check bid/ask spread
- Verify reject_on_cross enabled

### Funding Time Issues
- Check `next_funding_time` in product
- Implement 30 min quiet period
- Log funding accruals

## Next Steps

After successful Phase 2:

1. **Review all metrics** from health file and EventStore
2. **Verify PnL accuracy** against exchange
3. **Check latency** is acceptable (< 500ms)
4. **Confirm safety features** worked (kill switch, reduce-only)
5. **Document any issues** for resolution

If all criteria met, proceed to Phase 3 (Production Canary):
```bash
bash scripts/production_canary.sh
```

---

**Remember**: Start small, monitor closely, kill switch ready!