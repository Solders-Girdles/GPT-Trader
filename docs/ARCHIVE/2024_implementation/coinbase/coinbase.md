# Coinbase Brokerage Integration (Scaffold)

This document tracks the Coinbase Advanced Trade + derivatives integration.

## Scope
- Spot trading: balances, products, quotes, candles, orders (market/limit/stop/stop-limit), order status.
- Derivatives: futures/perpetual products, leverage, reduce-only, positions, fills.
- WebSocket streams: trades and order book with reconnection and resubscribe.

## Configuration
- `BROKER=coinbase`
- `COINBASE_API_KEY`
- `COINBASE_API_SECRET` (Advanced Trade returns base64 secret; paste as-is)
- `COINBASE_API_PASSPHRASE` (only if your key has one; often blank)
- `COINBASE_SANDBOX=1|0`
- `COINBASE_API_BASE` override (optional)
- `COINBASE_WS_URL` (optional; defaults included)
- `COINBASE_ENABLE_DERIVATIVES=0|1` (gated by permissions)

Example (.env.local; never commit):
```
BROKER=coinbase
COINBASE_SANDBOX=1
COINBASE_API_KEY=your-api-key
COINBASE_API_SECRET=your-base64-secret
COINBASE_API_PASSPHRASE=
```

Optional test toggles:
```
COINBASE_RUN_ORDER_TESTS=0
COINBASE_ORDER_SYMBOL=BTC-USD
COINBASE_TEST_LIMIT_PRICE=10
COINBASE_TEST_QTY=0.001
```

## Code Layout
- `src/bot_v2/features/brokerages/core/interfaces.py` – shared types and protocol
- `src/bot_v2/features/brokerages/coinbase/` – adapter, client, models, ws
- `src/bot_v2/orchestration/broker_factory.py` – selection by config/env

## Next Steps
1) Implement REST auth/signing and core endpoints
2) Product metadata caching and order validation
3) Order flows with retries/idempotency
4) WebSocket with reconnect/resubscribe
5) Tests (unit + opt-in integration)

## CLI Smoke
- Read-only:
```
python -m bot_v2.simple_cli broker --broker coinbase --sandbox
```
- Place + cancel (opt-in):
```
python -m bot_v2.simple_cli broker --broker coinbase --sandbox --run-order-tests --symbol BTC-USD --limit-price 10 --qty 0.001
```
