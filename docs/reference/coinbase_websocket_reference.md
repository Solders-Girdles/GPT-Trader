# Coinbase Advanced Trade API - WebSocket Reference

---
status: current
created: 2025-10-19
last-verified: 2025-10-19
verification-schedule: quarterly
scope: Advanced Trade WebSocket API (real-time market data and user updates)
documentation-venue: docs.cdp.coinbase.com/advanced-trade/docs/ws-overview
---

> **Maintenance Note**: WebSocket specifications and subscription limits verified quarterly. Last verified: 2025-10-19. Consult the official changelog at https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog for breaking changes.

## Overview

The Coinbase Advanced Trade WebSocket API provides real-time market data and user account updates. It's the recommended way to stream price updates, order book changes, and execution fills.

**WebSocket Endpoint (Production):**
```
wss://advanced-trade-ws.coinbase.com
```

**WebSocket Endpoint (Sandbox):**
```
wss://ws-feed-sandbox.exchange.coinbase.com
```

---

## Connection Setup

### Basic Connection

```python
import asyncio
import websockets
import json

async def connect_websocket():
    uri = "wss://advanced-trade-ws.coinbase.com"
    async with websockets.connect(uri) as websocket:
        # Subscribe to channels
        await websocket.send(json.dumps({
            "type": "subscribe",
            "channels": [
                {
                    "name": "ticker",
                    "product_ids": ["BTC-USD", "ETH-USD"]
                }
            ]
        }))

        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

asyncio.run(connect_websocket())
```

### Authentication (User Channel)

For private user account updates (`user` channel), authentication is required:

```python
import hmac
import hashlib
import base64
from datetime import datetime

def create_signature(timestamp, secret, path="/users/self/verify"):
    """Create WebSocket authentication signature

    ⚠️ CRITICAL: Message format is timestamp + "GET" + path (same as REST HMAC)
    NOT "subscribe" + timestamp!
    """
    message = f"{timestamp}GET{path}"
    message = message.encode('utf-8')
    hmac_key = base64.b64decode(secret)
    signature = hmac.new(hmac_key, message, hashlib.sha256)
    return base64.b64encode(signature).decode('utf-8')

async def connect_authenticated():
    uri = "wss://advanced-trade-ws.coinbase.com"

    timestamp = str(int(datetime.now().timestamp()))
    # ⚠️ Signature uses same pattern as REST: timestamp + "GET" + "/users/self/verify"
    signature = create_signature(timestamp, api_secret)

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "type": "subscribe",
            "channels": [{"name": "user"}],
            "signature": signature,
            "key": api_key,
            "passphrase": passphrase,
            "timestamp": timestamp
        }))
```

---

## Channels

### Public Channels (No Authentication)

#### 1. Ticker Channel

Real-time price updates and trade information. Batches consecutive matching events for efficiency.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channels": [
    {
      "name": "ticker",
      "product_ids": ["BTC-USD", "ETH-USD"]
    }
  ]
}
```

**Message Format:**
```json
{
  "type": "ticker",
  "sequence": 12345,
  "product_id": "BTC-USD",
  "price": "45123.45",
  "open_24h": "44500.00",
  "volume_24h": "12500.75",
  "low_24h": "44100.00",
  "high_24h": "45500.00",
  "volume_30d": "385000.00",
  "best_bid": "45123.00",
  "best_bid_size": "0.5",
  "best_ask": "45124.00",
  "best_ask_size": "1.2",
  "side": "buy",
  "time": "2025-10-19T12:34:56.789Z",
  "trade_id": "987654321",
  "last_size": "0.1"
}
```

**Use Cases:**
- Display current price on UI
- Track bid/ask spread
- Monitor 24h volume and price range

---

#### 2. Level2 Channel

Order book depth updates. Provides snapshots and incremental updates of the full order book.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channels": [
    {
      "name": "level2",
      "product_ids": ["BTC-USD"]
    }
  ]
}
```

**Snapshot Message (Initial):**
```json
{
  "type": "snapshot",
  "product_id": "BTC-USD",
  "bids": [
    ["45123.00", "1.5"],
    ["45122.50", "2.3"],
    ["45122.00", "0.8"]
  ],
  "asks": [
    ["45124.00", "1.2"],
    ["45124.50", "0.9"],
    ["45125.00", "2.1"]
  ]
}
```

**Update Message (Incremental):**
```json
{
  "type": "l2update",
  "product_id": "BTC-USD",
  "time": "2025-10-19T12:34:57.123Z",
  "changes": [
    ["buy", "45123.00", "2.0"],
    ["sell", "45124.00", "0.0"]
  ]
}
```

**Message Fields:**
- `type`: "snapshot" (initial) or "l2update" (incremental)
- `changes`: [side, price, size] - size of "0.0" means remove level
- Sides: "buy" (bids) or "sell" (asks)

**Use Cases:**
- Maintain real-time order book
- Calculate spread and depth
- Detect significant order book movements

---

#### 3. Matches Channel

Trade execution data. Broadcasts every executed trade on the product.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channels": [
    {
      "name": "matches",
      "product_ids": ["BTC-USD", "ETH-USD"]
    }
  ]
}
```

**Message Format:**
```json
{
  "type": "match",
  "trade_id": "987654321",
  "sequence": 12345,
  "maker_order_id": "order-uuid-1",
  "taker_order_id": "order-uuid-2",
  "time": "2025-10-19T12:34:57.456Z",
  "product_id": "BTC-USD",
  "price": "45123.50",
  "size": "0.5",
  "side": "sell"
}
```

**Message Fields:**
- `trade_id`: Unique identifier for this trade
- `maker_order_id`: Order that was on book
- `taker_order_id`: Order that crossed the spread
- `side`: "buy" (taker bought) or "sell" (taker sold)

**Use Cases:**
- Track volume and execution patterns
- Feed algorithms that react to trades
- Monitor unusual trading activity

---

### Private Channels (Authentication Required)

#### 4. User Channel

Personal account updates including orders, fills, and position changes.

**Subscribe (requires authentication):**
```json
{
  "type": "subscribe",
  "channels": [{"name": "user"}],
  "signature": "<computed_signature>",
  "key": "<api_key>",
  "passphrase": "<passphrase>",
  "timestamp": "<timestamp>"
}
```

**Message Types:**

**Order Placed:**
```json
{
  "type": "done",
  "order_id": "order-uuid",
  "reason": "filled",
  "product_id": "BTC-USD",
  "price": "45123.50",
  "remaining_size": "0.0",
  "side": "buy"
}
```

**Fill Event:**
```json
{
  "type": "done",
  "reason": "filled",
  "trade_id": "987654321",
  "order_id": "order-uuid",
  "user_id": "user-uuid",
  "profile_id": "profile-uuid",
  "liquidity": "T",
  "price_str": "45123.50",
  "size_str": "0.5",
  "fee": "13.54",
  "created_at": "2025-10-19T12:34:57.789Z",
  "side": "buy",
  "settled": false
}
```

**Use Cases:**
- Real-time order status updates
- Fill notifications
- Position tracking (for perps)

---

## Rate Limits

**WebSocket Connection Rate Limits:**
- **750 requests per second per IP address** (applies to all WebSocket connections from the same IP combined)
- This is a shared limit across all connections from your IP
- Multiple concurrent connections count toward this same limit

**Connection Behavior:**
- Maximum subscriptions per connection: Up to 100 channels (⚠️ **needs official verification**)
- Heartbeat: Required every 30 seconds to maintain connection
- Automatic reconnection with exponential backoff on disconnect

**Rate Limit Response:**
When limit exceeded, WebSocket may:
1. Reject new subscriptions with error
2. Close connection with code 1008 "policy violation"
3. Throttle message delivery

---

## Subscription Management

### Subscribe to Channels

```json
{
  "type": "subscribe",
  "channels": [
    {
      "name": "ticker",
      "product_ids": ["BTC-USD", "ETH-USD"]
    },
    {
      "name": "level2",
      "product_ids": ["BTC-USD"]
    }
  ]
}
```

### Unsubscribe from Channels

```json
{
  "type": "unsubscribe",
  "channels": [
    {
      "name": "ticker",
      "product_ids": ["BTC-USD"]
    }
  ]
}
```

### Heartbeat

Server sends heartbeat; client must respond:

**Server (incoming):**
```json
{
  "type": "heartbeat",
  "sequence": 12345,
  "timestamp": "2025-10-19T12:34:58.000Z"
}
```

**Client response (not required, but good practice):**
```json
{
  "type": "heartbeat",
  "timestamp": "2025-10-19T12:34:58.001Z"
}
```

---

## Error Handling

**Subscription Error:**
```json
{
  "type": "error",
  "message": "rate limit exceeded",
  "reason": "too_many_subscriptions"
}
```

**Authentication Error:**
```json
{
  "type": "error",
  "message": "Invalid Signature",
  "reason": "authentication_failed"
}
```

**Connection Close Codes:**
- `1000`: Normal closure
- `1008`: Policy violation (rate limit, invalid subscription)
- `1011`: Server internal error

---

## Best Practices

### 1. Connection Management

```python
import asyncio
from asyncio import sleep

async def connect_with_reconnect(max_retries=5):
    retries = 0
    backoff = 1  # seconds

    while retries < max_retries:
        try:
            async with websockets.connect(uri) as ws:
                retries = 0  # Reset on successful connection
                backoff = 1
                await handle_messages(ws)
        except Exception as e:
            retries += 1
            await sleep(backoff)
            backoff = min(backoff * 2, 60)  # Cap at 60 seconds
            logger.warning(f"Reconnecting... (attempt {retries})")
```

### 2. Heartbeat Monitoring

```python
async def monitor_heartbeat(ws, timeout=35):
    """Ensure heartbeats arrive within 35 seconds"""
    last_heartbeat = time.time()

    while True:
        if time.time() - last_heartbeat > timeout:
            logger.error("Heartbeat timeout - connection stale")
            await ws.close()
            break
        await sleep(10)
```

### 3. Subscription Efficiency

- Subscribe to only needed channels (not all 100+ products)
- Batch multiple products in one subscription
- Unsubscribe from unused channels to free capacity

### 4. Order Book Maintenance

```python
async def process_level2_updates(message):
    """Process order book updates"""
    if message["type"] == "snapshot":
        # Replace entire book
        order_book = {
            "bids": {price: size for price, size in message["bids"]},
            "asks": {price: size for price, size in message["asks"]}
        }
    elif message["type"] == "l2update":
        # Apply incremental changes
        for side_char, price, size in message["changes"]:
            side = "bids" if side_char == "buy" else "asks"
            if size == "0":
                del order_book[side][price]
            else:
                order_book[side][price] = size
```

---

## Maintenance & Versioning

- **API Version**: v3
- **Last Updated**: 2025-10-19
- **Verification Schedule**: Quarterly
- **WebSocket Status**: Check https://status.coinbase.com/

---

## See Also

- [coinbase_api_endpoints.md](coinbase_api_endpoints.md) - REST API endpoint catalog
- [coinbase_auth_guide.md](coinbase_auth_guide.md) - Authentication methods
- [coinbase_quick_reference.md](coinbase_quick_reference.md) - Quick reference card
- [coinbase_complete.md](coinbase_complete.md) - Complete integration guide
