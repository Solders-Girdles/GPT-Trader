# Coinbase Advanced Trade API - Quick Reference Card

---
status: current
created: 2025-10-19
last-verified: 2025-10-19
scope: Quick lookup for common API operations
documentation-venue: docs.cdp.coinbase.com/advanced-trade
---

## API Endpoints

### Base URLs
```
Production REST: https://api.coinbase.com/api/v3/brokerage
Sandbox REST:    https://api-public.sandbox.exchange.coinbase.com/api/v3/brokerage
Production WS:   wss://advanced-trade-ws.coinbase.com
Sandbox WS:      wss://ws-feed-sandbox.exchange.coinbase.com
```

**⚠️ Sandbox Limitations**: Only Accounts and Orders endpoints are available. Product, Portfolio, and Fees endpoints return 404. Responses are static/pre-defined, not live.

---

## Most-Used REST Endpoints

### Accounts
```bash
# List all accounts
GET /accounts

# Get specific account
GET /accounts/{account_id}

# Get account summary (all balances)
GET /accounts/summary
```

### Orders
```bash
# Create order
POST /orders
{
  "product_id": "BTC-USD",
  "side": "BUY",
  "order_configuration": {
    "market_market_ioc": {"quote_size": "100"}
  }
}

# Cancel order
DELETE /orders/{order_id}

# List orders with filters
GET /orders/batch?order_status=PENDING&product_id=BTC-USD

# Get fills/executions
GET /orders/historical/fills
```

### Products (Public Market Data - No Auth Required)

⚠️ **CRITICAL**: Use `/market/*` paths for unauthenticated access
- `/market/products` returns 200 (no auth)
- `/products` returns 401 (requires API key)

```bash
# ✅ List all products (no auth)
GET /market/products

# ✅ Get product details (no auth)
GET /market/products/BTC-USD

# ✅ Get price candles/OHLCV (no auth)
GET /market/candles?product_id=BTC-USD&start=<timestamp>&end=<timestamp>&granularity=300

# ✅ Get current ticker (no auth)
GET /market/ticker?product_id=BTC-USD

# ✅ Get order book (no auth)
GET /market/orderbook?product_id=BTC-USD

# ❌ These return 401 (do NOT use):
GET /products  # Wrong - requires auth
GET /products/BTC-USD  # Wrong - requires auth
```

### Fees
```bash
# Get 30-day volume and fee tier
GET /transaction_summary
```

---

## Rate Limits at a Glance

| Type | Limit | Notes |
|------|-------|-------|
| **REST Private** | 30 req/sec | Per account |
| **REST Public** | 10 req/sec | Per IP |
| **WebSocket** | 750 req/sec | Per IP (all WS connections combined) |
| **HTTP 429 Response** | Rate limited | Check `retry-after` header |

**Rate Limit Headers** (in every response):
```
CB-RATELIMIT-LIMIT:     30 (requests allowed in window)
CB-RATELIMIT-REMAINING: 28 (requests left in window)
CB-RATELIMIT-RESET:     1697750400 (Unix timestamp when resets)
```

---

## Authentication Quick Lookup

| Auth Type | When to Use | Command |
|-----------|------------|---------|
| **CDP/JWT** | Perpetual futures (INTX) | `Authorization: Bearer <jwt_token>` |
| **HMAC** | Spot trading + sandbox | Headers: `CB-ACCESS-KEY`, `CB-ACCESS-SIGN`, `CB-ACCESS-TIMESTAMP`, `CB-ACCESS-PASSPHRASE` |
| **OAuth2** | Multi-user apps | `Authorization: Bearer <access_token>` |

### Quick cURL Examples

**HMAC (Spot) - CORRECT Signature Generation (NULL-byte safe):**
```bash
# Setup variables
TIMESTAMP=$(date +%s)
METHOD="GET"
PATH="/api/v3/brokerage/accounts"  # ⚠️ CRITICAL: Full path, not just "/accounts"
SECRET_B64="your_base64_secret"
API_KEY="your_api_key"
PASSPHRASE="your_passphrase"

# Convert base64 secret to hex WITHOUT using shell variables for binary data
# (shell variables truncate on NULL bytes, breaking signatures)
KEY_HEX=$(printf %s "$SECRET_B64" | base64 -d | xxd -p -c256 | tr -d '\n')

# Create the message to sign and compute HMAC-SHA256 using hex key
# Message format: timestamp + method + full_path (e.g., "1697750400GET/api/v3/brokerage/accounts")
MESSAGE="${TIMESTAMP}${METHOD}${PATH}"
SIG=$(echo -n "$MESSAGE" | openssl dgst -sha256 -mac HMAC -macopt hexkey:$KEY_HEX -binary | base64)

# Make the request
curl -X $METHOD \
  -H "CB-ACCESS-KEY: $API_KEY" \
  -H "CB-ACCESS-SIGN: $SIG" \
  -H "CB-ACCESS-TIMESTAMP: $TIMESTAMP" \
  -H "CB-ACCESS-PASSPHRASE: $PASSPHRASE" \
  https://api.coinbase.com/api/v3/brokerage/accounts
```

**⚠️ CRITICAL: NULL Byte Handling**:
- ❌ **BROKEN**: `SECRET_DECODED=$(echo ... | base64 -d); openssl dgst -hmac "$SECRET_DECODED"`
  - Shell variables TRUNCATE on NULL bytes (common in binary data)
  - Signatures are silently wrong → 401 Unauthorized
  - Example: 32-byte secret becomes 30 bytes, signature fails

- ✅ **CORRECT**: Convert to hex via xxd, use `macopt hexkey:`
  - Keeps binary data out of shell variables
  - Safely handles any byte values including NULLs
  - Guaranteed correct signatures

**⚠️ Other Common HMAC Mistakes**:
1. ❌ Using file descriptor `<(echo ... | base64 -d)` - passes literal `/dev/fd/...` string
2. ❌ Wrong message format (must be exactly: `timestamp+method+path+body` no spaces)
3. ❌ Milliseconds instead of seconds in timestamp

**JWT (Perps):**
```bash
# Use coinbase-cli or generate JWT yourself (see auth_guide.md)
# Then:
curl -X GET \
  -H "Authorization: Bearer <your_jwt_token>" \
  https://api.coinbase.com/api/v3/brokerage/accounts
```

---

## WebSocket Channels

### Subscribe to Ticker (Price Updates)
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

### Subscribe to Order Book (Level2)
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

### Subscribe to Trades (Matches)
```json
{
  "type": "subscribe",
  "channels": [
    {
      "name": "matches",
      "product_ids": ["BTC-USD"]
    }
  ]
}
```

### Subscribe to User Updates (Requires Auth)
```json
{
  "type": "subscribe",
  "channels": [{"name": "user"}],
  "signature": "<signature>",
  "key": "<api_key>",
  "passphrase": "<passphrase>",
  "timestamp": "<timestamp>"
}
```

---

## Order Types Quick Reference

### Market Order (Immediate Execution)
```bash
POST /orders
{
  "product_id": "BTC-USD",
  "side": "BUY",
  "order_configuration": {
    "market_market_ioc": {
      "quote_size": "100"  # in quote currency (USD)
    }
  }
}
```

### Limit Order (GTC)
```bash
POST /orders
{
  "product_id": "BTC-USD",
  "side": "BUY",
  "order_configuration": {
    "limit_limit_gtc": {
      "base_size": "0.5",      # in base currency (BTC)
      "limit_price": "45000",
      "post_only": false
    }
  }
}
```

### Stop-Market Order (Perps Only)
```bash
POST /orders
{
  "product_id": "BTC-PERP",
  "side": "SELL",
  "order_configuration": {
    "stop_loss_stop_loss": {
      "base_size": "0.01",
      "stop_trigger_price": "40000"
    }
  }
}
```

### Stop-Limit Order (Perps Only)
```bash
POST /orders
{
  "product_id": "BTC-PERP",
  "side": "SELL",
  "order_configuration": {
    "stop_limit_stop_limit": {
      "base_size": "0.01",
      "stop_trigger_price": "40000",
      "limit_price": "39999"
    }
  }
}
```

---

## Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | ✅ Success | Use response data |
| 400 | Bad request | Check request format/parameters |
| 401 | Unauthorized | Verify API key/signature |
| 403 | Forbidden | Check permissions/IP whitelist |
| 404 | Not found | Verify product_id or account_id |
| 429 | Rate limited | **Wait** `retry-after` seconds, retry |
| 500 | Server error | Retry with exponential backoff |

---

## Environment-Specific Features

### Production
- **Spot Trading**: ✅ Always available
- **Perpetuals**: ✅ With INTX approval + `COINBASE_ENABLE_DERIVATIVES=1`
- **Order Preview**: ✅ Available (perps only)
- **Market Data**: ✅ Real-time

### Sandbox
- **Spot Trading**: ✅ Only Accounts and Orders endpoints
- **Perpetuals**: ❌ Not available
- **Market Data**: ❌ Not available (static responses)
- **Use Case**: Integration testing only

---

## Troubleshooting Checklist

### Authentication Failing
- [ ] API key format matches auth type (CDP starts with `organizations/`, HMAC is hex)
- [ ] Signature/JWT calculated correctly
- [ ] Timestamp within 30 seconds of server time
- [ ] Private key has headers/footers (for CDP)
- [ ] API key not expired or revoked

### Orders Rejected
- [ ] Product supports order type (e.g., no stop-loss on spot)
- [ ] Size meets minimum increment (e.g., 0.001 BTC)
- [ ] Sufficient balance in account
- [ ] Post-only order not crossing spread
- [ ] Account has trading permissions

### WebSocket Disconnecting
- [ ] Sending heartbeat every 30 seconds
- [ ] Not exceeding 100 channel subscriptions per connection
- [ ] Not exceeding 750 req/sec across all connections from your IP
- [ ] Network stable and SSL certificates valid

### Rate Limited (429)
- [ ] Check `CB-RATELIMIT-REMAINING` header
- [ ] Wait `retry-after` seconds before retrying
- [ ] Reduce request rate below limits
- [ ] Batch operations where possible

---

## Useful Links

**Official Docs**: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome

**By Task:**
- Place/Cancel Orders: https://docs.cdp.coinbase.com/advanced-trade/docs/api-overview
- Get Fills: https://docs.cdp.coinbase.com/advanced-trade/docs/api-overview
- View Products: https://docs.cdp.coinbase.com/advanced-trade/docs/api-overview
- WebSocket: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview

**Rate Limits:**
- REST: https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-rate-limits
- WebSocket: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-rate-limits

**Python SDK**: https://github.com/coinbase/coinbase-advanced-py

**Status**: https://status.coinbase.com/

---

## See Also

- [coinbase_api_endpoints.md](coinbase_api_endpoints.md) - Full endpoint catalog
- [coinbase_websocket_reference.md](coinbase_websocket_reference.md) - WebSocket details
- [coinbase_auth_guide.md](coinbase_auth_guide.md) - Authentication recipes
- [coinbase_complete.md](coinbase_complete.md) - Complete integration guide
