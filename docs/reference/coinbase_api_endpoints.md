# Coinbase Advanced Trade API - Endpoint Catalog

---
status: current
created: 2025-10-19
last-verified: 2025-10-19
verification-schedule: quarterly
scope: Advanced Trade API v3 REST endpoints
documentation-venue: docs.cdp.coinbase.com/advanced-trade/docs/api-overview
---

> **Maintenance Note**: Sandbox behavior and endpoint availability changes are verified quarterly. Last verified: 2025-10-19. If using this doc after this date, consult the official API changelog at https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog

## Overview

This document catalogs all REST endpoints in the Coinbase Advanced Trade API v3. The API is organized by resource categories: Accounts, Orders, Products, Fees, Portfolios, and Payments.

**Rate Limits:**
- **Private endpoints** (user-specific): 30 requests/second per account
- **Public endpoints** (market data): 10 requests/second per IP address
- **Rate limit headers**: Response includes `CB-RATELIMIT-LIMIT`, `CB-RATELIMIT-REMAINING`, `CB-RATELIMIT-RESET`

**Base URL (Production):**
```
https://api.coinbase.com/api/v3/brokerage
```

**Base URL (Sandbox - CDP Sandbox):**
```
https://api-public.sandbox.exchange.coinbase.com/api/v3/brokerage
```
⚠️ **Note**: Sandbox only supports Accounts and Orders endpoints. Other endpoints return 404 or are unavailable. Responses are static and pre-defined, not live trading.

---

## Accounts Endpoints

Access account information, balances, and portfolio details.

| Endpoint | Method | Rate Limit | Production | Sandbox | Notes |
|----------|--------|-----------|-----------|---------|-------|
| List Accounts | `GET /accounts` | Private (30/s) | ✅ | ✅ static | Get account summaries (verified 2025-10-19) |
| Get Account | `GET /accounts/{account_id}` | Private (30/s) | ✅ | ✅ static | Get specific account details |
| Get Account Summary | `GET /accounts/summary` | Private (30/s) | ✅ | ✅ static | Overview of all account balances |

**Example Request:**
```bash
curl -X GET https://api.coinbase.com/api/v3/brokerage/accounts \
  -H "Authorization: Bearer <token>"
```

**Example Response (truncated):**
```json
{
  "accounts": [
    {
      "uuid": "account-uuid",
      "name": "BTC Account",
      "currency": "BTC",
      "available_balance": {
        "value": "1.5",
        "currency": "BTC"
      }
    }
  ]
}
```

---

## Orders Endpoints

Create, manage, retrieve, and cancel orders.

| Endpoint | Method | Rate Limit | Production | Sandbox | Notes |
|----------|--------|-----------|-----------|---------|-------|
| Create Order | `POST /orders` | Private (30/s) | ✅ | ✅ static | Place new order (verified 2025-10-19) |
| Cancel Order | `DELETE /orders/{order_id}` | Private (30/s) | ✅ | ✅ static | Cancel open order |
| List Orders | `GET /orders/batch` | Private (30/s) | ✅ | ✅ static | Get orders with filtering |
| Get Order | `GET /orders/{order_id}` | Private (30/s) | ✅ | ✅ static | Retrieve specific order |
| List Fills | `GET /orders/historical/fills` | Private (30/s) | ✅ | ✅ static | Get fill/execution history |
| Preview Order | `POST /orders/preview` | Private (30/s) | ✅ | ❌ | Dry-run order validation (perps only) |

**Example Request (Create Market Order):**
```bash
curl -X POST https://api.coinbase.com/api/v3/brokerage/orders \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "BTC-USD",
    "side": "BUY",
    "order_configuration": {
      "market_market_ioc": {
        "quote_size": "100"
      }
    }
  }'
```

**Order Configuration Types:**
- `market_market_ioc` - Market order (immediate or cancel)
- `limit_limit_gtc` - Limit order (good till cancelled)
- `limit_limit_gtd` - Limit order with expiration date
- `limit_limit_fok` - Fill or kill (not supported by Coinbase)
- `stop_loss_stop_loss` - Stop market order
- `stop_limit_stop_limit` - Stop limit order

---

## Products Endpoints (Market Data - Unauthenticated)

Retrieve market data and product information **without authentication**.

⚠️ **CRITICAL**: These endpoints live under `/market/*` path, NOT `/products`
- Authenticated endpoints (which require API key) are at `/api/v3/brokerage/products` (returns 401 if unauthenticated)
- **Unauthenticated market data** is at `/api/v3/brokerage/market/products` (no auth required)

| Endpoint | Method | Rate Limit | Auth Required | Sandbox | Notes |
|----------|--------|-----------|--------------|---------|-------|
| List Products | `GET /market/products` | Public (10/s) | ❌ No | ❌ | Get all tradeable products (verified 2025-10-19) |
| Get Product | `GET /market/products/{product_id}` | Public (10/s) | ❌ No | ❌ | Get specific product details |
| Get Product Candles | `GET /market/products/{product_id}/candles` | Public (10/s) | ❌ No | ❌ | OHLCV data (Unix seconds + enum granularity); ⚠️ 350 bucket max per request |
| Get Product Ticker | `GET /market/products/{product_id}/ticker` | Public (10/s) | ❌ No | ❌ | Current price, best bid/ask, recent trades |
| Get Product Book | `GET /market/product_book?product_id={id}` | Public (10/s) | ❌ No | ❌ | Order book (bid/ask levels) |

**Correct Unauthenticated Request Examples:**
```bash
# ✅ List products (no auth)
curl -X GET "https://api.coinbase.com/api/v3/brokerage/market/products"

# ✅ Get product details (no auth)
curl -X GET "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD"

# ✅ Get OHLCV candles (Unix timestamps, enum granularity)
# ⚠️ CRITICAL: Candles are capped at 350 buckets per request
#    - ONE_MINUTE: max 350 minutes (~5.83 hours)
#    - FIVE_MINUTE: max 1750 minutes (~29.17 hours)
#    - ONE_HOUR: max 350 hours (~14.58 days)
# This example: 3-hour range with ONE_MINUTE granularity = 180 buckets ✅
curl -X GET "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles?start=1697750400&end=1697761200&granularity=ONE_MINUTE"

# ✅ Get ticker (includes bid/ask + recent trades)
curl -X GET "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/ticker"

# ✅ Get order book
curl -X GET "https://api.coinbase.com/api/v3/brokerage/market/product_book?product_id=BTC-USD"

# ❌ WRONG - These return 401 Unauthorized (require API key)
curl -X GET "https://api.coinbase.com/api/v3/brokerage/products/BTC-USD"
```

**Candle Bucket Limit**: The API caps responses at **350 buckets per request**. This means:
- ONE_MINUTE candles: maximum 350 minutes (~5.83 hours per request)
- FIVE_MINUTE candles: maximum 1,750 minutes (~29.17 hours per request)
- ONE_HOUR candles: maximum 350 hours (~14.58 days per request)
- ONE_DAY candles: maximum 350 days (~11.7 months per request)

Requesting a larger time range returns HTTP 400 with `INVALID_ARGUMENT` error. For longer timeframes, either paginate with multiple requests or use a larger granularity.

**Sandbox Limitation**: Market data endpoints are not available in sandbox. Use production credentials for market data or mock broker for testing.

---

## Fees Endpoints

Retrieve fee schedules and transaction fee information.

| Endpoint | Method | Rate Limit | Production | Sandbox | Notes |
|----------|--------|-----------|-----------|---------|-------|
| Get Transaction Summary | `GET /transaction_summary` | Private (30/s) | ✅ | ❌ | 30-day volume and fees (verified 2025-10-19) |

**Example Request:**
```bash
curl -X GET https://api.coinbase.com/api/v3/brokerage/transaction_summary \
  -H "Authorization: Bearer <token>"
```

**Example Response:**
```json
{
  "trailing_volume": 1500000.00,
  "total_volume": 1500000.00,
  "volume_30_day": 1500000.00,
  "fee_tier_name": "Advanced",
  "maker_fee_rate": "0.0004",
  "taker_fee_rate": "0.0006",
  "advanced_trade_only_volume": 0.00,
  "advanced_trade_only_maker_fee_rate": "0.0002"
}
```

---

## Portfolios Endpoints

Manage portfolios (spot trading accounts).

| Endpoint | Method | Rate Limit | Production | Sandbox | Notes |
|----------|--------|-----------|-----------|---------|-------|
| List Portfolios | `GET /portfolios` | Private (30/s) | ✅ | ❌ | List all portfolios (verified 2025-10-19) |
| Create Portfolio | `POST /portfolios` | Private (30/s) | ✅ | ❌ | Create new portfolio |
| Move Portfolio Funds | `POST /portfolios/{portfolio_uuid}/moves` | Private (30/s) | ✅ | ❌ | Transfer funds between portfolios |
| Delete Portfolio | `DELETE /portfolios/{portfolio_uuid}` | Private (30/s) | ✅ | ❌ | Remove empty portfolio |

**Example Request:**
```bash
curl -X GET https://api.coinbase.com/api/v3/brokerage/portfolios \
  -H "Authorization: Bearer <token>"
```

---

## Perpetuals-Specific Endpoints

These endpoints are available only for INTX (derivatives) accounts with `COINBASE_ENABLE_DERIVATIVES=1`.

| Endpoint | Method | Rate Limit | Production | Sandbox | Notes |
|----------|--------|-----------|-----------|---------|-------|
| Preview Perpetuals Order | `POST /orders/preview` | Private (30/s) | ✅ INTX | ❌ | Preflight order validation (perps only) |
| Get Position | `GET /positions/{product_id}` | Private (30/s) | ✅ INTX | ❌ | Current perpetuals position |
| Close Position | `DELETE /positions/{product_id}` | Private (30/s) | ✅ INTX | ❌ | Close open perpetuals position |

**Availability**: Production with INTX approval only. Sandbox does not support perpetual futures.

---

## Authentication & Headers

**All private endpoints require:**
- `Authorization: Bearer <JWT_token>` (for CDP/JWT auth)
- OR standard HMAC signature headers (see [coinbase_auth_guide.md](coinbase_auth_guide.md))

**Response headers include:**
- `CB-RATELIMIT-LIMIT` - Requests allowed in this window
- `CB-RATELIMIT-REMAINING` - Requests remaining in window
- `CB-RATELIMIT-RESET` - Unix timestamp when window resets
- `CB-BEFORE`, `CB-AFTER` - Pagination cursors (for list endpoints)

---

## Common Error Responses

| Status | Error | Meaning | Action |
|--------|-------|---------|--------|
| 400 | `INVALID_REQUEST` | Malformed request | Check endpoint path and body format |
| 401 | `AUTHENTICATION_ERROR` | Missing or invalid auth | Verify API credentials |
| 403 | `FORBIDDEN` | Permission denied | Check account permissions and IP whitelist |
| 404 | `NOT_FOUND` | Resource missing | Verify product_id or account_id exists |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Reduce request rate, retry after `retry-after` header |
| 500 | `INTERNAL_ERROR` | Server error | Retry with exponential backoff |

---

## Pagination

List endpoints support cursor-based pagination via response headers:

```bash
# First request
curl -X GET "https://api.coinbase.com/api/v3/brokerage/orders/batch?limit=10" \
  -H "Authorization: Bearer <token>"

# Check response headers for CB-AFTER cursor
# Next request uses cursor parameter
curl -X GET "https://api.coinbase.com/api/v3/brokerage/orders/batch?limit=10&cursor=<CB-AFTER>" \
  -H "Authorization: Bearer <token>"
```

---

## Maintenance & Versioning

- **API Version**: v3 (current production)
- **Last Updated**: 2025-10-19
- **Verification Schedule**: Quarterly
- **Changes**: Check [Changelog](https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog)

---

## See Also

- [coinbase_websocket_reference.md](coinbase_websocket_reference.md) - Real-time WebSocket channels
- [coinbase_auth_guide.md](coinbase_auth_guide.md) - Authentication methods and recipes
- [coinbase_quick_reference.md](coinbase_quick_reference.md) - Quick reference card
- [coinbase_complete.md](coinbase_complete.md) - Complete integration guide
