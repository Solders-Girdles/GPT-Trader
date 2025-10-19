# Coinbase Integration - Complete Reference

---
status: current
created: 2025-01-01
last-verified: 2025-10-19
verification-schedule: quarterly
scope: Coinbase Advanced Trade API v3 (spot + future-ready perps)
documentation-venues:
  - primary: docs.cdp.coinbase.com (current Coinbase Developer Platform)
  - legacy: docs.cloud.coinbase.com (older Coinbase Cloud - for reference only)
consolidates:
  - docs/reference/coinbase.md (retired)
  - docs/reference/coinbase_troubleshooting.md
  - src/bot_v2/features/brokerages/coinbase/README.md
  - src/bot_v2/features/brokerages/coinbase/COMPATIBILITY.md
  - 10+ archived Coinbase documentation files
---

## Overview

This is the complete, consolidated reference for Coinbase integration in GPT-Trader. The bot runs **spot trading by default** using Coinbase Advanced Trade, and keeps perpetual futures support in a **future-ready** state that activates only when Coinbase grants INTX access and `COINBASE_ENABLE_DERIVATIVES=1` is set.

## Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [Authentication Methods](#authentication-methods)
3. [API Endpoints & Products](#api-endpoints--products)
4. [Order Types & Compatibility](#order-types--compatibility)
5. [WebSocket Integration](#websocket-integration)
6. [Troubleshooting](#troubleshooting)
7. [Testing & Development](#testing--development)

## Environment Configuration

### Required Environment Variables

```bash
# Spot trading (default) - HMAC Advanced Trade
COINBASE_API_KEY=your_hmac_api_key
COINBASE_API_SECRET=your_hmac_api_secret
COINBASE_ENABLE_DERIVATIVES=0

# Sandbox (spot only)
COINBASE_SANDBOX=1
COINBASE_API_KEY=your_sandbox_key
COINBASE_API_SECRET=your_sandbox_secret
COINBASE_PASSPHRASE=your_sandbox_passphrase

# Derivatives (INTX accounts only)
# COINBASE_ENABLE_DERIVATIVES=1
# COINBASE_PROD_CDP_API_KEY=organizations/{org_id}/apiKeys/{key_id}
# COINBASE_PROD_CDP_PRIVATE_KEY="""-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"""
```

### Environment Setup

1. **Copy template**:
```bash
cp config/environments/.env.template .env
```

2. **Configure for your environment**:
   - **Spot Production**: Use HMAC authentication (default)
   - **Perps Production**: Enable only after INTX approval (set derivatives flag + CDP keys)
   - **Sandbox**: Use HMAC; sandbox does not support derivatives
   - **Development**: Use mock broker with `PERPS_FORCE_MOCK=1`

### Trading Profiles

| Profile | Environment | Products | Risk Level |
|---------|------------|----------|------------|
| **dev** | Mock broker | All | None (simulated) |
| **canary** | Production | Spot (perps when INTX enabled) | Ultra-conservative |
| **prod** | Production | Spot (perps when INTX enabled) | Normal limits |

## Authentication Methods

### CDP (JWT) Authentication - Production (INTX Accounts)

**Used for**: Perpetual futures trading once Coinbase approves INTX access

```python
# Automatic detection from key format
if api_key.startswith("organizations/"):
    # CDP JWT authentication
    auth = CDPAuth(api_key, api_secret)
```

**Key format**:
- API Key: `organizations/{org_id}/apiKeys/{key_id}`
- API Secret: Full EC private key including headers/footers

### HMAC Authentication - Spot & Sandbox

**Used for**: Default spot trading and sandbox testing (sandbox has no perps)

```python
# Legacy authentication for sandbox
if os.getenv("COINBASE_SANDBOX"):
    auth = HMACAuth(api_key, api_secret, passphrase)
```

**Important**: Sandbox does NOT support perpetuals. Use production with canary profile for safe perpetuals testing.

### OAuth2 Authentication (Sign in with Coinbase)

**Status**: ✅ Supported (as of 2024/2025)

**Used for**: Building applications with user consent flow and delegated access

**Key characteristics**:
- Refresh tokens expire after 1.5 years
- OAuth connections enforce portfolio account-level trade access
- Requires client credentials in token revocation

**Status**: OAuth2 is officially supported (confirmed in official changelog and docs). Exact launch date is not publicly documented. See changelog at https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog

## API Endpoints & Products

### Supported Products

- **Spot (default)**: BTC-USD, ETH-USD, and other Advanced Trade spot pairs
- **Derivatives (INTX required)**: BTC-PERP, ETH-PERP, SOL-PERP, XRP-PERP (dormant until enabled)
- **Sandbox**: Spot-only environment (no derivatives, static responses)
  - Note: Sandbox Accounts and Orders endpoints only; responses are pre-defined and not live

### API Endpoints

```python
# Production (Advanced Trade v3)
BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
WS_URL = "wss://advanced-trade-ws.coinbase.com"
# Sandbox (Advanced Trade v3 - CDP Sandbox)
BASE_URL = "https://api-public.sandbox.exchange.coinbase.com/api/v3/brokerage"
WS_URL = "wss://ws-feed-sandbox.exchange.coinbase.com"
```

## Order Types & Compatibility

### Fully Supported Order Types

| Order Type | API Support | Implementation | Notes |
|------------|-------------|----------------|-------|
| Market | ✅ Full | ✅ Complete | Immediate execution |
| Limit | ✅ Full | ✅ Complete | Post-only option available |
| Stop Loss | ✅ Full | ✅ Complete | Stop market orders |
| Stop Limit | ✅ Full | ✅ Complete | Stop with limit price |

### Time-In-Force (TIF) Support

| TIF | Support | Notes |
|-----|---------|-------|
| GTC (Good Till Cancelled) | ✅ | Default for most orders |
| IOC (Immediate or Cancel) | ✅ | For immediate fills |
| GTD (Good Till Date) | ✅ | With expiry time |
| FOK (Fill or Kill) | ❌ | Not supported by Coinbase |

### Advanced Features

- **Post-Only**: Maker-only orders for limit orders
- **Reduce-Only**: Can only reduce position size
- **Client Order ID**: Idempotency support
- **Leverage**: Available for perpetual contracts

### Order Preview Gating (Production Safety)

Perpetuals orders in Advanced/PROD can be preflighted via Coinbase’s `preview_order` endpoint before submission.

- Enable with `ORDER_PREVIEW_ENABLED=1` or rely on the default auto-enable when:
  - `COINBASE_API_MODE=advanced`, `COINBASE_SANDBOX=0`, derivatives enabled, and `auth_type=JWT`.
- On preview failures, the adapter raises a validation error and emits a structured JSON log with `event=order_reject` and `reason=preview_failed`.
- In DEV/paper or Exchange mode, preview gating is disabled by default.

## WebSocket Integration

### Connection Setup

```python
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket

ws = CoinbaseWebSocket(
    api_key=os.getenv("COINBASE_API_KEY"),
    api_secret=os.getenv("COINBASE_API_SECRET"),
    sandbox=bool(os.getenv("COINBASE_SANDBOX"))
)

# Subscribe to channels
await ws.subscribe(["ticker", "matches", "level2"], ["BTC-PERP"])
```

### Supported Channels

| Channel | Purpose | Implementation |
|---------|---------|----------------|
| ticker | Price updates | ✅ Normalized |
| matches | Trade data | ✅ Volume tracking |
| level2 | Order book | ✅ Depth calculation |

### Heartbeat & Reconnection

- Automatic heartbeat every 30 seconds
- Exponential backoff reconnection
- **Channel subscriptions**: Up to 100 per connection (⚠️ not explicitly documented; may be tier-specific)

### Rate Limiting

**WebSocket API**: 750 requests per second **per IP address** (applies to all connections from the same IP combined)

**Engineering note**: Current codebase uses 100 requests/min throttle (much more conservative than API limits)

### Streaming Toggles (CANARY/PROD)

Environment toggles to safely enable streaming in production:

```bash
# Enable background streaming in PerpsBot (CANARY/PROD only)
PERPS_ENABLE_STREAMING=1

# Attach JWT for WS user channel (when using CDP/JWT auth)
COINBASE_WS_USER_AUTH=1

# Select order book level for streaming (1=ticker, 2=level2)
PERPS_STREAM_LEVEL=1
```

Notes:
- REST `get_quote()` remains the fallback; WS reconnect/backoff is handled automatically.
- When enabled, the bot updates mark prices and staleness timestamps at sub-second cadence (in DEV, this is mocked).

## Troubleshooting

### Common Issues & Solutions

#### Authentication Errors

**Problem**: "Invalid API Key" or "Unauthorized"
**Solutions**:
1. Verify key format matches authentication type
2. For CDP: Ensure full private key with headers/footers
3. Check environment variables are loaded correctly
4. Verify sandbox vs production environment

#### WebSocket Disconnections

**Problem**: Frequent disconnects or no data
**Solutions**:
1. Check heartbeat implementation
2. Verify subscription limits (100 channels max)
3. Monitor network stability
4. Enable debug logging: `--log-level DEBUG`

#### Order Rejections

**Problem**: Orders fail to place
**Solutions**:
1. Check product size increments (e.g., 0.001 for BTC-PERP)
2. Verify margin requirements
3. Ensure proper order type support
4. Check reduce-only constraints for closing positions

#### "Post-only would cross"

**Problem**: Limit order rejected
**Solutions**:
1. Adjust limit price further from spread
2. Check current bid/ask prices
3. Disable post-only for aggressive orders

### Debug Workflow

```bash
# Validate environment + credentials
poetry run python scripts/production_preflight.py --profile canary

# Smoke test the trading loop (mock broker)
poetry run coinbase-trader run --profile dev --dev-fast

# Inspect streaming telemetry
poetry run python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 5

# Export Prometheus-compatible metrics
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json
```

## Testing & Development

### Mock Broker

For development without API calls:

```bash
PERPS_FORCE_MOCK=1 poetry run coinbase-trader run --profile dev
```

Features:
- Realistic market data simulation
- Deterministic order fills
- No API rate limits
- Instant execution

### Sandbox Testing

For integration testing with real API:

```bash
COINBASE_SANDBOX=1 poetry run coinbase-trader run --profile dev
```

**Limitations**:
- Spot trading only (no perpetuals)
- Different authentication (HMAC)
- Limited market data

### Production Testing

For safe production testing:

```bash
poetry run coinbase-trader run --profile canary --dry-run
```

Features:
- Real market data
- Tiny position sizes
- Extra safety checks
- Comprehensive logging

## Migration Notes

### From V2 to V3 API

The system currently uses V2 (Advanced Trade) which is stable and recommended. V3 migration considerations:

1. **Authentication**: V3 uses JWT exclusively
2. **Endpoints**: Different URL structure
3. **Order Schema**: New `order_configuration` format
4. **WebSocket**: Different subscription format

### From Equities to Perpetuals

If migrating from older equities-based system:

1. Replace all equity symbols with perpetual contracts
2. Update position sizing for contract specifications
3. Implement funding rate calculations
4. Adjust risk management for leverage

## Performance Optimization

### Connection Management
- Use connection pooling for HTTP requests
- Maintain single WebSocket connection per session
- Implement request batching where possible

### Data Processing
- Cache frequently accessed data
- Use incremental order book updates
- Implement efficient depth calculation

### API Rate Limits

**REST API (Official Coinbase Advanced Trade API):**
- **Private endpoints**: 30 requests/second (per user account)
- **Public endpoints**: 10 requests/second (per IP address)
- **Rate limit algorithm**: Token bucket (lazy-fill, starts full, refills continuously)

**WebSocket API:**
- **750 requests/second per IP address** (applies to all WebSocket connections from same IP combined)

**Rate Limit Response Headers** (available in HTTP responses):
- `CB-RATELIMIT-LIMIT` - Total request limit for current window
- `CB-RATELIMIT-REMAINING` - Requests remaining in current window
- `CB-RATELIMIT-RESET` - Unix timestamp when limit window resets
- `CB-BEFORE`, `CB-AFTER` - Pagination cursors (for list endpoints)

**HTTP 429 Response**: When rate limit exceeded, API responds with status 429 "Too Many Requests" and `retry-after` header (in seconds)

**Engineering implementation note**:
- Current codebase uses conservative 100 requests/min client-side throttle (well below API limits)
- See `src/bot_v2/features/brokerages/coinbase/client/base.py` for implementation

## Support Resources

### Official Documentation

**Primary (Current - Coinbase Developer Platform - CDP):**
- [Advanced Trade API Welcome](https://docs.cdp.coinbase.com/advanced-trade/docs/welcome)
- [REST API Rate Limits](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-rate-limits)
- [WebSocket Rate Limits](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-rate-limits)
- [WebSocket Channels & Overview](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview)
- [API Authentication](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth)
- [API Endpoints](https://docs.cdp.coinbase.com/advanced-trade/docs/api-overview)
- [Changelog](https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog)
- [Python SDK](https://github.com/coinbase/coinbase-advanced-py)

**Status & Support:**
- [API Status](https://status.coinbase.com/)
- [Coinbase Developer Platform](https://www.coinbase.com/developer-platform)

**Note**: Legacy docs at `docs.cloud.coinbase.com` are outdated; use CDP URLs above for current information.

### Internal Resources
- Logs: `var/logs/coinbase_trader.log`
- EventStore metrics: `var/data/coinbase_trader/<profile>/metrics.json`
- Configuration templates: `config/environments/`

## Quick Reference

### Essential Commands

```bash
# Run spot bot in production profile
poetry run coinbase-trader run --profile prod

# Check system health
poetry run python scripts/production_preflight.py --profile canary

# Account snapshot (balances, permissions, fee schedule)
poetry run coinbase-trader account snapshot

# Emergency stop
export RISK_KILL_SWITCH_ENABLED=1 && pkill -f coinbase-trader
```

### Key Files

- Main entry: `src/bot_v2/cli/__init__.py`
- Coinbase client: `src/bot_v2/features/brokerages/coinbase/client.py`
- WebSocket handler: `src/bot_v2/features/brokerages/coinbase/ws.py`
- Configuration: `config/environments/.env.template`

---

*This consolidated reference replaces all previous scattered Coinbase documentation. For historical documentation, see `/docs/ARCHIVE/2024_implementation/coinbase/`*
