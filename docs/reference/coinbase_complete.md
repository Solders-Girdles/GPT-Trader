# Coinbase Integration - Complete Reference

---
status: current
created: 2025-01-01
consolidates:
  - docs/reference/coinbase.md
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

## API Endpoints & Products

### Supported Products

- **Spot (default)**: BTC-USD, ETH-USD, and other Advanced Trade spot pairs
- **Derivatives (INTX required)**: BTC-PERP, ETH-PERP, SOL-PERP, XRP-PERP (dormant until enabled)
- **Sandbox**: Spot-only environment (no derivatives)

### API Endpoints

```python
# Production (Advanced Trade v3)
BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
WS_URL = "wss://advanced-trade-ws.coinbase.com"
# Sandbox (spot only)
BASE_URL = "https://api-sandbox.coinbase.com/api/v2"
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
- Maximum 100 channel subscriptions

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
poetry run perps-bot --profile dev --dev-fast

# Inspect streaming telemetry
poetry run python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 5

# Export Prometheus-compatible metrics
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json
```

## Testing & Development

### Mock Broker

For development without API calls:

```bash
PERPS_FORCE_MOCK=1 poetry run perps-bot --profile dev
```

Features:
- Realistic market data simulation
- Deterministic order fills
- No API rate limits
- Instant execution

### Sandbox Testing

For integration testing with real API:

```bash
COINBASE_SANDBOX=1 poetry run perps-bot --profile dev
```

**Limitations**:
- Spot trading only (no perpetuals)
- Different authentication (HMAC)
- Limited market data

### Production Testing

For safe production testing:

```bash
poetry run perps-bot --profile canary --dry-run
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

### Rate Limiting
- Production: 10 requests/second
- Sandbox: 5 requests/second
- WebSocket: 100 subscriptions maximum

## Support Resources

### Official Documentation
- [Coinbase Advanced Trade API](https://docs.cdp.coinbase.com/advanced-trade/docs/welcome)
- [WebSocket Feed](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview)
- [API Status](https://status.coinbase.com/)

### Internal Resources
- Logs: `var/logs/perps_bot.log`
- EventStore metrics: `var/data/perps_bot/<profile>/metrics.json`
- Configuration templates: `config/environments/`

## Quick Reference

### Essential Commands

```bash
# Run spot bot in production profile
poetry run perps-bot --profile prod

# Check system health
poetry run python scripts/production_preflight.py --profile canary

# Account snapshot (balances, permissions, fee schedule)
poetry run perps-bot --account-snapshot

# Emergency stop
export RISK_KILL_SWITCH_ENABLED=1 && pkill -f perps-bot
```

### Key Files

- Main entry: `src/bot_v2/cli.py`
- Coinbase client: `src/bot_v2/features/brokerages/coinbase/client.py`
- WebSocket handler: `src/bot_v2/features/brokerages/coinbase/ws.py`
- Configuration: `config/environments/.env.template`

---

*This consolidated reference replaces all previous scattered Coinbase documentation. For historical documentation, see `/docs/ARCHIVE/2024_implementation/coinbase/`*
