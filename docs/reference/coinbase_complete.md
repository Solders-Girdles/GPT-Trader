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
  - src/gpt_trader/features/brokerages/coinbase/README.md
  - src/gpt_trader/features/brokerages/coinbase/COMPATIBILITY.md
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

### SimpleAuth (JWT) - Spot Trading

**Used for**: Default spot trading

```python
# JWT-based authentication for spot trading
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth

auth = SimpleAuth(key_name=api_key, private_key=api_secret)
```

> **Note**: `HMACAuth` has been removed. All authentication now uses JWT-based methods. See [coinbase_auth_guide.md](coinbase_auth_guide.md) for historical HMAC reference.

**Important**: Sandbox does NOT support perpetuals. Use production with canary profile for safe perpetuals testing.

### OAuth2 Authentication (Sign in with Coinbase)

**Status**: ✅ Supported (as of 2024/2025)

**Used for**: Building applications with user consent flow and delegated access

**Key characteristics**:
- Refresh tokens expire after 1.5 years
- OAuth connections enforce portfolio account-level trade access
- Requires client credentials in token revocation

**Status**: OAuth2 is officially supported (confirmed in official changelog and docs). Exact launch date is not publicly documented. See changelog at https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog

> For end-to-end authentication walkthroughs (JWT generation, NULL-byte-safe HMAC signatures, OAuth2 token refresh flows), use the dedicated [coinbase_auth_guide.md](coinbase_auth_guide.md).

## API Endpoints & Products

This guide covers how GPT-Trader consumes Coinbase API surfaces:

- **Spot runtime**: Uses Accounts, Orders, and unauthenticated `/market/*` product endpoints. Sandbox responses are static and limited to Accounts + Orders.
- **Perpetuals runtime**: All INTX-only endpoints remain dormant until `COINBASE_ENABLE_DERIVATIVES=1` and JWT auth is configured.
- **Coverage tracking**: Run `uv run pytest tests/unit/gpt_trader/features/brokerages/coinbase/ -v` to verify endpoint coverage.

## Order Types & Compatibility

GPT-Trader supports the following order configurations:

- Spot profiles expose market/limit order paths by default, with post-only exposed via adapter flags.
- Perpetuals profiles add stop-market and stop-limit variants once INTX is available.
- Reduce-only enforcement and leverage handling are implemented in the brokerage adapter and validated by the unit suite noted in the coverage matrix.

### Order Preview Gating (Production Safety)

Perpetuals orders in Advanced/PROD can be preflighted via Coinbase’s `preview_order` endpoint before submission.

- Enable with `ORDER_PREVIEW_ENABLED=1` or rely on the default auto-enable when:
  - `COINBASE_API_MODE=advanced`, `COINBASE_SANDBOX=0`, derivatives enabled, and `auth_type=JWT`.
- On preview failures, the adapter raises a validation error and emits a structured JSON log with `event=order_reject` and `reason=preview_failed`.
- In DEV/paper or Exchange mode, preview gating is disabled by default.

## WebSocket Integration

Channel payloads, authentication recipes, and rate limits are documented in detail in [coinbase_websocket_reference.md](coinbase_websocket_reference.md). Use that reference for schemas and subscription examples; this section captures GPT-Trader-operational notes only.

### Streaming Toggles (CANARY/PROD)

Environment toggles to safely enable streaming in production:

```bash
# Enable background streaming in TradingBot (CANARY/PROD only)
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
uv run python scripts/production_preflight.py --profile canary

# Smoke test the trading loop (mock broker)
uv run coinbase-trader run --profile dev --dev-fast

# Inspect streaming telemetry via TUI
uv run coinbase-trader run --profile dev --tui

# Export Prometheus-compatible metrics
uv run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json
```

## Testing & Development

### Mock Broker

For development without API calls:

```bash
PERPS_FORCE_MOCK=1 uv run coinbase-trader run --profile dev
```

Features:
- Realistic market data simulation
- Deterministic order fills
- No API rate limits
- Instant execution

### Sandbox Testing

For integration testing with real API:

```bash
COINBASE_SANDBOX=1 uv run coinbase-trader run --profile dev
```

**Limitations**:
- Spot trading only (no perpetuals)
- Different authentication (HMAC)
- Limited market data

### Production Testing

For safe production testing:

```bash
uv run coinbase-trader run --profile canary --dry-run
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

For GPT-Trader operations:

- The adapters cap outbound REST traffic at ~100 req/min, well below Coinbase ceilings.
- WebSocket clients share a single connection per profile and monitor 30-second heartbeat cadence.
- On any 429, rely on the retry-after header and log the incident in `var/logs/coinbase_trader.log` for follow-up.

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
uv run coinbase-trader run --profile prod

# Check system health
uv run python scripts/production_preflight.py --profile canary

# Account snapshot (balances, permissions, fee schedule)
uv run coinbase-trader account snapshot

# Emergency stop
export RISK_KILL_SWITCH_ENABLED=1 && pkill -f coinbase-trader
```

### Key Files

- Main entry: `src/gpt_trader/cli/__init__.py`
- Coinbase client: `src/gpt_trader/features/brokerages/coinbase/client.py`
- WebSocket handler: `src/gpt_trader/features/brokerages/coinbase/ws.py`
- Configuration: `config/environments/.env.template`

---

## System Requirements

### Python & Dependencies
- **Python**: 3.12+ required
- **Poetry**: 1.0+ for dependency management
- **websockets**: >=13.0,<14.0 (Coinbase SDK compatibility)

### Network Requirements
- Outbound HTTPS (port 443) to api.coinbase.com
- WebSocket connections to advanced-trade-ws.coinbase.com
- No inbound connections required

### Hardware (Production)
- CPU: 4+ cores recommended
- RAM: 8+ GB
- Network: <50ms latency to Coinbase

## External Documentation Links

**Coinbase Developer Platform (CDP):**
- [Advanced Trade Welcome](https://docs.cdp.coinbase.com/advanced-trade/docs/welcome)
- [REST API Rate Limits](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-rate-limits)
- [WebSocket Channels](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels)
- [API Authentication](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth)
- [Python SDK](https://github.com/coinbase/coinbase-advanced-py)

**Note**: Legacy docs at `docs.cloud.coinbase.com` are outdated.

---

*This consolidated reference replaces all previous scattered Coinbase documentation.*
