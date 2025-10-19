# Coinbase API Links & Quick Reference

Purpose-built quicklinks for agents to jump straight to the right Coinbase docs.

> **Documentation Venues**:
> - **Primary (Current)**: `docs.cdp.coinbase.com` - Coinbase Developer Platform (up-to-date)
> - **Legacy**: `docs.cloud.coinbase.com` - Older Coinbase Cloud (for reference only)
>
> **Always use CDP URLs** for current information.

## Most-Used Tasks (with Rate Limits)

| Task | Endpoint | Rate Limit | Notes |
|------|----------|-----------|-------|
| Place/cancel orders | POST/DELETE /orders | 30 req/sec | See endpoint catalog |
| Get fills/executions | GET /orders/historical/fills | 30 req/sec | Execution history |
| List products | GET /market/products | 10 req/sec | Public market data endpoint, no auth required |
| Account/balances | GET /accounts | 30 req/sec | Private endpoint |
| WebSocket real-time | Subscribe to ticker/level2 | 750 req/sec per IP | See WebSocket reference |

**Quick Lookup Tips:**
- Use repo's `docs/reference/coinbase_quick_reference.md` for cURL examples
- Use `docs/reference/coinbase_api_endpoints.md` for full endpoint catalog
- Use `docs/reference/coinbase_auth_guide.md` for authentication recipes

## Advanced Trade API (Spot & Perps)

### Primary Documentation (CDP - Current)
- **Home**: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome
- **API Overview**: https://docs.cdp.coinbase.com/advanced-trade/docs/api-overview
- **Authentication**: https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth
- **REST Rate Limits**: https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-rate-limits
- **WebSocket Overview**: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview
- **WebSocket Channels**: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels
- **WebSocket Rate Limits**: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-rate-limits
- **Changelog**: https://docs.cdp.coinbase.com/coinbase-app/introduction/changelog

### API Base Paths
- **Production REST**: `https://api.coinbase.com/api/v3/brokerage`
- **Sandbox REST**: `https://api-public.sandbox.exchange.coinbase.com/api/v3/brokerage` (Accounts and Orders only)
- **Production WebSocket**: `wss://advanced-trade-ws.coinbase.com`
- **Sandbox WebSocket**: `wss://ws-feed-sandbox.exchange.coinbase.com`

### Authentication by Use Case
| Use Case | Auth Method | Status | Rate Limit |
|----------|------------|--------|-----------|
| **Spot Trading** | HMAC | ✅ Default | 30 req/sec |
| **Perpetual Futures** | CDP (JWT) | ✅ With INTX | 30 req/sec |
| **Sandbox Testing** | HMAC | ✅ Spot only | 30 req/sec |
| **Multi-user Apps** | OAuth2 | ✅ New | 30 req/sec |

### Rate Limit Headers (Available in All Responses)
```
CB-RATELIMIT-LIMIT:     30 (max requests in current window)
CB-RATELIMIT-REMAINING: 28 (requests left in window)
CB-RATELIMIT-RESET:     1697750400 (Unix timestamp - when window resets)
CB-BEFORE, CB-AFTER:    Pagination cursors (for list endpoints)
retry-after:            Seconds to wait (on 429 response)
```

### Key Points
- **Spot trading (default)**: Uses Advanced Trade v3 with HMAC credentials
- **Perpetual futures**: Requires INTX access + CDP (JWT) authentication + `COINBASE_ENABLE_DERIVATIVES=1`
- **Sandbox**: Spot trading only (no perpetuals); limited endpoint availability
- **WebSocket Subscriptions**: Up to 100 per connection (⚠️ needs verification)

## Python SDK & Tools

- **Coinbase Advanced Trade Python SDK**: https://github.com/coinbase/coinbase-advanced-py
  - Official SDK with authentication, HTTP pooling, and helpful methods
  - Recommended for production use

## Support & Status

- **API Status**: https://status.coinbase.com/
- **Coinbase Developer Platform**: https://www.coinbase.com/developer-platform
- **Community Forums**: Check GitHub Discussions or Stack Overflow tag `coinbase-api`

## Operational Reminders

| Scenario | API | Auth | Base URL |
|----------|-----|------|----------|
| **Spot (Production)** | Advanced Trade v3 | HMAC | `https://api.coinbase.com/api/v3/brokerage` |
| **Perpetuals (Production)** | Advanced Trade v3 | CDP/JWT | `https://api.coinbase.com/api/v3/brokerage` |
| **Spot (Sandbox)** | Advanced Trade v3 (Accounts/Orders only) | HMAC | `https://api-public.sandbox.exchange.coinbase.com/api/v3/brokerage` |
| **Perpetuals (Sandbox)** | ❌ Not supported | N/A | N/A |

## ⚠️ Items with Limited Official Documentation

**WebSocket Subscription Limit**: Documentation references "100 per connection" for Advanced Trade, but this is not explicitly documented in official Coinbase docs. Exchange accounts are documented as limited to 10 subscriptions per product/channel, but Advanced Trade limits appear tier-specific or undocumented. Verify with your API tier.

**OAuth2 Support**: OAuth2 is confirmed as officially supported for Advanced Trade API (references in official changelog and docs). The exact launch date is not publicly documented.

**Rate Limit Headers**:
- ✅ Confirmed: `CB-RATELIMIT-LIMIT`, `CB-RATELIMIT-REMAINING`, `CB-RATELIMIT-RESET`
- ✅ Confirmed: `retry-after` header on 429 responses
- Other headers: None found in official docs

---

## Internal References (This Repository)

**New Documentation** (added 2025-10-19):
- **[coinbase_api_endpoints.md](coinbase_api_endpoints.md)** - Complete REST endpoint catalog with production/sandbox availability
- **[coinbase_websocket_reference.md](coinbase_websocket_reference.md)** - WebSocket channels, authentication, and rate limits
- **[coinbase_auth_guide.md](coinbase_auth_guide.md)** - Authentication recipes (CDP/JWT, HMAC, OAuth2)
- **[coinbase_quick_reference.md](coinbase_quick_reference.md)** - Quick lookup card and cURL examples

**Existing Documentation**:
- **[coinbase_complete.md](coinbase_complete.md)** - Complete integration guide with environment setup
- **[compatibility_troubleshooting.md](compatibility_troubleshooting.md)** - Technical requirements and troubleshooting
- **[trading_logic_perps.md](trading_logic_perps.md)** - Perpetuals trading logic and INTX implementation

**Test Coverage**:
- **[coinbase_coverage_matrix.md](../testing/coinbase_coverage_matrix.md)** - API endpoint test coverage matrix
