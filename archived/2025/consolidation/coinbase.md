# Coinbase Integration Reference

---
status: current
last-updated: 2025-01-01
consolidates:
  - COINBASE_README.md
  - coinbase.md
  - coinbase_endpoints.md
  - COINBASE_API_SETUP.md
  - COINBASE_CDP_SETUP.md
  - COINBASE_CDP_INTEGRATION_COMPLETE.md
  - COINBASE_CB_VERSION_FIX.md
---

## Overview

Complete reference for Coinbase perpetual futures integration with GPT-Trader V2.

This module provides integration with Coinbase Advanced Trade API for cryptocurrency trading, including full perpetual futures support.

## Features

- Full Advanced Trade API v3 support
- WebSocket real-time data feeds
- Perpetual futures (BTC-PERP, ETH-PERP, SOL-PERP)
- CDP JWT authentication for derivatives
- Rate limiting and throttling
- Automatic retries with exponential backoff
- Connection pooling and keep-alive
- Product catalog with trading rules
- Order placement and management
- Portfolio tracking with PnL
- Comprehensive risk management

## Configuration

### Environment Variables

The Coinbase integration supports multiple authentication methods and environments:

#### Production (Perpetuals)
```bash
# Core Settings
BROKER=coinbase
COINBASE_SANDBOX=0                    # 0 for production
COINBASE_API_MODE=advanced            # Required for perpetuals
COINBASE_AUTH_TYPE=JWT                # CDP authentication

# CDP Credentials
COINBASE_PROD_CDP_API_KEY=organizations/{org-id}/apiKeys/{key-id}
COINBASE_PROD_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
[Your private key here]
-----END EC PRIVATE KEY-----"

# Feature Flags
COINBASE_ENABLE_DERIVATIVES=1         # Enable perpetuals
COINBASE_ENABLE_TRADING=1             # Enable live trading
```

#### Sandbox (Spot Only)
```bash
# Core Settings  
COINBASE_SANDBOX=1
COINBASE_API_MODE=exchange

# Legacy HMAC Auth
COINBASE_SANDBOX_API_KEY=your_key
COINBASE_SANDBOX_API_SECRET=your_secret
COINBASE_SANDBOX_API_PASSPHRASE=your_passphrase
```

### CDP Authentication Setup

1. **Create CDP API Key** at [Coinbase Developer Platform](https://portal.cdp.coinbase.com/)
2. **Required Permissions**:
   - `accounts:read` - Read account information
   - `orders:read` - View orders
   - `orders:create` - Place orders
   - `orders:cancel` - Cancel orders
   - `products:read` - View products

3. **Key Format**: `organizations/{org-id}/apiKeys/{key-id}`
4. **Private Key**: Must include EC PRIVATE KEY headers

## API Endpoints

### Public Endpoints (No Auth)
- `/api/v3/brokerage/market/products` - List products
- `/api/v3/brokerage/market/products/{id}/ticker` - Get ticker
- `/api/v3/brokerage/market/products/{id}/candles` - Get candles

### Private Endpoints (Auth Required)
- `/api/v3/brokerage/accounts` - List accounts
- `/api/v3/brokerage/orders` - Order management
- `/api/v3/brokerage/portfolios` - Portfolio details
- `/api/v3/brokerage/cfm/positions` - Perpetual positions

## Performance Optimizations

### Connection Pooling
- Automatic HTTP connection reuse (keep-alive)
- Reduces TCP handshake overhead
- 20-40ms latency improvement per request

### Rate Limiting
- Sliding window tracking (100 req/min default)
- Warning at 80% of limit
- Automatic throttling at limit

### Retry Logic
- Exponential backoff with jitter
- Automatic retry for transient errors
- Configurable max retries (default: 3)

## Troubleshooting

For common issues and solutions, see [Coinbase Troubleshooting Guide](coinbase_troubleshooting.md).

### Quick Diagnostics
```bash
# Test CDP authentication
python scripts/diagnose_cdp_key.py

# Check sandbox balance
python scripts/check_sandbox_balance.py

# Test WebSocket connectivity
python scripts/ws_probe.py
```
