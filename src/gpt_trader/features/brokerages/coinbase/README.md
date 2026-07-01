# Coinbase Brokerage Integration

Complete implementation of Coinbase Advanced Trade API v3 with limited Legacy Exchange (public endpoints only).

## Features Implemented

### Core Components
- ✅ **REST Client** (`client.py`): Full HTTP client with auth, retries, and rate limiting
- ✅ **WebSocket Streaming** (`ws.py`): Real-time data with auto-reconnect and transport abstraction
- ✅ **REST Service** (`rest_service.py`): High-level service layer for Coinbase operations
- ✅ **Data Models** (`models.py`): Type-safe data mappers and helpers
- ✅ **Error Handling** (`errors.py`): Comprehensive error mapping
- ✅ **Authentication**: JWT-based CDP keys (SimpleAuth/CDPJWTAuth)
- ✅ **Endpoint Registry** (`endpoints.py`): Complete endpoint definitions
- ✅ **Transport Layer** (`transports.py`): Pluggable WebSocket transports for testing

### API Mode Support
The integration supports both Coinbase API modes:

1. **Advanced Trade API (v3)** - Production trading with full features
   - Base URL: `https://api.coinbase.com`
   - Endpoints: `/api/v3/brokerage/*`
   - Auth: JWT (CDP key)
   - **Full feature set**: Portfolios, order management, CFM, etc. (INTX was removed; see [decision record](../../../../../docs/decisions/intx-default-derivatives-venue.md))

2. **Legacy Exchange API** - Sandbox testing with limited features
   - Base URL: `https://api-public.sandbox.exchange.coinbase.com`
   - Endpoints: `/products`, `/accounts`, `/orders`
   - Auth: Public endpoints only (no authenticated trading in GPT-Trader)
   - **Limited feature set**: Market data only (no portfolios, order preview, etc.)

**Note**: Exchange mode's feature set is intentionally limited. Advanced order management, portfolio operations, and derivatives trading require Advanced Trade API with JWT credentials.

3. Spot-First (Default)
   - Default symbols use spot pairs (e.g., `BTC-USD`).
   - INTX perpetuals were removed; symbols ending in `-PERP` are coerced to their spot equivalents with a warning (see [Deprecations](../../../../../docs/DEPRECATIONS.md)). Other retired INTX forms (e.g., `BTC-PERP-USDC`) are not coerced: CFM configurations reject them at config validation, while spot configurations pass them through unchanged rather than mapping them to spot.
   - CFM futures are the only supported derivatives venue; enable via `TRADING_MODES=cfm` + `CFM_ENABLED=1`.

### Critical Fixes Applied

1. **API Mode Routing**: Automatic detection and proper endpoint routing based on mode
2. **WebSocket Transport**: Default transport initialization, no more assertion errors
3. **Sandbox Support**: Correctly uses Exchange API for sandbox testing
4. **Environment Validation**: Clear warnings for configuration issues

## Configuration

Configure via environment variables in `.env`:

```bash
# Broker selection
BROKER=coinbase

# API Mode (auto-detected if not set)
COINBASE_API_MODE=advanced  # or "exchange"

# Sandbox mode (public endpoints only; no authenticated trading)
COINBASE_SANDBOX=1  # Set to 1 for public Exchange endpoints

# JWT Authentication (CDP key)
COINBASE_CREDENTIALS_FILE=/path/to/cdp_key.json
# or set both env vars:
COINBASE_CDP_API_KEY=your-cdp-key-name
COINBASE_CDP_PRIVATE_KEY=your-ec-private-key-pem

# Optional overrides
COINBASE_API_BASE=https://custom-url.com
COINBASE_WS_URL=wss://custom-ws-url.com
# WebSocket runtime tuning
COINBASE_WS_CONNECT_TIMEOUT=5.0       # Optional connect timeout (seconds)
COINBASE_WS_SUBPROTOCOLS=feed,private # Comma-separated list of subprotocols
COINBASE_WS_ENABLE_TRACE=1            # Enable websocket-client trace logging
```

The CLI and runtime settings snapshot capture these overrides once and reuse them across REST
and WebSocket transports, so the same values apply to reconnects, data providers, and the
adapter without repeated `os.getenv` calls.

## Usage Examples

### Basic Connection
```python
from gpt_trader.features.brokerages.factory import create_brokerage

# Will auto-configure from environment
broker, event_store, market_data, product_catalog = create_brokerage()
broker.connect()

# Get products
products = broker.list_products()
print(f"Available products: {len(products)}")
```

### Sandbox Testing
```bash
# Advanced Trade has no authenticated sandbox; use the mock broker instead
MOCK_BROKER=1 uv run gpt-trader run --profile dev --dev-fast
```

### WebSocket Streaming
```python
from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription

ws = CoinbaseWebSocket("wss://advanced-trade-ws.coinbase.com")
ws.subscribe(WSSubscription(
    channels=["ticker"],
    product_ids=["BTC-USD", "ETH-USD"]
))

for message in ws.stream_messages():
    print(f"Received: {message}")
```

## Testing

### Validation Script
```bash
# Run preflight checks (JWT credentials required for remote checks)
uv run python scripts/production_preflight.py --profile dev --warn-only
```

### Unit Tests
```bash
# Run Coinbase-specific tests
pytest tests/unit/gpt_trader/features/brokerages/coinbase/ -v

# Run targeted auth tests
pytest tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_auth.py -v
```

## Known Limitations

1. **INTX Removed**
   - INTX perpetuals were removed as a supported venue (see [decision record](../../../../../docs/decisions/intx-default-derivatives-venue.md)); `-PERP`-suffixed symbols are coerced to their spot equivalents.
   - `COINBASE_ENABLE_INTX_PERPS` is retained only as a deprecated alias: a truthy value warns and substitutes for `CFM_ENABLED=1` only (config validation still requires `TRADING_MODES` to include `cfm`); falsey/unset values are ignored (see [Deprecations](../../../../../docs/DEPRECATIONS.md)). Use spot symbols like `BTC-USD`, or `TRADING_MODES=cfm` + `CFM_ENABLED=1` for CFM futures.

2. **Sandbox Limitations**
   - Advanced Trade API does not have an authenticated sandbox
   - Exchange mode is limited to public endpoints only
   - Use the mock broker for integration testing

3. **Authentication Notes**
   - JWT (CDP) credentials are required for authenticated endpoints
   - HMAC is not implemented in GPT-Trader

4. **WebSocket Requirements**
   - Install the live trading extras to enable websocket-client support:
     - `pip install gpt-trader[live-trade]`, or
     - `uv sync --extra live-trade`

## Troubleshooting

### Common Issues

**404 Errors in Sandbox**
- `COINBASE_SANDBOX=1` only enables public Exchange endpoints
- Use the mock broker for authenticated flows

**WebSocket Connection Failures**
- Ensure the live trade extras are installed (`pip install gpt-trader[live-trade]`)
- Check WebSocket URL matches API mode

**Authentication Errors**
- Verify CDP key name + private key are correct
- Check CDP private key format (PEM with EC)

## Development

### Adding New Endpoints
1. Add to `ENDPOINT_MAP` in `client.py._get_endpoint_path()`
2. Implement method in `CoinbaseClient`
3. Add wrapper in `CoinbaseBrokerage` adapter
4. Update tests

### Custom Transports
```python
from gpt_trader.features.brokerages.coinbase.transports import MockTransport

# For testing
transport = MockTransport([
    {"type": "ticker", "price": "50000.00"}
])
ws = CoinbaseWebSocket("wss://test", transport=transport)
```

## Status: Critical Fixes Applied ⚠️

### ✅ Completed:
- WebSocket transport initialization fixed
- Duplicate code removed
- Basic endpoint routing for common methods
- Sandbox mode detection for public endpoints
- Environment template updated

### 🚧 In Progress:
- Complete endpoint routing for all methods (20+ remaining)
- Comprehensive test coverage for both API modes
- Paper engine decoupling

The integration is functional for basic operations. Full production readiness pending completion of endpoint routing.
