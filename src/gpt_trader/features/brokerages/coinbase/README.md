# Coinbase Brokerage Integration

Complete implementation of Coinbase Advanced Trade API v3 and Legacy Exchange API support.

## Features Implemented

### Core Components
- ✅ **REST Client** (`client.py`): Full HTTP client with auth, retries, and rate limiting
- ✅ **WebSocket Streaming** (`ws.py`): Real-time data with auto-reconnect and transport abstraction
- ✅ **Brokerage Adapter** (`adapter.py`): Implements `IBrokerage` interface for Coinbase
- ✅ **Data Models** (`models.py`): Type-safe data mappers and helpers
- ✅ **Error Handling** (`errors.py`): Comprehensive error mapping
- ✅ **Authentication**: Both HMAC and CDP JWT (v2) authentication supported
- ✅ **Endpoint Registry** (`endpoints.py`): Complete endpoint definitions
- ✅ **Transport Layer** (`transports.py`): Pluggable WebSocket transports for testing

### API Mode Support
The integration supports both Coinbase API modes:

1. **Advanced Trade API (v3)** - Production trading with full features
   - Base URL: `https://api.coinbase.com`
   - Endpoints: `/api/v3/brokerage/*`
   - Auth: HMAC (no passphrase) or CDP JWT
   - **Full feature set**: Portfolios, order management, INTX, CFM, etc.

2. **Legacy Exchange API** - Sandbox testing with limited features
   - Base URL: `https://api-public.sandbox.exchange.coinbase.com`
   - Endpoints: `/products`, `/accounts`, `/orders`
   - Auth: HMAC with passphrase required
   - **Limited feature set**: Basic trading only (no portfolios, order preview, etc.)

**Note**: Exchange mode's feature set is intentionally limited. Advanced order management, portfolio operations, and derivatives trading require Advanced Trade API.

3. Spot-First (Recommended If Not INTX-Eligible)
   - Default symbols now use spot pairs (e.g., `BTC-USD`).
   - Perpetuals are blocked unless `COINBASE_ENABLE_DERIVATIVES=1` is set and your account is eligible.
   - Error messages guide you to switch to spot symbols if a `-PERP` product is used inadvertently.

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

# Sandbox mode (forces exchange mode)
COINBASE_SANDBOX=1  # Set to 1 for sandbox

# HMAC Authentication
COINBASE_API_KEY=your-api-key
COINBASE_API_SECRET=your-api-secret
COINBASE_API_PASSPHRASE=your-passphrase  # Required for exchange mode

# OR CDP JWT Authentication
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
from gpt_trader.orchestration.broker_factory import create_brokerage

# Will auto-configure from environment
broker, event_store, market_data, product_catalog = create_brokerage()
broker.connect()

# Get products
products = broker.list_products()
print(f"Available products: {len(products)}")
```

### Sandbox Testing
```bash
# Set environment for sandbox
export COINBASE_SANDBOX=1
export COINBASE_API_KEY=your-sandbox-key
export COINBASE_API_SECRET=your-sandbox-secret
export COINBASE_API_PASSPHRASE=your-sandbox-passphrase

# Run smoke test (spot)
poetry run coinbase-trader --profile dev --symbols BTC-USD --dry-run --dev-fast
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
# Run shared validation harness for brokerage safety checks
python scripts/validation/verify_core.py --check all
```

### Unit Tests
```bash
# Run Coinbase-specific tests
pytest tests/unit/gpt_trader/features/brokerages/coinbase/ -v

# Run critical fixes tests
pytest tests/unit/gpt_trader/features/brokerages/coinbase/test_critical_fixes.py -v
```

## Known Limitations

1. **INTX Eligibility for Perps**
   - Placing orders for Coinbase Perpetuals requires a Coinbase International Exchange (INTX) account and API keys created at international.coinbase.com.
   - If you are not eligible for INTX, keep `COINBASE_ENABLE_DERIVATIVES=0` and use spot symbols like `BTC-USD`.

2. **Sandbox Limitations**
   - Advanced Trade API does not have a public sandbox
   - Sandbox only supports Legacy Exchange API endpoints
   - Some features (portfolios, converts) not available in sandbox

3. **Authentication Notes**
   - Exchange mode requires passphrase for HMAC auth
   - CDP JWT not compatible with Exchange API
   - Advanced Trade supports both HMAC (no passphrase) and CDP JWT

4. **WebSocket Requirements**
   - Install the live trading extras to enable websocket-client support:
     - `pip install gpt-trader[live-trade]`, or
     - `poetry install --with live-trade`

## Troubleshooting

### Common Issues

**404 Errors in Sandbox**
- Ensure `COINBASE_API_MODE=exchange` when using sandbox
- Check that passphrase is set for HMAC auth

**WebSocket Connection Failures**
- Ensure the live trade extras are installed (`pip install gpt-trader[live-trade]`)
- Check WebSocket URL matches API mode

**Authentication Errors**
- Verify API keys are correct
- Ensure passphrase is provided for exchange mode
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
- Sandbox mode detection and configuration
- Environment template updated

### 🚧 In Progress:
- Complete endpoint routing for all methods (20+ remaining)
- Comprehensive test coverage for both API modes
- Paper engine decoupling

The integration is functional for basic operations. Full production readiness pending completion of endpoint routing.
