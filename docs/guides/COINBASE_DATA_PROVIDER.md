# Coinbase Data Provider Integration

## Overview

The GPT-Trader system now supports fetching real market data from the Coinbase API, in addition to the existing mock data used for testing. This allows for more realistic backtesting and analysis using actual market conditions.

## Architecture

The data provider system uses a factory pattern with multiple implementations:

```
DataProvider (Abstract Base)
├── MockProvider       - Deterministic test data
├── YFinanceProvider   - Yahoo Finance (equities)
├── AlpacaProvider     - Alpaca Markets
└── CoinbaseDataProvider - Real Coinbase API data (NEW)
```

## Configuration

Control data source via environment variables:

```bash
# Use mock data (default for testing)
COINBASE_USE_REAL_DATA=0

# Use real Coinbase API data
COINBASE_USE_REAL_DATA=1

# Enable WebSocket streaming (optional)
COINBASE_ENABLE_STREAMING=1

# Cache TTL in seconds (default: 5)
COINBASE_DATA_CACHE_TTL=5
```

## Usage Examples

### Basic Usage

```python
from bot_v2.data_providers import get_data_provider

# Automatically selects provider based on environment
provider = get_data_provider()

# Get current price
btc_price = provider.get_current_price('BTC')

# Get historical data
df = provider.get_historical_data('BTC', period='7d', interval='1d')

# Get multiple symbols
data = provider.get_multiple_symbols(['BTC', 'ETH', 'SOL'], period='30d')
```

### Explicit Provider Selection

```python
# Force specific provider type
provider = get_data_provider('coinbase')  # Real Coinbase data
provider = get_data_provider('mock')      # Mock data

# Or use the factory function
from bot_v2.data_providers.coinbase_provider import create_coinbase_provider

# Explicit configuration
provider = create_coinbase_provider(
    use_real_data=True,
    enable_streaming=True
)
```

### With Context Manager (for streaming)

```python
from bot_v2.data_providers.coinbase_provider import CoinbaseDataProvider

# Streaming lifecycle managed automatically
with CoinbaseDataProvider(enable_streaming=True) as provider:
    # WebSocket connection active here
    prices = provider.get_current_price('BTC-PERP')
# WebSocket closed on exit
```

## Mock vs Real Data

### Mock Data (Default)
- **Pros:**
  - Deterministic results for testing
  - No network dependencies
  - No API rate limits
  - Fast execution
  - Predictable for unit tests

- **Cons:**
  - Not real market conditions
  - Can't test actual volatility
  - Fixed patterns

### Real Coinbase Data
- **Pros:**
  - Actual market prices
  - Real volatility and spreads
  - Live market conditions
  - WebSocket streaming available
  - Accurate backtesting

- **Cons:**
  - Network dependent
  - API rate limits apply
  - Non-deterministic
  - Requires internet connection

## Symbol Normalization

The provider automatically normalizes symbols:

- `'BTC'` → `'BTC-USD'` (spot)
- `'ETH'` → `'ETH-USD'` (spot)
- `'AAPL'` → `'AAPL-USD'` (spot/equities)

Perpetual symbols (`*-PERP`) remain available for INTX-approved accounts when derivatives are enabled.

## Implementation Details

### Data Flow

1. **REST API** (default):
   - Public market data endpoints
   - No authentication required
   - Cached with configurable TTL
   - Rate limited (100 req/min)

2. **WebSocket** (optional):
   - Real-time ticker updates
   - Reduced API calls
   - Background thread management
   - Automatic reconnection

### Caching Strategy

- Historical data: Cached for `cache_ttl` seconds (default: 5)
- Current prices: WebSocket cache if fresh (<5s), REST fallback
- Product info: Cached for 15 minutes

### Error Handling

- Network failures: Falls back to mock data
- Invalid symbols: Returns mock data with warning
- Rate limits: Automatic throttling
- WebSocket disconnect: Auto-reconnect with backoff

## Testing

### Run Tests

```bash
# Test with mock data (default)
poetry run python demos/test_coinbase_data_provider.py

# Test with real API
COINBASE_USE_REAL_DATA=1 poetry run python demos/test_coinbase_data_provider.py

# Test transition demo
poetry run python demos/coinbase_data_transition_demo.py
```

### Unit Testing

Tests should use mock data for determinism:

```python
def test_strategy():
    # Force mock provider for tests
    os.environ['TESTING'] = 'true'
    provider = get_data_provider()
    assert provider.__class__.__name__ == 'MockProvider'
```

## Migration Path

### Phase 1: Current State (Complete)
- ✅ Mock data for all testing
- ✅ Deterministic results
- ✅ No external dependencies

### Phase 2: Hybrid Mode (Complete)
- ✅ CoinbaseDataProvider implemented
- ✅ Environment-based switching
- ✅ Backward compatible

### Phase 3: Production Ready
- Use real data for backtesting
- Mock data for unit tests
- Real data for integration tests
- Streaming for live trading

## Performance Considerations

- **Backtest Speed**: Mock data is ~100x faster
- **API Limits**: 100 requests/minute (REST)
- **WebSocket**: 100 channel subscriptions max
- **Memory**: Minimal overhead (<10MB for cache)

## Troubleshooting

### Common Issues

1. **"No data returned for symbol"**
   - Check symbol format (use perpetuals format)
   - Verify API connectivity
   - Falls back to mock data automatically

2. **Rate limit errors**
   - Reduce request frequency
   - Enable caching (increase TTL)
   - Use WebSocket streaming

3. **WebSocket disconnects**
   - Auto-reconnects with exponential backoff
   - Check network stability
   - Verify subscription limits

## Future Enhancements

- [ ] Historical data pagination for large requests
- [ ] Order book depth via Level 2 data
- [ ] Trade tick data streaming
- [ ] Cross-exchange arbitrage data
- [ ] Options chain data support

## References

- [Coinbase API Documentation](https://docs.cdp.coinbase.com/advanced-trade/docs)
- [WebSocket API](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview)
- [Rate Limits](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-rate-limits)
