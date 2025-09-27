# Paper Engine Decoupling Guide

## Overview
The Paper Execution Engine has been decoupled from broker dependencies to support offline testing and explicit dependency injection.

## Key Changes

### 1. Constructor Parameters
The `PaperExecutionEngine` now accepts optional dependencies:

```python
def __init__(
    self,
    commission: float = 0.006,
    slippage: float = 0.001,
    initial_capital: float = 10_000,
    config: Optional[Dict] = None,
    bot_id: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    quote_provider: Optional[Callable[[str], Optional[float]]] = None,  # NEW
    broker: Optional[IBrokerage] = None  # NEW
)
```

### 2. Quote Provider Interface
A quote provider is a simple callable that returns prices:

```python
def quote_provider(symbol: str) -> Optional[float]:
    """Return mid price for symbol, or None if not available."""
    return price
```

### 3. Quote Resolution Priority
The engine resolves quotes in this order:
1. **Quote Provider** (if set) - highest priority
2. **Broker** (if set and connected) - fallback
3. **None** - if no quote source available

## Usage Examples

### Offline Testing (Default)
```python
from bot_v2.orchestration.execution import PaperExecutionEngine
from bot_v2.orchestration.quote_providers import create_default_test_provider

# Completely offline - no network
engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=create_default_test_provider()
)

# No connection needed
price = engine.get_mid("BTC-USD")  # Returns 50000.0 from test provider
trade = engine.buy("BTC-USD", 1000)  # Works offline
```

### With Static Prices
```python
from bot_v2.orchestration.quote_providers import create_static_quote_provider

# Custom static prices
provider = create_static_quote_provider({
    "BTC-USD": 45000.0,
    "ETH-USD": 2800.0
})

engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=provider
)
```

### With Random Walk
```python
from bot_v2.orchestration.quote_providers import create_random_walk_provider

# Prices that vary slightly each call
provider = create_random_walk_provider(
    base_prices={"BTC-USD": 50000.0},
    volatility=0.01  # 1% max change
)

engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=provider
)
```

### With Real Broker (Integration Tests)
```python
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig

# Only when real quotes are needed
cfg = APIConfig(
    api_key="...",
    api_secret="...",
    base_url="https://api.coinbase.com"
)
broker = CoinbaseBrokerage(cfg)

engine = PaperExecutionEngine(
    initial_capital=10000,
    broker=broker  # Explicit injection
)

# Must connect to use broker quotes
engine.connect()
price = engine.get_mid("BTC-USD")  # Real price from Coinbase
```

### Dynamic Broker Addition
```python
# Start offline
engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=create_default_test_provider()
)

# Later add broker if needed
broker = create_real_broker()
engine.set_broker(broker)
engine.connect()
```

## Testing Strategy

### Unit Tests
- Use `quote_provider` with static prices
- No broker, no network calls
- Fast and deterministic

### Integration Tests
- Explicitly inject broker when needed
- Skip if credentials not available
- Test real quote fetching separately

### Example Test Structure
```python
class TestPaperTrading:
    def test_offline_trading(self):
        """Fast offline test."""
        engine = PaperExecutionEngine(
            quote_provider=create_default_test_provider()
        )
        # Test logic...
    
    @pytest.mark.integration
    @pytest.mark.skipif(not has_credentials(), reason="No credentials")
    def test_with_real_quotes(self):
        """Integration test with real broker."""
        broker = create_real_broker()
        engine = PaperExecutionEngine(broker=broker)
        engine.connect()
        # Test with real quotes...
```

## Migration Guide

### Before (Auto-init broker)
```python
engine = PaperExecutionEngine(initial_capital=10000)
# Broker was auto-created internally
```

### After (Explicit dependencies)
```python
# For offline testing
engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=create_default_test_provider()
)

# For real quotes
broker = CoinbaseBrokerage(config)
engine = PaperExecutionEngine(
    initial_capital=10000,
    broker=broker
)
```

## Benefits

1. **No Hidden Dependencies**: Broker is explicit, not auto-created
2. **Offline by Default**: Tests run without network
3. **Flexible Testing**: Easy to mock different market conditions
4. **Better Performance**: No network I/O in unit tests
5. **Clear Boundaries**: Obvious when real broker is needed

## Available Quote Providers

- `create_default_test_provider()` - Common crypto prices
- `create_static_quote_provider(prices)` - Custom static prices
- `create_random_walk_provider(base_prices, volatility)` - Varying prices
- `create_spread_provider(mid_prices, spread_bps)` - With bid-ask spread

## API Reference

### PaperExecutionEngine Methods
- `get_mid(symbol: str) -> Optional[float]` - Get quote from provider or broker
- `connect() -> bool` - Connect broker if provided
- `set_broker(broker: IBrokerage) -> None` - Set broker after initialization

### Quote Provider Helpers
See `src/bot_v2/orchestration/quote_providers.py` for all available providers.

## Next Steps
- PR 2: Type Consolidation - Migrate live_trade to core interfaces
- PR 3: Performance Optimizations - HTTP keep-alive and batching