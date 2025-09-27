# PR 1: Paper Engine Decoupling

## Summary
This PR decouples the Paper Execution Engine from implicit broker dependencies, enabling offline testing by default and supporting explicit dependency injection.

## Scope
**Goal**: Remove implicit broker dependency; support offline quotes by default; allow optional broker injection.

## Changes Made

### Modified Files
1. **src/bot_v2/orchestration/execution.py**
   - Added `quote_provider` and `broker` parameters to constructor
   - Removed `_init_broker()` auto-initialization
   - Updated `get_mid()` to use quote provider first, then broker
   - Modified `connect()` to handle optional broker
   - Added `set_broker()` helper method
   - Changed imports to relative

### New Files
2. **src/bot_v2/orchestration/quote_providers.py**
   - `create_static_quote_provider()` - Static prices for testing
   - `create_random_walk_provider()` - Prices with variation
   - `create_default_test_provider()` - Common crypto prices
   - `create_spread_provider()` - With bid-ask spread

3. **tests/unit/bot_v2/orchestration/test_paper_engine_decoupling.py**
   - Tests for default behavior (no broker/provider)
   - Tests with quote provider (offline)
   - Tests with injected broker
   - Tests for priority (provider > broker)
   - Tests for no network in offline mode

4. **docs/PAPER_ENGINE_DECOUPLING.md**
   - Complete usage guide
   - Migration instructions
   - Examples for all scenarios

### Updated Files
5. **tests/integration/paper_trade/test_coinbase_paper_integration.py**
   - Added `paper_engine_offline` fixture for offline tests
   - Added `paper_engine_with_broker` fixture for integration tests
   - New test for offline paper trading

## Behavior Changes

### Before
```python
# Broker was auto-created internally
engine = PaperExecutionEngine(initial_capital=10000)
# Always tried to connect to Coinbase
```

### After
```python
# Offline by default
engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=create_default_test_provider()
)

# Explicit broker when needed
broker = CoinbaseBrokerage(config)
engine = PaperExecutionEngine(
    initial_capital=10000,
    broker=broker
)
```

## Testing Performed

### Unit Tests
```bash
pytest tests/unit/bot_v2/orchestration/test_paper_engine_decoupling.py -v
# Result: 8 passed ✅
```

### Test Coverage
- ✅ Default behavior with no dependencies
- ✅ Offline trading with quote provider
- ✅ Broker injection and connection
- ✅ Quote provider priority over broker
- ✅ Dynamic broker addition via `set_broker()`
- ✅ Random walk price variation
- ✅ No network calls in offline mode

## Acceptance Criteria ✅

1. **Offline tests pass with no network** ✅
   - Tests use `quote_provider` and don't touch network
   
2. **Existing tests unchanged or updated pass** ✅
   - Integration tests updated to use explicit fixtures
   
3. **No hidden broker init in PaperExecutionEngine** ✅
   - `_init_broker()` removed, broker is injected only

## How to Validate

```bash
# Run unit tests
pytest tests/unit/bot_v2/orchestration/test_paper_engine_decoupling.py -v

# Run offline integration test
pytest tests/integration/paper_trade/test_coinbase_paper_integration.py::TestCoinbasePaperIntegration::test_offline_paper_trading -v

# Verify no network in tests
python -c "
from src.bot_v2.orchestration.execution import PaperExecutionEngine
from src.bot_v2.orchestration.quote_providers import create_default_test_provider

engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=create_default_test_provider()
)
print(f'Offline quote: {engine.get_mid(\"BTC-USD\")}')
print('✅ No network required!')
"
```

## Known Limitations
- Deprecation warnings for `datetime.utcnow()` (pre-existing, not introduced by this PR)
- Product catalog still requires broker connection (as intended)

## Migration Guide

For existing code using PaperExecutionEngine:

### Minimal Change (Keep Network)
```python
# Old
engine = PaperExecutionEngine(initial_capital=10000)

# New - explicit broker
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig

cfg = APIConfig(...)
broker = CoinbaseBrokerage(cfg)
engine = PaperExecutionEngine(
    initial_capital=10000,
    broker=broker
)
```

### Recommended (Offline Testing)
```python
from bot_v2.orchestration.quote_providers import create_default_test_provider

engine = PaperExecutionEngine(
    initial_capital=10000,
    quote_provider=create_default_test_provider()
)
```

## Next Steps
- **PR 2**: Type Consolidation - Migrate live_trade to core interfaces
- **PR 3**: Performance Optimizations - HTTP keep-alive and batching

## Checklist
- [x] Code changes complete
- [x] Unit tests added
- [x] Integration tests updated
- [x] Documentation added
- [x] No regressions in existing tests
- [x] Acceptance criteria met

---

**Branch**: `feat/paper-engine-decouple`
**Ready for review** ✅