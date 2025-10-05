# Integration Scenario Tests

**Purpose**: End-to-end scenario-based integration tests for broker and orchestration workflows.

---

## Overview

The `scenarios/` test suite provides comprehensive end-to-end testing of complete trading workflows, from signal generation to order execution, position management, and error recovery. These tests use realistic fixtures and validate behavior across multiple system components.

**Test Philosophy**:
- **Scenario-based**: Each test represents a realistic user workflow
- **End-to-end**: Tests cross multiple system boundaries (strategy → orchestration → broker)
- **Fixture-driven**: Reusable fixtures reduce boilerplate and ensure consistency
- **Documentation**: Tests serve as executable specifications of expected behavior

---

## Test Structure

```
scenarios/
├── conftest.py                      # Shared fixtures for all scenario tests
├── test_trading_lifecycle.py        # Complete trade lifecycle scenarios
├── test_broker_edge_cases.py        # Broker API edge cases and error handling
├── test_orchestration_state.py      # State management and persistence
└── README.md                        # This file
```

---

## Test Categories

### Trading Lifecycle (`test_trading_lifecycle.py`)

**Complete Trade Workflows**:
- Signal → Order → Fill → Position creation
- Position → Close signal → Liquidation
- Multi-strategy coordination and signal aggregation

**Risk Guard Enforcement**:
- Daily loss limit triggering reduce-only mode
- Max trade value blocking oversized orders
- Per-symbol position caps enforcement

**Error Recovery**:
- Order placement retry on rate limit
- Broker disconnection and reconnection
- Partial fill tracking and adjustment

**Position Reconciliation**:
- Position drift detection between bot and broker
- Manual trades external to bot
- State synchronization

### Broker Edge Cases (`test_broker_edge_cases.py`)

**Rate Limiting**:
- Exponential backoff on rate limit errors
- Multiple concurrent requests hitting rate limit
- Shared rate limit state across requests

**Network Errors**:
- Connection timeout retry
- WebSocket disconnection recovery
- DNS resolution failure handling

**Invalid Responses**:
- Missing required fields in API responses
- Unexpected enum values
- Negative or impossible values (e.g., negative balance)

**Order Status Tracking**:
- Status lifecycle: pending → open → filled
- Order cancelled before any fill
- Order rejected after initial acceptance

**Authentication and Permissions**:
- Expired API credentials
- Insufficient permissions for futures trading
- Read-only API key blocking trades

**Insufficient Funds**:
- Order blocked by insufficient funds
- Partial fill due to margin limits

### Orchestration State (`test_orchestration_state.py`)

**Bot Initialization**:
- Cold start (first-time initialization)
- Warm start with existing positions
- Startup with config changes

**Mark Price Tracking**:
- Window population with recent prices
- Window size limit and FIFO eviction
- Staleness detection and warnings

**Position State Synchronization**:
- Reconciliation detecting drift
- Position update after order fill
- Position removal after full close

**Background Tasks**:
- Tasks spawned on bot start
- Task error handling and restart
- Tasks cancelled cleanly on shutdown

**State Persistence**:
- Event store persistence across restarts
- Crash recovery from checkpoints
- Corrupted state recovery

**Concurrency**:
- Concurrent position updates
- Race conditions between order and fill

---

## Shared Fixtures (`conftest.py`)

### Configuration Fixtures

**`scenario_config`** - Standard bot configuration for scenarios
```python
def test_example(scenario_config):
    bot = PerpsBot(scenario_config)
    # Config includes: CANARY profile, BTC/ETH, mock broker, risk limits
```

### Broker Fixtures

**`funded_broker`** - Mock broker with $10K starting capital
```python
def test_example(funded_broker):
    balances = funded_broker.list_balances()
    # Returns: $10,000 USD, no positions
```

**`broker_with_positions`** - Mock broker with open BTC and ETH positions
```python
def test_example(broker_with_positions):
    positions = broker_with_positions.list_positions()
    # Returns: BTC and ETH positions with unrealized P&L
```

**`volatile_market_broker`** - Mock broker with volatile price movements
```python
def test_example(volatile_market_broker):
    quote = volatile_market_broker.get_quote("BTC-USD")
    # Returns: Price with ±5% volatility
```

### Factory Fixtures

**`realistic_order_factory`** - Create realistic order scenarios
```python
def test_example(realistic_order_factory):
    successful_order = realistic_order_factory.create_successful_market_order(
        symbol="BTC-USD",
        quantity=Decimal("0.01"),
    )
```

**`position_scenarios`** - Create position test scenarios
```python
def test_example(position_scenarios):
    profitable_position = position_scenarios.create_profitable_position(
        symbol="BTC-USD",
        entry_price=Decimal("48000"),
        current_price=Decimal("52000"),
    )
```

**`market_conditions`** - Simulate market conditions
```python
def test_example(market_conditions):
    volatile_market = market_conditions.create_volatile_market()
    # Returns: 8% daily volatility, medium liquidity, 20 bps spread
```

---

## Running Scenario Tests

### Run all scenario tests
```bash
pytest tests/integration/scenarios/ -v
```

### Run specific scenario file
```bash
pytest tests/integration/scenarios/test_trading_lifecycle.py -v
```

### Run tests by marker
```bash
# All integration scenarios
pytest -m "integration and scenario" -v

# Edge case scenarios only
pytest -m "edge_case" -v

# State management scenarios only
pytest -m "state" -v
```

### Run specific scenario
```bash
pytest tests/integration/scenarios/test_trading_lifecycle.py::TestCompleteTradeLifecycle::test_signal_to_order_to_position_flow -v
```

### Run with coverage
```bash
pytest tests/integration/scenarios/ --cov=bot_v2.orchestration --cov=bot_v2.features.brokerages -v
```

---

## Writing New Scenario Tests

### 1. Choose Appropriate Test File

- **Trading workflows**: `test_trading_lifecycle.py`
- **Broker errors**: `test_broker_edge_cases.py`
- **State management**: `test_orchestration_state.py`
- **New category**: Create new file `test_<category>.py`

### 2. Use Existing Fixtures

```python
@pytest.mark.asyncio
async def test_my_scenario(monkeypatch, tmp_path, scenario_config, funded_broker):
    """
    Scenario: <describe the workflow>

    Verifies:
    - <expected behavior 1>
    - <expected behavior 2>
    """
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    bot = PerpsBot(scenario_config)
    bot.broker = funded_broker

    # Test implementation
```

### 3. Create Realistic Test Data

Use factories instead of manual construction:

**Good**:
```python
order = realistic_order_factory.create_successful_market_order(
    symbol="BTC-USD",
    quantity=Decimal("0.01"),
)
```

**Avoid**:
```python
order = Mock(
    order_id="test-123",
    symbol="BTC-USD",
    side=OrderSide.BUY,
    status="filled",
    size=Decimal("0.01"),
    # ... 10 more fields
)
```

### 4. Document Scenario Intent

Each test should include:
- **Scenario**: One-line workflow description
- **Verifies**: Bullet list of expected behaviors

### 5. Apply Appropriate Markers

```python
@pytest.mark.integration      # All scenario tests
@pytest.mark.scenario          # Scenario-based test
@pytest.mark.edge_case         # Edge case or error handling
@pytest.mark.state             # State management test
@pytest.mark.slow              # Tests taking >1 second
@pytest.mark.asyncio           # Async test (required for async functions)
```

### 6. Handle Not-Yet-Implemented Features

If testing unimplemented functionality:

```python
@pytest.mark.asyncio
async def test_future_feature(scenario_config):
    """
    Scenario: <future workflow>

    Verifies:
    - <expected behavior>
    """
    pytest.skip("Feature X not yet implemented")
```

This documents expected behavior while avoiding test failures.

---

## Best Practices

### Do's

✅ **Use descriptive scenario descriptions**
```python
def test_daily_loss_limit_triggers_reduce_only(...):
    """
    Scenario: Loss exceeds daily limit → Reduce-only mode activated → New orders blocked
    """
```

✅ **Leverage fixtures for setup**
```python
def test_example(funded_broker, scenario_config):
    # Broker and config already set up
```

✅ **Test one workflow per test**
```python
# Good: Focused test
def test_signal_to_order_flow(...):
    # Test: signal generation → order placement

# Good: Separate test
def test_order_to_position_flow(...):
    # Test: order fill → position creation
```

✅ **Mock at appropriate boundaries**
```python
# Mock broker API responses
funded_broker.place_order.return_value = Mock(...)

# Don't mock internal orchestration logic
```

✅ **Use realistic test data**
```python
# Good: Realistic price
price = Decimal("50000.00")

# Avoid: Unrealistic price
price = Decimal("1.00")
```

### Don'ts

❌ **Don't test multiple unrelated workflows in one test**
```python
# Bad: Too much in one test
def test_everything(...):
    # Test signal generation
    # Test order placement
    # Test position management
    # Test risk guards
    # Test reconciliation
```

❌ **Don't duplicate unit test coverage**
```python
# Bad: This belongs in unit tests
def test_decimal_addition(...):
    assert Decimal("1.0") + Decimal("2.0") == Decimal("3.0")
```

❌ **Don't hardcode mock data inline**
```python
# Bad: Hardcoded mock
broker.get_quote.return_value = Mock(
    last=50000.0,
    bid=49950.0,
    ask=50050.0,
    ts=datetime.now(),
    volume=1000000,
)

# Good: Use factory
broker.get_quote.return_value = CoinbaseQuoteFactory.create_quote(
    symbol="BTC-USD",
    price="50000.00",
)
```

❌ **Don't skip assertions**
```python
# Bad: No verification
bot.place_order(...)
# Test ends without verifying anything

# Good: Verify behavior
result = bot.place_order(...)
assert result.status == "filled"
assert result.order_id is not None
```

❌ **Don't use `sleep()` for synchronization**
```python
# Bad: Flaky timing
await bot.run_cycle()
await asyncio.sleep(1)  # Hope background task finished
assert task_completed

# Good: Use explicit coordination
await bot.run_cycle()
await bot.lifecycle_service._cleanup()  # Wait for tasks
assert all_tasks_cancelled
```

---

## Continuous Integration

Scenario tests run automatically in CI on:
- Pull requests to `main`
- Daily scheduled runs
- Manual workflow dispatch

**CI Configuration**: `.github/workflows/integration-tests.yml`

**CI Requirements**:
- All scenario tests must pass
- Coverage must not decrease
- No flaky tests (retry limit: 2)

---

## Troubleshooting

### Test hangs or times out

**Cause**: Async task not properly cancelled

**Fix**: Ensure proper cleanup in fixtures
```python
@pytest.fixture
async def my_async_fixture():
    task = asyncio.create_task(background_work())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

### Mock not being called

**Cause**: Wrong attribute mocked or multiple mock instances

**Fix**: Verify mock assignment
```python
# Ensure bot uses the mocked broker
bot.broker = funded_broker

# Verify mock was configured
assert funded_broker.place_order.return_value is not None
```

### State pollution between tests

**Cause**: Shared mutable state not reset

**Fix**: Use `tmp_path` for persistent state
```python
def test_example(tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    # Each test gets isolated tmp directory
```

### Flaky test due to timing

**Cause**: Race condition or timing assumption

**Fix**: Use explicit synchronization
```python
# Bad: Assumes timing
await bot.update_marks()
await asyncio.sleep(0.1)  # Hope it's done

# Good: Explicit completion
await bot.update_marks()
# update_marks is awaited, guaranteed complete
```

---

## Metrics and Coverage

**Current Scenario Test Coverage**:
- Trading lifecycle: 6 scenarios (4 implemented, 2 skipped)
- Broker edge cases: 15 scenarios (8 implemented, 7 skipped)
- Orchestration state: 12 scenarios (6 implemented, 6 skipped)

**Coverage Gaps** (prioritized for implementation):
1. Multi-strategy coordination (not yet implemented)
2. WebSocket reconnection (tested separately)
3. Background task error recovery (not yet implemented)
4. Concurrent state updates (requires threading support)

**Target Coverage**: 90% of broker and orchestration modules

---

## Future Enhancements

### Planned Additions

1. **Performance Scenarios**
   - High-frequency trading stress test
   - Large position portfolio scaling
   - Memory usage under long-running bot

2. **Multi-Market Scenarios**
   - Cross-market arbitrage
   - Portfolio rebalancing across symbols
   - Correlated position management

3. **Failure Injection**
   - Chaos testing (random broker failures)
   - Network partition scenarios
   - Resource exhaustion (OOM, disk full)

4. **Regulatory Scenarios**
   - Trading halt detection and compliance
   - Position limit enforcement
   - Wash trade prevention

---

**Last Updated**: 2025-10-05
**Maintainers**: Trading Infrastructure Team
**Status**: ✅ Active Development

For questions or contributions, see `CONTRIBUTING.md` or reach out to the team.
