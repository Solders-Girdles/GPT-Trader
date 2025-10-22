# Orchestration Test Architecture Guide

## Overview

This guide captures the proven test patterns and architectural decisions established during the orchestration test coverage enhancement initiative (2024-10-20).

## Achievements Summary

### Coverage Metrics
- **Overall Orchestration Coverage**: 42.73% (significant uplift from ~26%)
- **Key Module Coverage Gains**:
  - Execution Coordinator: 36.90% → 55.22% (+18.32%)
  - Telemetry Coordinator: ~54% → 73.46% (+19%+)
  - Order Reconciler: 88.24% → 92.35% (+4.11%)

### Test Quality Improvements
- **32/32 tests passing** in Execution Coordinator (100% success rate)
- **Production semantics alignment** - tests now match log-and-continue behavior
- **API compatibility fixes** - resolved constructor and method mismatches

## Core Test Patterns

### 1. Async Loop Control Pattern

**Purpose**: Test background task lifecycle with proper cancellation and error handling.

```python
@pytest.mark.asyncio
async def test_background_task_with_error_handling(self) -> None:
    """Test background task handles exceptions and continues operation."""
    coordinator = ExecutionCoordinator(_make_context())
    coordinator.initialize(context)

    # Mock to trigger exception after first call
    call_count = 0
    def failing_guard_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated failure")
        return True

    # Mock the runtime guard execution
    with patch('bot_v2.orchestration.coordinators.execution.run_in_thread', side_effect=failing_guard_call):
        # Start background task
        task = asyncio.create_task(coordinator.run_runtime_guards())

        # Let it run and handle error
        await asyncio.sleep(0.1)

        # Task should still be running after error
        assert not task.done()

        # Clean up
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
```

**Key Principles**:
- Use `asyncio.create_task()` for genuine async operations
- Test error recovery by triggering exceptions in controlled manner
- Always clean up with proper cancellation
- Verify tasks continue running after errors

### 2. Log-and-Continue Assertion Pattern

**Purpose**: Test production semantics where errors are logged but don't crash the system.

```python
@pytest.mark.asyncio
async def test_execute_decision_handles_invalid_input_gracefully(self) -> None:
    """Test execute_decision handles invalid input with logging instead of exceptions."""
    coordinator = ExecutionCoordinator(_make_context())
    coordinator.initialize(context)

    decision = Mock()
    mark = Decimal("0")  # Invalid mark
    product = _create_test_product()

    # Should handle invalid input gracefully by logging error and returning
    result = await coordinator.execute_decision("BTC-PERP", decision, mark, product, None)
    # Method returns None when error occurs during execution
    assert result is None
```

**Key Principles**:
- Don't expect exceptions when the implementation logs errors
- Test return values (usually None) to indicate graceful handling
- Focus on behavior outcomes rather than exception paths
- Match production semantics exactly

### 3. Context Immutability Pattern

**Purpose**: Test configuration changes using proper context update methods.

```python
def test_config_controller_integration(self) -> None:
    """Test proper integration with config controller."""
    config_controller = Mock()
    context = _make_context()
    # Use with_updates for context modifications (dataclass is frozen)
    context = context.with_updates(config_controller=config_controller)
    coordinator = ExecutionCoordinator(context)
    coordinator.initialize(context)

    # Should have config controller reference
    assert coordinator._config_controller is not None

    # Test context updates
    new_config_controller = Mock()
    new_context = context.with_updates(config_controller=new_config_controller)
    coordinator.update_context(new_context)
    assert coordinator._config_controller is new_config_controller
```

**Key Principles**:
- Never modify frozen dataclasses directly
- Use `with_updates()` method for context changes
- Test both initial setup and dynamic updates
- Respect dataclass immutability constraints

### 4. Diff Builder Pattern for Reconciliation

**Purpose**: Test order reconciliation with comprehensive scenario coverage.

```python
def test_reconcile_missing_orders_comprehensive(self) -> None:
    """Test order reconciliation with various mismatch scenarios."""
    # Create test scenarios
    local_orders = {
        "local-only": ScenarioBuilder.create_order(id="local-only"),
        "shared": ScenarioBuilder.create_order(id="shared"),
    }
    exchange_orders = {
        "exchange-only": ScenarioBuilder.create_order(id="exchange-only"),
        "shared": ScenarioBuilder.create_order(id="shared"),
    }

    # Calculate expected diff
    expected_diff = OrderDiff(
        missing_on_exchange={"local-only": local_orders["local-only"]},
        missing_locally={"exchange-only": exchange_orders["exchange-only"]},
    )

    # Test diff calculation
    actual_diff = OrderReconciler.diff_orders(local_orders, exchange_orders)
    assert actual_diff == expected_diff
```

**Key Principles**:
- Use builder patterns for complex test data creation
- Test both individual scenarios and edge cases
- Validate diff calculations comprehensively
- Include malformed data scenarios for robustness

### 5. Component Integration Pattern

**Purpose**: Test end-to-end workflows with realistic component interactions.

```python
@pytest.mark.asyncio
async def test_background_task_integration_with_orchestration(self) -> None:
    """Test background tasks integrate properly with orchestration components."""
    context = _make_context()
    coordinator = ExecutionCoordinator(context)
    coordinator.initialize(context)

    # Mock runtime state components
    mock_engine = Mock()
    coordinator.context.runtime_state.exec_engine = mock_engine

    # Start background tasks
    coordinator.start_background_tasks()

    # Verify tasks are running
    assert len(coordinator._background_tasks) == 2

    # Mock execution to verify integration
    with patch('bot_v2.orchestration.coordinators.execution.run_in_thread') as mock_run:
        # Let tasks run briefly
        await asyncio.sleep(0.1)

        # Verify background tasks executed
        assert mock_run.call_count >= 1

    # Clean shutdown
    coordinator.stop_background_tasks()
    assert len(coordinator._background_tasks) == 0
```

**Key Principles**:
- Test real component interactions, not just mocks
- Verify task lifecycle management (start/stop)
- Test integration points between components
- Include proper cleanup in tests

## Helper Functions and Fixtures

### Context Builder Pattern

```python
def _make_context(
    *,
    dry_run: bool = False,
    advanced: bool = False,
    symbols: tuple[str, ...] = ("BTC-PERP",),
    broker=None,
    risk_manager=None,
) -> CoordinatorContext:
    """Create a test context with proper mocking and configurable options."""
    config = BotConfig(profile=Profile.PROD, dry_run=dry_run)
    runtime_state = PerpsBotRuntimeState(list(symbols))

    # Default mocks with override capability
    if broker is None:
        broker = Mock()
    if risk_manager is None:
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = advanced

    return CoordinatorContext(
        config=config,
        registry=ServiceRegistry(config=config, broker=broker, risk_manager=risk_manager),
        event_store=Mock(),
        orders_store=Mock(),
        broker=broker,
        risk_manager=risk_manager,
        symbols=symbols,
        bot_id="test_bot",
        runtime_state=runtime_state,
    )
```

### Test Data Creation Patterns

```python
def _create_test_order(
    *,
    id: str = "test-order-1",
    symbol: str = "BTC-PERP",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: Decimal = Decimal("1.0"),
    price: Decimal | None = None,
    status: OrderStatus = OrderStatus.SUBMITTED,
) -> Order:
    """Create a test order with proper constructor parameters."""
    now = datetime.now(UTC)
    return Order(
        id=id,
        client_id="test-client",
        symbol=symbol,
        side=side,
        type=order_type,  # Note: 'type' not 'order_type'
        quantity=quantity,
        price=price,
        stop_price=None,
        tif=TimeInForce.GTC,
        status=status,
        filled_quantity=Decimal("0"),
        avg_fill_price=None,
        submitted_at=now,
        updated_at=now,
    )

def _create_test_product(
    *,
    symbol: str = "BTC-PERP",
    base_currency: str = "BTC",
    quote_currency: str = "USD",
    min_size: Decimal = Decimal("0.001"),
    step_size: Decimal = Decimal("0.001"),
) -> Product:
    """Create a test product with correct constructor parameters."""
    return Product(
        symbol=symbol,
        base_asset=base_currency,  # Note: 'base_asset' not 'base_currency'
        quote_asset=quote_currency,  # Note: 'quote_asset' not 'quote_currency'
        market_type=MarketType.SPOT,
        min_size=min_size,
        step_size=step_size,
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
    )
```

## Error Handling Testing Strategy

### Production Semantics Alignment

**DO**: Test graceful error handling with logging
```python
# ✅ Correct - matches production behavior
result = await coordinator.execute_decision("BTC-PERP", decision, invalid_mark, product, None)
assert result is None  # Method returns None on error
```

**DON'T**: Expect exceptions when implementation logs errors
```python
# ❌ Incorrect - production doesn't raise this exception
with pytest.raises(AssertionError, match="Invalid mark"):
    await coordinator.execute_decision("BTC-PERP", decision, invalid_mark, product, None)
```

### Exception Re-raising Scenarios

**DO**: Test exceptions that are intentionally re-raised after logging
```python
# ✅ Correct - ExecutionError is re-raised after logging
with pytest.raises(ExecutionError, match="Placement failed"):
    await coordinator.place_order(exec_engine, symbol="BTC-PERP")
```

## Test Organization Structure

### Module Organization
```
tests/unit/bot_v2/orchestration/
├── coordinators/
│   ├── test_execution.py           # Original tests
│   ├── test_execution_enhanced.py  # Enhanced comprehensive tests (our work)
│   ├── test_telemetry.py          # Enhanced telemetry tests
│   └── test_runtime.py            # Runtime coordinator tests
├── order_reconciler/
│   ├── test_fetch_orders.py       # Enhanced with fallback scenarios
│   ├── test_reconcile_flow.py     # Enhanced with error handling
│   └── test_snapshot_positions.py # Enhanced with edge cases
└── configuration/
    └── test_core.py               # Configuration core tests
```

### Test Class Organization
```python
class TestExecutionCoordinatorBackgroundTasks:
    """Test background task lifecycle management and resilience."""

class TestExecutionCoordinatorConfiguration:
    """Test configuration-driven behavior changes and component management."""

class TestExecutionCoordinatorErrorResilience:
    """Test error handling and recovery mechanisms."""

class TestExecutionCoordinatorIntegration:
    """Test end-to-end workflows and component interactions."""
```

## Best Practices Summary

### Test Design Principles
1. **Match Production Semantics**: Tests should reflect actual system behavior
2. **Use Proper Async Patterns**: Handle async operations correctly with cancellation
3. **Respect Immutability**: Use proper context update methods
4. **Comprehensive Coverage**: Test success, error, and edge cases
5. **Clean Resource Management**: Always clean up background tasks and resources

### API Compatibility
1. **Verify Constructor Parameters**: Use correct parameter names (e.g., `base_asset` vs `base_currency`)
2. **Check Method Names**: Use correct method names (e.g., `place_order` vs `place`)
3. **Test Type Compatibility**: Ensure test data matches expected types

### Error Handling Testing
1. **Log-and-Continue**: Test graceful handling when errors are logged
2. **Exception Re-raising**: Test when exceptions are intentionally re-raised
3. **Resource Cleanup**: Test proper cleanup after errors
4. **State Recovery**: Test system state after error handling

This architecture provides a solid foundation for future testing work and ensures consistency across the orchestration test suite.