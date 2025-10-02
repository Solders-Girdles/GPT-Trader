# Execution Coordinator Refactoring Summary

## Overview

This refactoring extracted execution coordination logic into three focused collaborators:
- **ExecutionEngineFactory**: Engine creation and configuration
- **OrderPlacementService**: Decision-to-order translation and placement
- **ExecutionRuntimeSupervisor**: Background runtime tasks

## Changes

### New Components

#### `src/bot_v2/orchestration/execution/engine_factory.py`

**Responsibilities:**
- Parse `SLIPPAGE_MULTIPLIERS` environment variable
- Determine engine type based on risk config flags
- Create and configure `LiquidityService` + impact estimator
- Instantiate `AdvancedExecutionEngine` or `LiveExecutionEngine`

**Key Methods:**
- `parse_slippage_multipliers()` → `dict[str, float]`
- `should_use_advanced_engine(risk_manager)` → `bool`
- `create_impact_estimator(broker, risk_manager)` → closure
- `create_engine(broker, risk_manager, event_store, bot_id, enable_preview)` → engine

#### `src/bot_v2/orchestration/execution/order_placement.py`

**Responsibilities:**
- Translate `Decision` DTO to order parameters
- Handle `AdvancedExecutionEngine` vs `LiveExecutionEngine` param differences
- Manage async order lock
- Track order statistics (attempted/successful/failed)
- Update orders store

**Key Methods:**
- `execute_decision(symbol, decision, mark, product, position_state, exec_engine, ...)`
- `_build_place_kwargs(...)` → `dict[str, Any]` (handles engine-specific params)
- `_place_order(exec_engine, **kwargs)` → `Order | None`
- `_place_order_inner(exec_engine, **kwargs)` → `Order | None`

#### `src/bot_v2/orchestration/execution/runtime_supervisor.py`

**Responsibilities:**
- Run runtime guards background loop
- Run order reconciliation background loop
- Manage `OrderReconciler` lifecycle

**Key Methods:**
- `run_runtime_guards(running_flag)`
- `run_order_reconciliation(running_flag, interval_seconds)`
- `reset_order_reconciler()`

### Updated Components

#### `src/bot_v2/orchestration/execution_coordinator.py`

**Changes:**
- `init_execution()` now delegates to `ExecutionEngineFactory.create_engine()`
- `execute_decision()` now delegates to `OrderPlacementService.execute_decision()`
- `_place_order*()` methods delegate to `OrderPlacementService`
- `run_runtime_guards()` delegates to `ExecutionRuntimeSupervisor`
- `run_order_reconciliation()` delegates to `ExecutionRuntimeSupervisor`

**Backward Compatibility:**
- All existing public APIs preserved
- Delegation ensures existing callers continue to work
- Private methods maintained for internal callers (PerpsBot)

## Data Flow Changes

### Before

```
ExecutionCoordinator.init_execution()
  → Parse SLIPPAGE_MULTIPLIERS env
  → Check risk_manager.config flags
  → Create LiquidityService + impact estimator closure
  → Instantiate AdvancedExecutionEngine or LiveExecutionEngine
  → Register in bot.registry
```

### After

```
ExecutionCoordinator.init_execution()
  → ExecutionEngineFactory.create_engine(...)
    → parse_slippage_multipliers()
    → should_use_advanced_engine()
    → create_impact_estimator() if needed
    → return configured engine
  → Register in bot.registry
```

### Before

```
ExecutionCoordinator.execute_decision(...)
  → Validate inputs
  → Calculate order quantity
  → Determine side, reduce_only
  → Extract order params (type, prices, tif)
  → Build kwargs for specific engine type
  → Call _place_order(...)
```

### After

```
ExecutionCoordinator.execute_decision(...)
  → OrderPlacementService.execute_decision(...)
    → All validation and translation logic
    → _place_order(...)
```

## Benefits

1. **Single Responsibility**: Each component has one clear purpose
2. **Testability**: Can test env parsing, engine selection, decision translation independently
3. **Maintainability**: Changes to engine creation don't affect order placement logic
4. **Dependency Injection**: Services can be mocked/swapped for testing
5. **Clearer Boundaries**: Easier to understand what each piece does

## Testing

### New Tests

- `tests/unit/bot_v2/orchestration/execution/test_engine_factory.py`
  - Tests for slippage parsing
  - Tests for engine selection logic
  - Tests for engine creation

### Updated Tests

- `tests/unit/bot_v2/orchestration/test_execution_coordinator.py`
  - Updated to work with delegation pattern
  - Tests now verify services are called correctly
  - Integration tests still pass (verifies end-to-end behavior)

## Migration Guide

### For code that creates engines

**Before:**
```python
# Manual engine creation with env parsing
coordinator.init_execution()
```

**After:**
```python
# Still works! Delegates to factory
coordinator.init_execution()

# Or use factory directly
engine = ExecutionEngineFactory.create_engine(
    broker=broker,
    risk_manager=risk_manager,
    event_store=event_store,
    bot_id="my_bot",
    enable_preview=False,
)
```

### For code that places orders

**Before:**
```python
await coordinator.execute_decision(
    symbol="BTC-USD",
    decision=decision,
    mark=mark,
    product=product,
    position_state=position,
)
```

**After:**
```python
# Still works! Delegates to service
await coordinator.execute_decision(
    symbol="BTC-USD",
    decision=decision,
    mark=mark,
    product=product,
    position_state=position,
)

# Or use service directly
service = OrderPlacementService(
    orders_store=orders_store,
    order_stats=order_stats,
    broker=broker,
    dry_run=False,
)
await service.execute_decision(...)
```

## Future Work

1. **Remove delegation layer**: Once callers are migrated, remove ExecutionCoordinator shims
2. **More granular services**: Could split OrderPlacementService further
3. **Configuration object**: Instead of many params, pass config objects
4. **Event-driven coordination**: Use events instead of direct method calls

## Files Changed

### Created

- `src/bot_v2/orchestration/execution/engine_factory.py`
- `src/bot_v2/orchestration/execution/order_placement.py`
- `src/bot_v2/orchestration/execution/runtime_supervisor.py`
- `tests/unit/bot_v2/orchestration/execution/test_engine_factory.py`

### Modified

- `src/bot_v2/orchestration/execution/__init__.py` (added exports)
- `src/bot_v2/orchestration/execution_coordinator.py` (delegation)
- `tests/unit/bot_v2/orchestration/test_execution_coordinator.py` (updated assertions)

## Performance

No performance impact - delegation adds negligible overhead (single function call). All heavy logic remains the same, just organized differently.

## Rollback Plan

If issues arise, revert `execution_coordinator.py` to directly implement logic (pre-refactor state). No external API changes means minimal risk.
