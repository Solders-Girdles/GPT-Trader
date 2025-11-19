import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderType

@pytest.mark.asyncio
async def test_place_order_inner_updates_stats(
    coordinator: ExecutionCoordinator,
    test_order: Order,
) -> None:
    runtime_state = coordinator.context.runtime_state
    exec_engine = Mock(spec=AdvancedExecutionEngine)
    exec_engine.place_order.return_value = test_order

    await coordinator.place_order_inner(exec_engine, symbol="BTC-PERP")

    stats = runtime_state.order_stats
    assert stats["attempted"] == 1
    assert stats["successful"] == 1
    coordinator.context.orders_store.upsert.assert_called_once_with(test_order)


@pytest.mark.asyncio
async def test_place_order_inner_fetches_from_broker_for_ids(
    coordinator: ExecutionCoordinator,
    test_order: Order,
) -> None:
    exec_engine = Mock()
    exec_engine.place_order.return_value = "order-id"
    coordinator.context.broker.get_order.return_value = test_order

    result = await coordinator.place_order_inner(exec_engine, symbol="BTC-PERP")

    assert result is test_order
    coordinator.context.broker.get_order.assert_called_once_with("order-id")


@pytest.mark.asyncio
async def test_place_order_handles_validation_errors(coordinator: ExecutionCoordinator) -> None:
    exec_engine = Mock()
    exec_engine.place_order.side_effect = ValidationError("invalid")

    with pytest.raises(ValidationError):
        await coordinator.place_order(exec_engine, symbol="BTC-PERP")

    assert coordinator.context.runtime_state.order_stats["failed"] == 1


@pytest.mark.asyncio
async def test_place_order_handles_execution_errors(coordinator: ExecutionCoordinator) -> None:
    exec_engine = Mock()
    exec_engine.place_order.side_effect = ExecutionError("failed")

    with pytest.raises(ExecutionError):
        await coordinator.place_order(exec_engine, symbol="BTC-PERP")

    assert coordinator.context.runtime_state.order_stats["failed"] == 1


@pytest.mark.asyncio
async def test_place_order_handles_unexpected_errors(coordinator: ExecutionCoordinator) -> None:
    exec_engine = Mock()
    exec_engine.place_order.side_effect = RuntimeError("boom")

    with pytest.raises(ExecutionError, match="boom"):
        await coordinator.place_order(exec_engine, symbol="BTC-PERP")

    assert coordinator.context.runtime_state.order_stats["failed"] == 1


@pytest.mark.asyncio
async def test_place_order_handles_order_validation_failure(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test place_order handles order validation failures."""
    exec_engine = Mock()
    exec_engine.place_order = Mock(side_effect=ValidationError("invalid_order"))
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    with pytest.raises(ValidationError):
        await coordinator.place_order(exec_engine, symbol="BTC-PERP")

    assert base_context.runtime_state.order_stats["failed"] == 1


@pytest.mark.asyncio
async def test_place_order_handles_runtime_state_unavailable(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test place_order handles unavailable runtime state."""
    base_context = base_context.with_updates(runtime_state=None)
    coordinator.update_context(base_context)

    with pytest.raises(RuntimeError, match="Runtime state is unavailable"):
        await coordinator.place_order(
            exec_engine=Mock(),
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )


@pytest.mark.asyncio
async def test_place_order_handles_execution_engine_not_initialized(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test place_order handles execution engine not initialized."""
    import asyncio
    runtime_state = base_context.runtime_state
    runtime_state.order_lock = asyncio.Lock()
    runtime_state.exec_engine = None  # Not initialized
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    with pytest.raises(ExecutionError):
        await coordinator.place_order(
            exec_engine=None,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )
