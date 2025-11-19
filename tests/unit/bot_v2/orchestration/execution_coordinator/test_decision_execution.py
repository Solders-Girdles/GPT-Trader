import pytest
from unittest.mock import Mock, patch
from decimal import Decimal
from types import SimpleNamespace
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.features.brokerages.core.interfaces import OrderType, TimeInForce, OrderSide
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.features.brokerages.core.interfaces import Product
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

@pytest.mark.asyncio
async def test_execute_decision_skips_in_dry_run(
    coordinator: ExecutionCoordinator,
    test_product: Product,
) -> None:
    coordinator.context.config.dry_run = True
    coordinator.context.runtime_state.exec_engine = Mock()

    decision = SimpleNamespace(
        action=Action.BUY,
        quantity=Decimal("0.1"),
        reduce_only=False,
        leverage=None,
        target_notional=Decimal("0"),
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_trigger=None,
        time_in_force=TimeInForce.GTC,
    )

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=test_product,
        position_state=None,
    )

    coordinator.context.runtime_state.exec_engine.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_execute_decision_invokes_engine(
    coordinator: ExecutionCoordinator,
    test_product: Product,
) -> None:
    runtime_state = coordinator.context.runtime_state
    exec_engine = Mock()
    runtime_state.exec_engine = exec_engine
    decision = SimpleNamespace(
        action=Action.BUY,
        quantity=Decimal("0.1"),
        reduce_only=False,
        leverage=None,
        target_notional=Decimal("0"),
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_trigger=None,
        time_in_force=TimeInForce.GTC,
    )

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=test_product,
        position_state={"quantity": Decimal("0")},
    )

    exec_engine.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_execute_decision_handles_missing_product(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision handles missing product gracefully."""
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = Mock()
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.BUY, quantity=Decimal("0.1"))

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=None,  # Missing product
        position_state=None,
    )

    # Should not place order due to missing product
    base_context.runtime_state.exec_engine.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_execute_decision_handles_invalid_mark(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision handles invalid mark price."""
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = Mock()
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.BUY, quantity=Decimal("0.1"))
    product = ScenarioBuilder.create_product()

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("0"),  # Invalid mark
        product=product,
        position_state=None,
    )

    # Should not place order due to invalid mark
    base_context.runtime_state.exec_engine.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_execute_decision_handles_close_without_position(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision skips close when no position exists."""
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = Mock()
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.CLOSE, quantity=Decimal("0.1"))

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=ScenarioBuilder.create_product(),
        position_state=None,  # No position
    )

    # Should not place order due to no position to close
    base_context.runtime_state.exec_engine.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_execute_decision_handles_execution_exception(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision handles execution exceptions."""
    exec_engine = Mock()
    exec_engine.place_order = Mock(side_effect=Exception("execution_failed"))
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.BUY, quantity=Decimal("0.1"))
    product = ScenarioBuilder.create_product()

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=product,
        position_state=None,
    )

    # Should have attempted to place order despite failure
    exec_engine.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_execute_decision_respects_reduce_only_global(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision respects global reduce-only mode."""
    exec_engine = Mock()
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    config_controller = Mock()
    config_controller.is_reduce_only_mode = Mock(return_value=True)
    base_context = base_context.with_updates(
        runtime_state=runtime_state, config_controller=config_controller
    )
    coordinator.update_context(base_context)

    decision = SimpleNamespace(
        action=Action.BUY,
        quantity=Decimal("0.1"),
        reduce_only=False,  # Decision says not reduce-only
    )
    product = ScenarioBuilder.create_product()

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=product,
        position_state=None,
    )

    # Should have set reduce_only=True due to global mode
    call_kwargs = exec_engine.place_order.call_args[1]
    assert call_kwargs["reduce_only"] is True


@pytest.mark.asyncio
async def test_execute_decision_handles_close_position_side_detection(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision correctly detects position side for close orders."""
    exec_engine = Mock()
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.CLOSE)
    product = ScenarioBuilder.create_product()
    position_state = {"quantity": Decimal("0.5"), "side": "long"}

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=product,
        position_state=position_state,
    )

    # Should sell to close long position
    call_kwargs = exec_engine.place_order.call_args[1]
    assert call_kwargs["side"] == OrderSide.SELL


@pytest.mark.asyncio
async def test_execute_decision_handles_leverage_override(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test execute_decision handles leverage override in decision."""
    exec_engine = Mock()
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(
        action=Action.BUY, quantity=Decimal("0.1"), leverage=Decimal("2.0")
    )
    product = ScenarioBuilder.create_product()

    await coordinator.execute_decision(
        symbol="BTC-PERP",
        decision=decision,
        mark=Decimal("50000"),
        product=product,
        position_state=None,
    )

    # Should pass leverage to execution engine
    call_kwargs = exec_engine.place_order.call_args[1]
    assert call_kwargs["leverage"] == Decimal("2.0")


@pytest.mark.asyncio
async def test_execute_decision_missing_runtime_state_logs_and_returns(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test execute_decision handles missing runtime state gracefully."""
    base_context = base_context.with_updates(runtime_state=None)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.BUY, quantity=Decimal("0.1"))
    product = ScenarioBuilder.create_product()

    with patch("bot_v2.orchestration.coordinators.execution.orders.execution.workflow.logger") as mock_logger:
        await coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=product,
            position_state=None,
        )

        # Should log debug message and return early
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[1]
        assert call_args["operation"] == "execution_decision"
        assert call_args["stage"] == "runtime_state"


@pytest.mark.asyncio
async def test_execute_decision_missing_position_quantity_logs_error(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test execute_decision handles position state missing quantity."""
    exec_engine = Mock()
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.CLOSE, quantity=Decimal("0.1"))
    product = ScenarioBuilder.create_product()
    position_state = {"some_other_field": "value"}  # Missing 'quantity' key

    with patch("bot_v2.orchestration.coordinators.execution.orders.execution.workflow.logger") as mock_logger:
        await coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=product,
            position_state=position_state,
        )

        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[1]
        assert "Position state missing quantity" in str(call_args) or "Position state missing quantity" in str(mock_logger.error.call_args)


@pytest.mark.asyncio
async def test_execute_decision_close_without_position_logs_and_returns(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test execute_decision handles close action with no position."""
    exec_engine = Mock()
    runtime_state = base_context.runtime_state
    runtime_state.exec_engine = exec_engine
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    decision = SimpleNamespace(action=Action.CLOSE, quantity=Decimal("0.1"))
    product = ScenarioBuilder.create_product()

    with patch("bot_v2.orchestration.coordinators.execution.orders.execution.workflow.logger") as mock_logger:
        await coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=product,
            position_state=None,  # No position
        )

        # Should log warning about no position to close
        mock_logger.warning.assert_called()
        args, _ = mock_logger.warning.call_args
        assert "no position" in args[0].lower()
