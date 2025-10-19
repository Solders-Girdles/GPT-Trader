"""
ExecutionCoordinator tests for the coordinator implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def base_context() -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], dry_run=False)
    config.time_in_force = "GTC"
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

    broker = Mock()
    risk_manager = Mock()
    risk_manager.config = Mock(
        enable_dynamic_position_sizing=False, enable_market_impact_guard=False
    )
    risk_manager.set_impact_estimator = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=event_store,
        orders_store=orders_store,
    )

    controller = Mock()
    controller.is_reduce_only_mode.return_value = False
    controller.sync_with_risk_manager = Mock()

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="perps_bot",
        runtime_state=runtime_state,
        config_controller=controller,
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def coordinator(base_context: CoordinatorContext) -> ExecutionCoordinator:
    return ExecutionCoordinator(base_context)


@pytest.fixture
def test_product() -> Product:
    product = Mock(spec=Product)
    product.symbol = "BTC-PERP"
    product.market_type = MarketType.PERPETUAL
    product.base_increment = Decimal("0.0001")
    product.quote_increment = Decimal("0.01")
    return product


@pytest.fixture
def test_order() -> Order:
    now = datetime.now(timezone.utc)
    return Order(
        id="order-1",
        client_id="client-1",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        price=None,
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        avg_fill_price=Decimal("50000"),
        submitted_at=now,
        updated_at=now,
    )


def test_initialize_builds_execution_engine(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    updated = coordinator.initialize(base_context)

    assert updated.runtime_state.exec_engine is not None
    assert "execution_engine" in updated.registry.extras


def test_should_use_advanced_flag() -> None:
    config = Mock(enable_dynamic_position_sizing=True, enable_market_impact_guard=False)
    assert ExecutionCoordinator._should_use_advanced(config) is True
    config.enable_dynamic_position_sizing = False
    config.enable_market_impact_guard = False
    assert ExecutionCoordinator._should_use_advanced(config) is False


@pytest.mark.asyncio
async def test_ensure_order_lock_creates_lock(coordinator: ExecutionCoordinator) -> None:
    runtime_state = coordinator.context.runtime_state
    assert runtime_state.order_lock is None

    lock = coordinator.ensure_order_lock()

    assert lock is coordinator.context.runtime_state.order_lock


def test_get_order_reconciler_caches_instance(coordinator: ExecutionCoordinator) -> None:
    first = coordinator._get_order_reconciler()
    second = coordinator._get_order_reconciler()

    assert first is second


def test_reset_order_reconciler(coordinator: ExecutionCoordinator) -> None:
    coordinator._get_order_reconciler()
    assert coordinator._order_reconciler is not None

    coordinator.reset_order_reconciler()

    assert coordinator._order_reconciler is None


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

    result = await coordinator.place_order(exec_engine, symbol="BTC-PERP")

    assert result is None
    assert coordinator.context.runtime_state.order_stats["failed"] == 1


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
async def test_run_order_reconciliation_cycle(
    coordinator: ExecutionCoordinator,
) -> None:
    reconciler = Mock()
    reconciler.fetch_local_open_orders.return_value = {}
    reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
    reconciler.diff_orders.return_value = Mock(missing_on_exchange=[], missing_locally=[])
    reconciler.reconcile_missing_on_exchange = AsyncMock()
    reconciler.reconcile_missing_locally = Mock()
    reconciler.record_snapshot = AsyncMock()
    reconciler.snapshot_positions = AsyncMock(return_value={})

    await coordinator._run_order_reconciliation_cycle(reconciler)

    reconciler.record_snapshot.assert_called_once()
