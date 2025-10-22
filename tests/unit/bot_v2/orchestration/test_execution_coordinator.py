"""
Enhanced ExecutionCoordinator tests for the coordinator implementation.

Tests execution engine initialization, order placement flows, reconciliation,
async coordination, failure handling, and guard trigger scenarios.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

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

BOT_ID = "coinbase_trader"


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
        bot_id=BOT_ID,
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


class TestExecutionCoordinatorAsyncFlows:
    """Test async execution flows and coordination."""

    @pytest.mark.asyncio
    async def test_start_background_tasks_initializes_reconciliation_and_guards(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test start_background_tasks starts reconciliation and guard loops."""
        base_context = base_context.with_updates(
            config=base_context.config.with_overrides(dry_run=False),
            runtime_state=PerpsBotRuntimeState(["BTC-PERP"]),
        )
        coordinator.update_context(base_context)

        tasks = await coordinator.start_background_tasks()

        assert len(tasks) == 2
        # Tasks should be running (not done)
        assert not any(task.done() for task in tasks)

        # Cancel tasks to clean up
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_background_tasks_skips_in_dry_run(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test start_background_tasks skips in dry run mode."""
        base_context = base_context.with_updates(
            config=base_context.config.with_overrides(dry_run=True)
        )
        coordinator.update_context(base_context)

        tasks = await coordinator.start_background_tasks()

        assert tasks == []

    @pytest.mark.asyncio
    async def test_run_runtime_guards_loop_handles_exceptions(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _run_runtime_guards_loop handles exceptions gracefully."""
        exec_engine = Mock()
        exec_engine.run_runtime_guards = Mock(side_effect=Exception("guard_error"))
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = exec_engine
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        # Run for a short time then cancel
        import asyncio

        task = asyncio.create_task(coordinator._run_runtime_guards_loop())
        await asyncio.sleep(0.1)  # Let it run one iteration
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        # Should have called run_runtime_guards despite exception
        base_context.runtime_state.exec_engine.run_runtime_guards.assert_called()

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_loop_with_custom_interval(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _run_order_reconciliation_loop respects custom interval."""
        # Mock reconciler to avoid actual reconciliation
        coordinator._get_order_reconciler = Mock(return_value=Mock())
        coordinator._run_order_reconciliation_cycle = AsyncMock()

        import asyncio

        task = asyncio.create_task(coordinator._run_order_reconciliation_loop(interval_seconds=1))
        await asyncio.sleep(2.1)  # Should run ~2 cycles
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        # Should have run multiple cycles
        assert coordinator._run_order_reconciliation_cycle.call_count >= 2


class TestExecutionCoordinatorFailureHandling:
    """Test failure handling and error recovery."""

    @pytest.mark.asyncio
    async def test_execute_decision_handles_missing_product(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision handles missing product gracefully."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision handles invalid mark price."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision skips close when no position exists."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision handles execution exceptions."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=Exception("execution_failed"))
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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
    async def test_place_order_handles_order_validation_failure(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test place_order handles order validation failures."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=ValidationError("invalid_order"))
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = exec_engine
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        with pytest.raises(ValidationError):
            result = await coordinator.place_order(exec_engine, symbol="BTC-PERP")

        assert base_context.runtime_state.order_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_place_order_handles_unexpected_errors(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test place_order handles unexpected errors."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=RuntimeError("unexpected"))
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = exec_engine
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        result = await coordinator.place_order(exec_engine, symbol="BTC-PERP")

        assert result is None
        assert base_context.runtime_state.order_stats["failed"] == 1


class TestExecutionCoordinatorGuardTriggers:
    """Test guard trigger handling and blocking."""

    @pytest.mark.asyncio
    async def test_execute_decision_respects_reduce_only_global(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision respects global reduce-only mode."""
        exec_engine = Mock()
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision correctly detects position side for close orders."""
        exec_engine = Mock()
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test execute_decision handles leverage override in decision."""
        exec_engine = Mock()
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
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


class TestExecutionCoordinatorAdvancedEngine:
    """Test advanced execution engine integration."""

    def test_initialize_uses_advanced_engine_for_dynamic_sizing(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test initialize uses advanced engine when dynamic sizing enabled."""
        risk_manager = Mock()
        risk_manager.config = Mock(enable_dynamic_position_sizing=True)
        base_context = base_context.with_updates(risk_manager=risk_manager)
        coordinator.update_context(base_context)

        result = coordinator.initialize(base_context)

        # Should have initialized AdvancedExecutionEngine
        assert isinstance(result.runtime_state.exec_engine, AdvancedExecutionEngine)

    def test_initialize_uses_advanced_engine_for_market_impact_guard(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test initialize uses advanced engine when market impact guard enabled."""
        risk_manager = Mock()
        risk_manager.config = Mock(enable_market_impact_guard=True)
        base_context = base_context.with_updates(risk_manager=risk_manager)
        coordinator.update_context(base_context)

        result = coordinator.initialize(base_context)

        # Should have initialized AdvancedExecutionEngine
        assert isinstance(result.runtime_state.exec_engine, AdvancedExecutionEngine)

    def test_build_impact_estimator_creates_estimator_function(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _build_impact_estimator creates impact estimator function."""
        broker = Mock()
        broker.get_quote = Mock(
            return_value=Mock(last=Decimal("50000"), bid=Decimal("49900"), ask=Decimal("50100"))
        )
        broker.order_books = {
            "BTC-PERP": ([(Decimal("49900"), Decimal("100"))], [(Decimal("50100"), Decimal("100"))])
        }
        base_context = base_context.with_updates(broker=broker)
        coordinator.update_context(base_context)

        estimator = coordinator._build_impact_estimator(base_context)

        # Should be callable
        assert callable(estimator)

        # Test with mock request

        from bot_v2.utilities import utc_now

        req = SimpleNamespace(
            symbol="BTC-PERP", quantity=Decimal("0.1"), side=OrderSide.BUY, timestamp=utc_now()
        )
        result = estimator(req)

        # Should return impact estimate
        from bot_v2.features.live_trade.liquidity_service import ImpactEstimate

        assert isinstance(result, ImpactEstimate)


class TestExecutionCoordinatorOrderLock:
    """Test order lock management."""

    def test_ensure_order_lock_creates_lock_when_missing(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test ensure_order_lock creates lock when missing."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.order_lock = None
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        lock = coordinator.ensure_order_lock()

        assert lock is not None
        assert base_context.runtime_state.order_lock is lock

    def test_ensure_order_lock_returns_existing_lock(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test ensure_order_lock returns existing lock."""
        existing_lock = Mock()
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.order_lock = existing_lock
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        lock = coordinator.ensure_order_lock()

        assert lock is existing_lock

    def test_ensure_order_lock_handles_asyncio_error(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test ensure_order_lock handles asyncio initialization errors."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.order_lock = None
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        # Mock asyncio.Lock to raise RuntimeError
        # import asyncio  # Not needed at module level

        original_lock = asyncio.Lock
        asyncio.Lock = Mock(side_effect=RuntimeError("no event loop"))

        try:
            with pytest.raises(RuntimeError):
                coordinator.ensure_order_lock()
        finally:
            asyncio.Lock = original_lock


class TestExecutionCoordinatorHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_unhealthy_without_engine(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test health_check returns unhealthy when no execution engine."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = None
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        status = coordinator.health_check()

        assert status.healthy is False
        assert status.component == "execution"
        assert status.details["has_execution_engine"] is False

    def test_health_check_returns_healthy_with_engine(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test health_check returns healthy when execution engine present."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = Mock()
        runtime_state.order_stats = {"attempted": 5, "successful": 4, "failed": 1}
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        status = coordinator.health_check()

        assert status.healthy is True
        assert status.details["has_execution_engine"] is True
        assert status.details["order_stats"]["attempted"] == 5


class TestExecutionCoordinatorCoverageEnhancements:
    """Coverage enhancement tests for missing critical paths."""

    def test_initialize_handles_runtime_settings_not_runtime_settings(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test initialize handles non-RuntimeSettings in registry."""
        from bot_v2.orchestration.runtime_settings import RuntimeSettings

        # Create registry with non-RuntimeSettings object
        registry = Mock()
        registry.runtime_settings = Mock()  # Not RuntimeSettings instance
        registry.extras = {}
        base_context = base_context.with_updates(registry=registry)

        with patch(
            "bot_v2.orchestration.coordinators.execution.load_runtime_settings",
            return_value=RuntimeSettings(),
        ):
            result = coordinator.initialize(base_context)

            # Should have updated registry with proper RuntimeSettings
            assert isinstance(result.registry.runtime_settings, RuntimeSettings)

    @pytest.mark.asyncio
    async def test_execute_decision_missing_runtime_state_logs_and_returns(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test execute_decision handles missing runtime state gracefully."""
        base_context = base_context.with_updates(runtime_state=None)
        coordinator.update_context(base_context)

        decision = SimpleNamespace(action=Action.BUY, quantity=Decimal("0.1"))
        product = ScenarioBuilder.create_product()

        with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
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
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test execute_decision handles position state missing quantity."""
        exec_engine = Mock()
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = exec_engine
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        decision = SimpleNamespace(action=Action.CLOSE, quantity=Decimal("0.1"))
        product = ScenarioBuilder.create_product()
        position_state = {"some_other_field": "value"}  # Missing 'quantity' key

        with pytest.raises(AssertionError, match="Position state missing quantity"):
            await coordinator.execute_decision(
                symbol="BTC-PERP",
                decision=decision,
                mark=Decimal("50000"),
                product=product,
                position_state=position_state,
            )

    @pytest.mark.asyncio
    async def test_execute_decision_close_without_position_logs_and_returns(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test execute_decision handles close action with no position."""
        exec_engine = Mock()
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = exec_engine
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        decision = SimpleNamespace(action=Action.CLOSE, quantity=Decimal("0.1"))
        product = ScenarioBuilder.create_product()

        with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
            await coordinator.execute_decision(
                symbol="BTC-PERP",
                decision=decision,
                mark=Decimal("50000"),
                product=product,
                position_state=None,  # No position
            )

            # Should log debug message about no position to close
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args[1]
            assert "no position" in call_args["message"].lower()

    @pytest.mark.asyncio
    async def test_place_order_handles_runtime_state_unavailable(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test place_order handles unavailable runtime state."""
        base_context = base_context.with_updates(runtime_state=None)
        coordinator.update_context(base_context)

        with pytest.raises(RuntimeError, match="Runtime state is unavailable"):
            await coordinator.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

    @pytest.mark.asyncio
    async def test_place_order_handles_lock_initialization_error(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test place_order handles order lock initialization error."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.order_lock = None
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        # Mock asyncio.Lock to raise RuntimeError
        original_lock = asyncio.Lock
        asyncio.Lock = Mock(side_effect=RuntimeError("no event loop"))

        try:
            with pytest.raises(RuntimeError, match="no event loop"):
                await coordinator.place_order(
                    symbol="BTC-PERP",
                    side=OrderSide.BUY,
                    quantity=Decimal("0.1"),
                    order_type=OrderType.MARKET,
                )
        finally:
            asyncio.Lock = original_lock

    @pytest.mark.asyncio
    async def test_place_order_handles_execution_engine_not_initialized(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test place_order handles execution engine not initialized."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.order_lock = asyncio.Lock()
        runtime_state.exec_engine = None  # Not initialized
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        with pytest.raises(ExecutionError, match="Execution engine not initialized"):
            await coordinator.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.MARKET,
            )

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_loop_handles_custom_interval(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test reconciliation loop respects custom interval setting."""
        from bot_v2.orchestration.runtime_settings import RuntimeSettings

        runtime_settings = RuntimeSettings(raw_env={"EXECUTION_RECONCILIATION_INTERVAL": "2.5"})
        registry = Mock()
        registry.runtime_settings = runtime_settings
        registry.extras = {}
        base_context = base_context.with_updates(
            registry=registry, runtime_state=PerpsBotRuntimeState(["BTC-PERP"])
        )
        coordinator.update_context(base_context)

        # Mock _run_order_reconciliation_cycle to track call timing
        call_times = []

        async def mock_cycle(reconciler):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)  # Small delay

        with patch.object(coordinator, "_run_order_reconciliation_cycle", side_effect=mock_cycle):
            # Mock _ensure_order_reconciler to return Mock
            with patch.object(coordinator, "_ensure_order_reconciler", return_value=Mock()):
                # Cancel after a short time
                task = asyncio.create_task(coordinator._run_order_reconciliation_loop())
                await asyncio.sleep(0.1)  # Let it run a bit
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have been called
                assert len(call_times) >= 1

    @pytest.mark.asyncio
    async def test_run_runtime_guards_loop_handles_continuously(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test runtime guards loop runs continuously."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.exec_engine = Mock()
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        call_count = 0

        async def mock_collect_and_run_guards():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:  # Stop after 3 calls
                raise asyncio.CancelledError()
            await asyncio.sleep(0.01)

        with patch(
            "bot_v2.orchestration.coordinators.execution.collect_and_run_guards",
            side_effect=mock_collect_and_run_guards,
        ):
            # Should run continuously until cancelled
            task = asyncio.create_task(coordinator._run_runtime_guards_loop())

            try:
                await asyncio.wait_for(task, timeout=0.1)
            except asyncio.CancelledError:
                pass

            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_loop_handles_reconciler_none(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test reconciliation loop handles None reconciler gracefully."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        # Mock _ensure_order_reconciler to return None
        with patch.object(coordinator, "_ensure_order_reconciler", return_value=None):
            with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
                await coordinator._run_order_reconciliation_loop()

                # Should log warning about missing reconciler
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[1]
                assert "reconciler unavailable" in call_args["message"].lower()

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_cycle_handles_missing_runtime_state(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test reconciliation cycle handles missing runtime state."""
        base_context = base_context.with_updates(runtime_state=None)
        coordinator.update_context(base_context)

        reconciler = Mock()

        with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
            await coordinator._run_order_reconciliation_cycle(reconciler)

            # Should log warning about missing runtime state
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args[1]
            assert "runtime state missing" in call_args["message"].lower()

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_cycle_handles_runtime_state_exception(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test reconciliation cycle handles runtime state exceptions."""
        runtime_state = Mock()
        runtime_state.order_stats = {"failed": 1}

        # Mock accessing runtime_state.exec_broker to raise exception
        def raise_attribute_error():
            raise AttributeError("Test error")

        runtime_state.__getattr__ = raise_attribute_error

        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        reconciler = Mock()

        with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
            await coordinator._run_order_reconciliation_cycle(reconciler)

            # Should log error but continue
            mock_logger.error.assert_called()
            call_args = mock_logger.error.call_args[1]
            assert "reconciliation cycle error" in call_args["operation"]

    def test_should_use_advanced_checks_dynamic_sizing(self, coordinator: ExecutionCoordinator):
        """Test _should_use_advanced checks dynamic sizing flag."""
        risk_config = Mock()
        risk_config.enable_dynamic_position_sizing = True

        result = coordinator._should_use_advanced(risk_config)

        assert result is True

    def test_should_use_advanced_checks_market_impact_guard(
        self, coordinator: ExecutionCoordinator
    ):
        """Test _should_use_advanced checks market impact guard flag."""
        risk_config = Mock()
        risk_config.enable_dynamic_position_sizing = False
        risk_config.enable_market_impact_guard = True

        result = coordinator._should_use_advanced(risk_config)

        assert result is True

    def test_should_use_advanced_handles_none_risk_config(self, coordinator: ExecutionCoordinator):
        """Test _should_use_advanced handles None risk config."""
        result = coordinator._should_use_advanced(None)

        assert result is False

    def test_should_use_advanced_handles_missing_attributes(
        self, coordinator: ExecutionCoordinator
    ):
        """Test _should_use_advanced handles config with missing attributes."""
        risk_config = Mock(spec=[])  # No attributes

        result = coordinator._should_use_advanced(risk_config)

        assert result is False

    @pytest.mark.asyncio
    async def test_collect_and_publish_metrics_handles_execution_stats(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test collect_and_publish_metrics processes execution stats correctly."""
        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
        runtime_state.order_stats = {"attempted": 10, "successful": 8, "failed": 2}
        base_context = base_context.with_updates(runtime_state=runtime_state)
        coordinator.update_context(base_context)

        with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
            await coordinator.collect_and_publish_metrics()

            # Should log metrics
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[1]
            assert "execution_metrics" in call_args["operation"]

    @pytest.mark.asyncio
    async def test_collect_and_publish_metrics_handles_no_runtime_state(
        self, coordinator: ExecutionCoordinator, base_context: CoordinatorContext
    ):
        """Test collect_and_publish_metrics handles missing runtime state."""
        base_context = base_context.with_updates(runtime_state=None)
        coordinator.update_context(base_context)

        with patch("bot_v2.orchestration.coordinators.execution.logger") as mock_logger:
            await coordinator.collect_and_publish_metrics()

            # Should log debug message
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args[1]
            assert "runtime state missing" in call_args["message"].lower()
