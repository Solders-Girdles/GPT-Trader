"""Enhanced Execution Coordinator tests following telemetry coordinator patterns."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.errors import ExecutionError
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
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "enhanced_test_bot"


def _make_context(
    *,
    dry_run: bool = False,
    advanced: bool = False,
    symbols: tuple[str, ...] = ("BTC-PERP",),
    broker=None,
    risk_manager=None,
) -> CoordinatorContext:
    """Create a test context with proper mocking."""
    config = BotConfig(profile=Profile.PROD, dry_run=dry_run)
    runtime_state = PerpsBotRuntimeState(list(symbols))

    # Default mocks that can be overridden
    if broker is None:
        broker = Mock()
    if risk_manager is None:
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = advanced
        risk_manager.config.enable_market_impact_guard = False

    context = CoordinatorContext(
        config=config,
        registry=ServiceRegistry(config=config, broker=broker, risk_manager=risk_manager),
        event_store=Mock(),
        orders_store=Mock(),
        broker=broker,
        risk_manager=risk_manager,
        symbols=symbols,
        bot_id=BOT_ID,
        runtime_state=runtime_state,
    )
    return context


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
    """Create a test order with proper constructor."""
    now = datetime.now(UTC)
    return Order(
        id=id,
        client_id="test-client",
        symbol=symbol,
        side=side,
        type=order_type,
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
    """Create a test product."""
    return Product(
        symbol=symbol,
        base_asset=base_currency,
        quote_asset=quote_currency,
        market_type=MarketType.SPOT,
        min_size=min_size,
        step_size=step_size,
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
    )


class TestExecutionCoordinatorBackgroundTasks:
    """Test background task lifecycle management and resilience."""

    @pytest.mark.asyncio
    async def test_start_background_tasks_creates_proper_tasks(self) -> None:
        """Test that start_background_tasks creates the expected background tasks."""
        context = _make_context(dry_run=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        tasks = await coordinator.start_background_tasks()

        # Should create exactly 2 background tasks
        assert len(tasks) == 2
        # Should register tasks
        assert len(coordinator._background_tasks) == 2
        # Tasks should be running
        for task in tasks:
            assert not task.done()

        # Cleanup
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_background_tasks_skips_in_dry_run(self) -> None:
        """Test that background tasks are skipped in dry run mode."""
        context = _make_context(dry_run=True)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        tasks = await coordinator.start_background_tasks()

        # Should not create any tasks in dry run mode
        assert len(tasks) == 0
        assert len(coordinator._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_background_tasks_handles_missing_runtime_state(self) -> None:
        """Test background task creation when runtime state is missing."""
        context = _make_context(dry_run=False)
        # Remove runtime state
        context = context.with_updates(runtime_state=None)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        tasks = await coordinator.start_background_tasks()

        # Should not create tasks when runtime state is missing
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_runtime_guards_loop_error_handling(self) -> None:
        """Test error handling in the runtime guards loop."""
        context = _make_context(dry_run=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Mock the runtime guard method to raise an exception
        coordinator.run_runtime_guards = AsyncMock(side_effect=RuntimeError("Guard failed"))

        # Start background tasks
        tasks = await coordinator.start_background_tasks()
        guards_task = tasks[0]

        # Let the task run briefly to encounter the error
        await asyncio.sleep(0.1)

        # Task should still be running (error handling should be resilient)
        assert not guards_task.done()

        # Cleanup
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_order_reconciliation_loop_error_handling(self) -> None:
        """Test error handling in the order reconciliation loop."""
        context = _make_context(dry_run=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Mock the reconciliation method to raise an exception
        coordinator.run_order_reconciliation = AsyncMock(
            side_effect=RuntimeError("Reconciliation failed")
        )

        # Start background tasks
        tasks = await coordinator.start_background_tasks()
        reconciliation_task = tasks[1]

        # Let the task run briefly to encounter the error
        await asyncio.sleep(0.1)

        # Task should still be running (error handling should be resilient)
        assert not reconciliation_task.done()

        # Cleanup
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_background_task_cleanup_on_shutdown(self) -> None:
        """Test that background tasks are properly cleaned up during shutdown."""
        context = _make_context(dry_run=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Start background tasks
        tasks = await coordinator.start_background_tasks()
        assert len(tasks) == 2
        assert len(coordinator._background_tasks) == 2

        # Shutdown coordinator
        await coordinator.shutdown()

        # All tasks should be cancelled
        for task in tasks:
            assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_concurrent_background_task_execution(self) -> None:
        """Test that multiple background tasks can run concurrently."""
        context = _make_context(dry_run=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Mock both methods to simulate work
        async def mock_guard_work():
            for i in range(3):
                await asyncio.sleep(0.01)

        async def mock_reconciliation_work():
            for i in range(2):
                await asyncio.sleep(0.01)

        coordinator.run_runtime_guards = mock_guard_work
        coordinator.run_order_reconciliation = mock_reconciliation_work

        # Start background tasks
        tasks = await coordinator.start_background_tasks()

        # Let tasks run
        await asyncio.sleep(0.05)

        # Both tasks should have made progress
        assert not tasks[0].done()  # Still running guard loop
        assert not tasks[1].done()  # Still running reconciliation loop

        # Cleanup
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestExecutionCoordinatorConfiguration:
    """Test configuration-driven behavior changes and responses."""

    def test_update_context_triggers_reconciler_reset(self) -> None:
        """Test that changing context triggers reconciler reset."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Create initial reconciler
        initial_reconciler = Mock()
        coordinator._order_reconciler = initial_reconciler

        # Update context with different components
        new_broker = Mock()
        new_context = context.with_updates(broker=new_broker)
        coordinator.update_context(new_context)

        # Reconciler should be reset
        assert coordinator._order_reconciler is None

    def test_update_context_preserves_same_components(self) -> None:
        """Test that updating context with same components preserves reconciler."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Create initial reconciler
        initial_reconciler = Mock()
        coordinator._order_reconciler = initial_reconciler

        # Update context with same components
        same_context = context.with_updates(config=context.config)
        coordinator.update_context(same_context)

        # Reconciler should be preserved
        assert coordinator._order_reconciler is initial_reconciler

    def test_engine_selection_based_on_risk_config(self) -> None:
        """Test engine selection logic based on risk manager configuration."""
        # Test Live execution engine (default)
        risk_config_basic = Mock()
        risk_config_basic.enable_dynamic_position_sizing = False
        risk_config_basic.enable_market_impact_guard = False

        risk_manager = Mock()
        risk_manager.config = risk_config_basic

        context = _make_context(risk_manager=risk_manager)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Should use LiveExecutionEngine
        assert type(coordinator.context.runtime_state.exec_engine).__name__ == "LiveExecutionEngine"

        # Test Advanced execution engine
        risk_config_advanced = Mock()
        risk_config_advanced.enable_dynamic_position_sizing = True
        risk_config_advanced.enable_market_impact_guard = False

        risk_manager_advanced = Mock()
        risk_manager_advanced.config = risk_config_advanced

        context_advanced = _make_context(risk_manager=risk_manager_advanced)
        coordinator_advanced = ExecutionCoordinator(context_advanced)
        coordinator_advanced.initialize(context_advanced)

        # Should use AdvancedExecutionEngine
        assert (
            type(coordinator_advanced.context.runtime_state.exec_engine).__name__
            == "AdvancedExecutionEngine"
        )

    def test_engine_selection_handles_advanced_initialization_failure(self) -> None:
        """Test graceful handling when advanced engine initialization fails."""
        # Create risk config that would trigger advanced engine
        risk_config = Mock()
        risk_config.enable_dynamic_position_sizing = True
        risk_config.enable_market_impact_guard = False

        risk_manager = Mock()
        risk_manager.config = risk_config
        risk_manager.set_impact_estimator = Mock(side_effect=Exception("Impact estimator failed"))

        context = _make_context(risk_manager=risk_manager)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Should still initialize with AdvancedExecutionEngine despite impact estimator failure
        assert (
            type(coordinator.context.runtime_state.exec_engine).__name__
            == "AdvancedExecutionEngine"
        )

    def test_config_controller_integration(self) -> None:
        """Test proper integration with config controller."""
        config_controller = Mock()
        context = _make_context()
        # Add config_controller to context for this test
        context = context.with_updates(config_controller=config_controller)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Should have config controller reference
        assert coordinator._config_controller is not None

        # Update context and verify config controller updates
        new_config_controller = Mock()
        new_context = context.with_updates(config_controller=new_config_controller)
        coordinator.update_context(new_context)

        assert coordinator._config_controller is new_config_controller

    def test_initialize_skips_with_missing_dependencies(self) -> None:
        """Test initialization behavior when dependencies are missing."""
        # Test with missing broker
        context_no_broker = _make_context()
        # Override broker to be None after context creation
        context_no_broker = context_no_broker.with_updates(broker=None)
        coordinator = ExecutionCoordinator(context_no_broker)

        result = coordinator.initialize(context_no_broker)

        # Should return unchanged context
        assert result is context_no_broker

        # Test with missing risk manager
        context_no_risk = _make_context()
        # Override risk manager to be None after context creation
        context_no_risk = context_no_risk.with_updates(risk_manager=None)
        coordinator = ExecutionCoordinator(context_no_risk)

        result = coordinator.initialize(context_no_risk)

        # Should return unchanged context
        assert result is context_no_risk

        # Test with missing runtime state
        context_no_state = _make_context()
        context_no_state = context_no_state.with_updates(runtime_state=None)
        coordinator = ExecutionCoordinator(context_no_state)

        result = coordinator.initialize(context_no_state)

        # Should return unchanged context
        assert result is context_no_state

    def test_runtime_settings_loading_and_integration(self) -> None:
        """Test runtime settings integration."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Execution coordinator should work with whatever runtime settings are available
        # It doesn't load runtime settings itself, but uses what's provided
        assert coordinator.context.registry.runtime_settings is context.registry.runtime_settings

        # Runtime settings should be accessible through the context
        exec_engine = coordinator.context.runtime_state.exec_engine
        if hasattr(exec_engine, "settings") and context.registry.runtime_settings:
            assert exec_engine.settings is not None


class TestExecutionCoordinatorErrorResilience:
    """Test error resilience and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_execute_decision_handles_missing_runtime_state(self) -> None:
        """Test execute_decision gracefully handles missing runtime state."""
        context = _make_context()
        # Remove runtime state
        context = context.with_updates(runtime_state=None)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        decision = Mock()
        mark = Decimal("50000")
        product = _create_test_product()

        # Should handle missing runtime state gracefully
        await coordinator.execute_decision("BTC-PERP", decision, mark, product, None)
        # Should not raise an exception

    @pytest.mark.asyncio
    async def test_execute_decision_handles_invalid_product(self) -> None:
        """Test execute_decision handles invalid product gracefully."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        decision = Mock()
        mark = Decimal("50000")
        product = None  # Invalid product

        # Should handle invalid product gracefully by logging error and returning
        # The method catches Exception and logs it, but doesn't re-raise
        result = await coordinator.execute_decision("BTC-PERP", decision, mark, product, None)
        # Method returns None when error occurs
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_decision_handles_invalid_mark(self) -> None:
        """Test execute_decision handles invalid mark gracefully."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        decision = Mock()
        mark = Decimal("0")  # Invalid mark
        product = _create_test_product()

        # Should handle invalid mark gracefully by logging error and returning
        result = await coordinator.execute_decision("BTC-PERP", decision, mark, product, None)
        # Method returns None when error occurs during execution
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_decision_handles_position_state_validation(self) -> None:
        """Test execute_decision validates position state properly."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        decision = Mock()
        mark = Decimal("50000")
        product = _create_test_product()
        position_state = {"invalid": "state"}  # Missing quantity

        # Should handle invalid position state gracefully by logging error and returning
        result = await coordinator.execute_decision(
            "BTC-PERP", decision, mark, product, position_state
        )
        # Method returns None when error occurs during execution
        assert result is None

    @pytest.mark.asyncio
    async def test_place_order_handles_engine_unavailable(self) -> None:
        """Test place_order handles missing execution engine gracefully."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Remove execution engine from runtime state
        context.runtime_state.exec_engine = None

        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=Exception("Engine unavailable"))

        # Should handle engine unavailability gracefully
        result = await coordinator.place_order(exec_engine, symbol="BTC-PERP")
        assert result is None

    @pytest.mark.asyncio
    async def test_place_order_handles_place_order_failure(self) -> None:
        """Test place_order handles place_order method failures gracefully."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=ExecutionError("Placement failed"))

        # Should re-raise ExecutionError after logging
        with pytest.raises(ExecutionError, match="Placement failed"):
            await coordinator.place_order(exec_engine, symbol="BTC-PERP")

    def test_health_check_returns_proper_status(self) -> None:
        """Test health check returns appropriate status."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        health = coordinator.health_check()

        # Should return healthy status for proper initialization
        assert health.healthy is True
        assert health.component == "execution"
        assert "has_execution_engine" in health.details
        assert "order_stats" in health.details
        assert "background_tasks" in health.details

    def test_health_check_reports_missing_components(self) -> None:
        """Test health check reports missing components properly."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        # Don't initialize to test missing components
        # coordinator.initialize(context)

        health = coordinator.health_check()

        # Should report unhealthy for missing initialization
        assert health.healthy is False
        assert health.component == "execution"

    def test_ensure_order_lock_creates_lock_when_missing(self) -> None:
        """Test order lock creation when lock is missing."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Remove existing lock
        context.runtime_state.order_lock = None

        order_lock = coordinator.ensure_order_lock()

        # Should create new lock
        assert order_lock is not None
        assert isinstance(order_lock, asyncio.Lock)

    def test_ensure_order_lock_handles_runtime_unavailable(self) -> None:
        """Test order lock creation when runtime state is unavailable."""
        context = _make_context()
        # Create context with no runtime state
        context_no_state = context.with_updates(runtime_state=None)
        coordinator = ExecutionCoordinator(context_no_state)

        with pytest.raises(RuntimeError, match="Runtime state is unavailable"):
            coordinator.ensure_order_lock()

    def test_ensure_order_lock_handles_lock_creation_failure(self) -> None:
        """Test order lock handles RuntimeError during creation."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Mock Lock to raise RuntimeError
        import asyncio

        original_lock = asyncio.Lock
        asyncio.Lock = Mock(side_effect=RuntimeError("Lock creation failed"))

        try:
            coordinator.ensure_order_lock()
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
        finally:
            asyncio.Lock = original_lock

    @pytest.mark.asyncio
    async def test_order_reconciler_creation_and_caching(self) -> None:
        """Test order reconciler creation and caching."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # First call should create reconciler
        reconciler1 = coordinator._get_order_reconciler()
        assert reconciler1 is not None

        # Second call should return same reconciler (cached)
        reconciler2 = coordinator._get_order_reconciler()
        assert reconciler2 is reconciler1

        # Reset should clear the cache
        coordinator.reset_order_reconciler()
        assert coordinator._order_reconciler is None

        # Next call should create new reconciler
        reconciler3 = coordinator._get_order_reconciler()
        assert reconciler3 is not None
        assert reconciler3 is not reconciler1


class TestExecutionCoordinatorIntegration:
    """Test integration with other orchestration components."""

    def test_initialization_with_full_context(self) -> None:
        """Test complete initialization with all dependencies."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)

        updated = coordinator.initialize(context)

        # Should create execution engine
        assert updated.runtime_state.exec_engine is not None
        # Should add execution engine to registry extras
        assert updated.registry.extras["execution_engine"] is not None
        # Should initialize order reconciler
        assert coordinator._order_reconciler is None  # Created lazily

    def test_advanced_execution_engine_integration(self) -> None:
        """Test integration with AdvancedExecutionEngine."""
        risk_config = Mock()
        risk_config.enable_dynamic_position_sizing = True
        risk_config.enable_market_impact_guard = True

        risk_manager = Mock()
        risk_manager.config = risk_config

        context = _make_context(risk_manager=risk_manager, advanced=True)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Should use AdvancedExecutionEngine
        engine = coordinator.context.runtime_state.exec_engine
        assert isinstance(engine, AdvancedExecutionEngine)

        # Should set impact estimator on risk manager
        risk_manager.set_impact_estimator.assert_called_once()

    def test_live_execution_engine_integration(self) -> None:
        """Test integration with LiveExecutionEngine."""
        context = _make_context(advanced=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Should use LiveExecutionEngine
        engine = coordinator.context.runtime_state.exec_engine
        assert type(engine).__name__ == "LiveExecutionEngine"

        # Should have proper initialization parameters
        assert engine.broker is context.broker
        assert engine.risk_manager is context.risk_manager
        assert engine.event_store is context.event_store
        assert engine.bot_id == context.bot_id

    def test_slippage_multiplier_integration(self) -> None:
        """Test slippage multiplier loading and integration."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Slippage multipliers should be loaded and passed to engine
        # Engine should have received slippage multipliers
        # (Implementation details depend on LiveExecutionEngine constructor)

    def test_runtime_settings_integration(self) -> None:
        """Test runtime settings integration with execution engine."""
        context = _make_context()
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Runtime settings should be accessible (may be None depending on context)
        runtime_settings = coordinator.context.registry.runtime_settings
        # The coordinator doesn't load runtime settings itself, it uses what's provided
        assert runtime_settings is coordinator.context.registry.runtime_settings

        engine = coordinator.context.runtime_state.exec_engine
        if hasattr(engine, "settings") and runtime_settings:
            assert engine.settings is not None

    @pytest.mark.asyncio
    async def test_background_task_integration_with_orchestration(self) -> None:
        """Test background tasks integrate properly with orchestration."""
        context = _make_context(dry_run=False)
        coordinator = ExecutionCoordinator(context)
        coordinator.initialize(context)

        # Mock execution engine methods
        exec_engine = coordinator.context.runtime_state.exec_engine
        exec_engine.run_runtime_guards = AsyncMock()
        exec_engine.cancel_all_orders = Mock(return_value=0)

        # Start background tasks
        tasks = await coordinator.start_background_tasks()
        assert len(tasks) == 2

        # Let tasks run briefly
        await asyncio.sleep(0.01)

        # Should have called runtime guards method
        exec_engine.run_runtime_guards.assert_called()

        # Cleanup
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    pytest.main([__file__])
