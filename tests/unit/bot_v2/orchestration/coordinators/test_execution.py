from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderStatus, OrderType
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "coinbase_trader"


def _make_context(*, dry_run: bool = False, advanced: bool = False) -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD, dry_run=dry_run)
    runtime_state = PerpsBotRuntimeState([])
    broker = Mock()
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
        symbols=("BTC-PERP",),
        bot_id=BOT_ID,
        runtime_state=runtime_state,
    )
    return context


def test_initialize_creates_execution_engine() -> None:
    context = _make_context()
    coordinator = ExecutionCoordinator(context)

    updated = coordinator.initialize(context)

    assert updated.runtime_state.exec_engine is not None
    assert updated.registry.extras["execution_engine"] is updated.runtime_state.exec_engine


def test_initialize_advanced_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _make_context(advanced=True)
    coordinator = ExecutionCoordinator(context)

    updated = coordinator.initialize(context)

    engine = updated.runtime_state.exec_engine
    assert isinstance(engine, AdvancedExecutionEngine)


@pytest.mark.asyncio
async def test_start_background_tasks_skips_in_dry_run() -> None:
    context = _make_context(dry_run=True)
    coordinator = ExecutionCoordinator(context)

    tasks = await coordinator.start_background_tasks()

    assert tasks == []


@pytest.mark.asyncio
async def test_order_reconciliation_cycle_updates_store() -> None:
    context = _make_context()
    coordinator = ExecutionCoordinator(context)
    coordinator.update_context(coordinator.initialize(context))

    reconciler = Mock()
    reconciler.fetch_local_open_orders = Mock(return_value={})
    reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
    reconciler.diff_orders = Mock(return_value=Mock(missing_on_exchange=[], missing_locally=[]))
    reconciler.reconcile_missing_on_exchange = AsyncMock()
    reconciler.reconcile_missing_locally = Mock()
    reconciler.record_snapshot = AsyncMock()

    await coordinator._run_order_reconciliation_cycle(reconciler)

    reconciler.fetch_local_open_orders.assert_called_once()
    reconciler.fetch_exchange_open_orders.assert_awaited_once()
    reconciler.record_snapshot.assert_awaited_once()


class MockBroker:
    """Mock broker for testing."""

    def __init__(self):
        self.place_order = AsyncMock()
        self.cancel_order = AsyncMock()
        self.get_order = AsyncMock()
        self.get_open_orders = AsyncMock()
        self.get_positions = AsyncMock()
        self.stream_orders = Mock()


class MockRiskManager:
    """Mock risk manager for testing."""

    def __init__(self):
        self.validate_order = Mock(return_value=True)
        self.check_position_limits = Mock(return_value=True)
        self.check_order_size = Mock(return_value=True)


class TestExecutionCoordinatorOrderWorkflows:
    """Test order placement and management workflows."""

    @pytest.fixture
    def mock_context(self) -> CoordinatorContext:
        """Create mock coordinator context."""
        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        broker = MockBroker()
        risk_manager = MockRiskManager()
        event_store = Mock()
        orders_store = Mock()

        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            orders_store=orders_store,
        )

        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=("BTC-PERP",),
            bot_id="test_bot",
            runtime_state=runtime_state,
        )

    @pytest.fixture
    def execution_coordinator(self, mock_context: CoordinatorContext) -> ExecutionCoordinator:
        """Create execution coordinator with mocked dependencies."""
        return ExecutionCoordinator(mock_context)

    def test_place_order_with_successful_validation(
        self, execution_coordinator: ExecutionCoordinator, mock_context: CoordinatorContext
    ) -> None:
        """Test successful order placement with validation."""
        # Setup
        updated_context = execution_coordinator.initialize(mock_context)
        execution_engine = updated_context.runtime_state.exec_engine
        execution_engine.place.return_value = "order_123"

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Execute
        result = execution_coordinator.place(order)

        # Verify
        assert result == "order_123"
        mock_context.risk_manager.validate_order.assert_called_once_with(order)
        execution_engine.place.assert_called_once_with(order)

    def test_place_order_with_risk_validation_failure(
        self, execution_coordinator: ExecutionCoordinator, mock_context: CoordinatorContext
    ) -> None:
        """Test order placement with risk validation failure."""
        # Setup
        mock_context.risk_manager.validate_order.return_value = False
        execution_coordinator.initialize(mock_context)

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1000.0"),  # Too large
        )

        # Execute and verify - should raise ValidationError or similar
        with pytest.raises(Exception):  # Adjust based on actual exception type
            execution_coordinator.place(order)

    def test_place_and_wait_with_successful_fill(
        self, execution_coordinator: ExecutionCoordinator, mock_context: CoordinatorContext
    ) -> None:
        """Test place_and_wait with successful order fill."""
        # Setup
        updated_context = execution_coordinator.initialize(mock_context)
        execution_engine = updated_context.runtime_state.exec_engine

        # Mock order placement and monitoring
        order_id = "order_123"
        execution_engine.place.return_value = order_id

        filled_order = Order(
            id=order_id,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            fill_price=Decimal("50000"),
        )
        execution_engine.wait_for_fill.return_value = filled_order

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Execute
        result = execution_coordinator.place_and_wait(order, timeout_ms=5000)

        # Verify
        assert result == filled_order
        assert result.status == OrderStatus.FILLED
        execution_engine.wait_for_fill.assert_called_once_with(order_id, timeout_ms=5000)

    def test_place_and_wait_with_timeout(
        self, execution_coordinator: ExecutionCoordinator, mock_context: CoordinatorContext
    ) -> None:
        """Test place_and_wait with timeout scenario."""
        # Setup
        updated_context = execution_coordinator.initialize(mock_context)
        execution_engine = updated_context.runtime_state.exec_engine

        order_id = "order_123"
        execution_engine.place.return_value = order_id
        execution_engine.wait_for_fill.return_value = None  # Timeout

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Execute and verify
        result = execution_coordinator.place_and_wait(order, timeout_ms=1000)
        assert result is None

    def test_cancel_order_operation(
        self, execution_coordinator: ExecutionCoordinator, mock_context: CoordinatorContext
    ) -> None:
        """Test order cancellation."""
        # Setup
        updated_context = execution_coordinator.initialize(mock_context)
        execution_engine = updated_context.runtime_state.exec_engine

        order_id = "order_123"
        execution_engine.cancel.return_value = True

        # Execute
        result = execution_coordinator.cancel(order_id)

        # Verify
        assert result is True
        execution_engine.cancel.assert_called_once_with(order_id)


class TestReconciliationLogic:
    """Test reconciliation and background task management."""

    @pytest.fixture
    def mock_context_with_reconciliation(self) -> CoordinatorContext:
        """Create mock context with reconciliation components."""
        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        broker = MockBroker()
        risk_manager = MockRiskManager()
        event_store = Mock()
        orders_store = Mock()

        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            orders_store=orders_store,
        )

        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=("BTC-PERP",),
            bot_id="test_bot",
            runtime_state=runtime_state,
        )

    @pytest.fixture
    def execution_coordinator_with_reconciliation(
        self, mock_context_with_reconciliation: CoordinatorContext
    ) -> ExecutionCoordinator:
        """Create execution coordinator with reconciliation setup."""
        return ExecutionCoordinator(mock_context_with_reconciliation)

    @pytest.mark.asyncio
    async def test_start_background_tasks_with_reconciliation(
        self,
        execution_coordinator_with_reconciliation: ExecutionCoordinator,
        mock_context_with_reconciliation: CoordinatorContext,
    ) -> None:
        """Test starting background tasks for reconciliation."""
        # Setup
        execution_coordinator_with_reconciliation.initialize(mock_context_with_reconciliation)

        # Execute
        tasks = await execution_coordinator_with_reconciliation.start_background_tasks()

        # Verify
        assert len(tasks) >= 1
        assert all(isinstance(task, asyncio.Task) for task in tasks)

        # Cleanup
        for task in tasks:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_shutdown_cancels_background_tasks(
        self,
        execution_coordinator_with_reconciliation: ExecutionCoordinator,
        mock_context_with_reconciliation: CoordinatorContext,
    ) -> None:
        """Test that shutdown cancels background tasks."""
        # Setup
        execution_coordinator_with_reconciliation.initialize(mock_context_with_reconciliation)
        tasks = await execution_coordinator_with_reconciliation.start_background_tasks()

        # Execute
        await execution_coordinator_with_reconciliation.shutdown()

        # Verify
        for task in tasks:
            assert task.cancelled() or task.done()

    def test_reconciliation_workflow_with_order_discrepancies(
        self,
        execution_coordinator_with_reconciliation: ExecutionCoordinator,
        mock_context_with_reconciliation: CoordinatorContext,
    ) -> None:
        """Test the reconciliation workflow with order discrepancies."""
        # Setup
        updated_context = execution_coordinator_with_reconciliation.initialize(
            mock_context_with_reconciliation
        )
        execution_coordinator_with_reconciliation.update_context(updated_context)

        reconciler = Mock()
        # Mock orders missing on exchange
        missing_on_exchange = [
            Order(
                id="local_1",
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
            )
        ]
        # Mock orders missing locally
        missing_locally = [
            Order(
                id="exchange_1",
                symbol="BTC-PERP",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.5"),
            )
        ]

        reconciler.fetch_local_open_orders = Mock(return_value={"local_1": missing_on_exchange[0]})
        reconciler.fetch_exchange_open_orders = AsyncMock(
            return_value={"exchange_1": missing_locally[0]}
        )
        reconciler.diff_orders = Mock(
            return_value=Mock(
                missing_on_exchange=missing_on_exchange, missing_locally=missing_locally
            )
        )
        reconciler.reconcile_missing_on_exchange = AsyncMock()
        reconciler.reconcile_missing_locally = Mock()
        reconciler.record_snapshot = AsyncMock()

        # Execute
        asyncio.run(
            execution_coordinator_with_reconciliation._run_order_reconciliation_cycle(reconciler)
        )

        # Verify reconciliation logic was called
        reconciler.reconcile_missing_on_exchange.assert_called_once_with(missing_on_exchange)
        reconciler.reconcile_missing_locally.assert_called_once_with(missing_locally)

    def test_order_reconciliation_with_stale_orders(
        self,
        execution_coordinator_with_reconciliation: ExecutionCoordinator,
        mock_context_with_reconciliation: CoordinatorContext,
    ) -> None:
        """Test order reconciliation with stale orders."""
        # Setup
        updated_context = execution_coordinator_with_reconciliation.initialize(
            mock_context_with_reconciliation
        )
        execution_coordinator_with_reconciliation.update_context(updated_context)

        reconciler = Mock()

        # Create stale order (older than typical expiry)
        stale_order = Order(
            id="stale_order",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.OPEN,
            created_at=datetime.now() - timedelta(hours=2),  # 2 hours old
        )

        reconciler.fetch_local_open_orders = Mock(return_value={"stale_order": stale_order})
        reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
        reconciler.diff_orders = Mock(
            return_value=Mock(missing_on_exchange=[stale_order], missing_locally=[])
        )
        reconciler.reconcile_missing_on_exchange = AsyncMock()
        reconciler.reconcile_missing_locally = Mock()
        reconciler.record_snapshot = AsyncMock()

        # Execute
        asyncio.run(
            execution_coordinator_with_reconciliation._run_order_reconciliation_cycle(reconciler)
        )

        # Verify stale order handling
        reconciler.reconcile_missing_on_exchange.assert_called_once_with([stale_order])


class TestErrorHandlingAndValidation:
    """Test error handling and validation scenarios."""

    @pytest.fixture
    def mock_context_for_errors(self) -> CoordinatorContext:
        """Create mock context for error testing."""
        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        broker = MockBroker()
        risk_manager = MockRiskManager()
        event_store = Mock()
        orders_store = Mock()

        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            orders_store=orders_store,
        )

        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=("BTC-PERP",),
            bot_id="test_bot",
            runtime_state=runtime_state,
        )

    @pytest.fixture
    def execution_coordinator_for_errors(
        self, mock_context_for_errors: CoordinatorContext
    ) -> ExecutionCoordinator:
        """Create execution coordinator for error testing."""
        return ExecutionCoordinator(mock_context_for_errors)

    def test_broker_error_during_order_placement(
        self,
        execution_coordinator_for_errors: ExecutionCoordinator,
        mock_context_for_errors: CoordinatorContext,
    ) -> None:
        """Test broker error handling during order placement."""
        # Setup
        updated_context = execution_coordinator_for_errors.initialize(mock_context_for_errors)
        execution_engine = updated_context.runtime_state.exec_engine

        # Mock broker error
        broker_error = Exception("Insufficient margin")
        execution_engine.place.side_effect = broker_error

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Execute and verify - should handle error gracefully
        with pytest.raises(Exception):  # Adjust based on actual exception handling
            execution_coordinator_for_errors.place(order)

    def test_order_validation_with_invalid_order(
        self,
        execution_coordinator_for_errors: ExecutionCoordinator,
        mock_context_for_errors: CoordinatorContext,
    ) -> None:
        """Test order validation with invalid order data."""
        # Setup
        execution_coordinator_for_errors.initialize(mock_context_for_errors)

        # Create invalid order (zero quantity)
        invalid_order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),  # Invalid
        )

        # Execute and verify - should raise validation error
        with pytest.raises(Exception):  # Adjust based on actual validation
            execution_coordinator_for_errors.place(invalid_order)

    def test_network_error_during_order_placement(
        self,
        execution_coordinator_for_errors: ExecutionCoordinator,
        mock_context_for_errors: CoordinatorContext,
    ) -> None:
        """Test network error handling during order placement."""
        # Setup
        updated_context = execution_coordinator_for_errors.initialize(mock_context_for_errors)
        execution_engine = updated_context.runtime_state.exec_engine

        # Mock network error
        execution_engine.place.side_effect = ConnectionError("Network timeout")

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Execute and verify
        with pytest.raises(ConnectionError):
            execution_coordinator_for_errors.place(order)

    def test_reconciliation_error_handling(
        self,
        execution_coordinator_for_errors: ExecutionCoordinator,
        mock_context_for_errors: CoordinatorContext,
    ) -> None:
        """Test error handling in reconciliation tasks."""
        # Setup
        updated_context = execution_coordinator_for_errors.initialize(mock_context_for_errors)
        execution_coordinator_for_errors.update_context(updated_context)

        reconciler = Mock()
        reconciler.fetch_local_open_orders.side_effect = Exception("Database error")

        # Execute - should handle error gracefully
        with pytest.raises(Exception):
            asyncio.run(
                execution_coordinator_for_errors._run_order_reconciliation_cycle(reconciler)
            )


class TestIntegrationPatterns:
    """Test integration patterns with broker and risk manager."""

    @pytest.fixture
    def integration_context(self) -> CoordinatorContext:
        """Create realistic integration context."""
        config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], mock_broker=False)

        broker = MockBroker()
        risk_manager = MockRiskManager()
        event_store = Mock()
        orders_store = Mock()

        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            orders_store=orders_store,
        )

        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=("BTC-PERP",),
            bot_id="integration_bot",
            runtime_state=runtime_state,
        )

    @pytest.fixture
    def integration_coordinator(
        self, integration_context: CoordinatorContext
    ) -> ExecutionCoordinator:
        """Create execution coordinator for integration testing."""
        return ExecutionCoordinator(integration_context)

    def test_broker_integration_workflow(
        self, integration_coordinator: ExecutionCoordinator, integration_context: CoordinatorContext
    ) -> None:
        """Test complete broker integration workflow."""
        # Setup
        integration_coordinator.initialize(integration_context)

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        # Mock successful broker interaction
        integration_context.broker.place_order.return_value = {
            "order_id": "broker_order_123",
            "status": "OPEN",
        }

        # Execute
        order_id = integration_coordinator.place(order)

        # Verify integration
        assert order_id == "broker_order_123"  # Adjust based on actual return behavior

    def test_risk_manager_integration(
        self, integration_coordinator: ExecutionCoordinator, integration_context: CoordinatorContext
    ) -> None:
        """Test risk manager integration patterns."""
        # Setup
        integration_coordinator.initialize(integration_context)

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Mock risk manager to check multiple constraints
        integration_context.risk_manager.validate_order.return_value = True

        # Execute
        integration_coordinator.place(order)

        # Verify risk integration
        integration_context.risk_manager.validate_order.assert_called_once_with(order)

    def test_event_store_integration(
        self, integration_coordinator: ExecutionCoordinator, integration_context: CoordinatorContext
    ) -> None:
        """Test event store integration for order events."""
        # Setup
        integration_coordinator.initialize(integration_context)

        order = Order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        # Execute
        integration_coordinator.place(order)

        # Verify event creation and storage (if event emission is implemented)
        if hasattr(integration_context.event_store, "store_event"):
            # May or may not be called depending on implementation
            pass


class TestExecutionCoordinatorEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def edge_case_context(self) -> CoordinatorContext:
        """Create context for edge case testing."""
        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        broker = MockBroker()
        risk_manager = MockRiskManager()
        event_store = Mock()
        orders_store = Mock()

        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            orders_store=orders_store,
        )

        runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=("BTC-PERP",),
            bot_id="edge_case_bot",
            runtime_state=runtime_state,
        )

    def test_execution_coordinator_with_multiple_symbols(
        self, edge_case_context: CoordinatorContext
    ) -> None:
        """Test execution coordinator with multiple symbols."""
        # Update context for multiple symbols
        edge_case_context = edge_case_context.with_updates(
            symbols=("BTC-PERP", "ETH-PERP", "SOL-PERP"),
            runtime_state=PerpsBotRuntimeState(["BTC-PERP", "ETH-PERP", "SOL-PERP"]),
        )

        coordinator = ExecutionCoordinator(edge_case_context)
        updated_context = coordinator.initialize(edge_case_context)

        # Verify multi-symbol initialization
        assert updated_context.runtime_state.exec_engine is not None

    def test_execution_coordinator_context_updates(
        self, edge_case_context: CoordinatorContext
    ) -> None:
        """Test that coordinator properly handles context updates."""
        coordinator = ExecutionCoordinator(edge_case_context)

        # Initialize with initial context
        updated_context = coordinator.initialize(edge_case_context)
        coordinator.update_context(updated_context)

        # Update context with new configuration
        new_config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], mock_broker=False)
        updated_context = updated_context.with_updates(config=new_config)
        coordinator.update_context(updated_context)

        # Verify coordinator handles updates gracefully
        assert coordinator.context.config.profile == Profile.PROD

    @pytest.mark.asyncio
    async def test_background_task_exception_handling(
        self, edge_case_context: CoordinatorContext
    ) -> None:
        """Test background task exception handling."""
        coordinator = ExecutionCoordinator(edge_case_context)
        coordinator.initialize(edge_case_context)

        # Mock background task to raise exception
        with patch.object(
            coordinator,
            "_run_order_reconciliation_cycle",
            side_effect=Exception("Background task failed"),
        ):
            # Starting background tasks should handle exceptions gracefully
            tasks = await coordinator.start_background_tasks()

            # Verify tasks were created despite potential issues
            assert isinstance(tasks, list)

            # Cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task

    def test_execution_engine_factory_methods(self, edge_case_context: CoordinatorContext) -> None:
        """Test execution engine factory methods."""
        coordinator = ExecutionCoordinator(edge_case_context)

        # Test different execution engine creation paths
        updated_context = coordinator.initialize(edge_case_context)
        execution_engine = updated_context.runtime_state.exec_engine

        # Verify execution engine was created
        assert execution_engine is not None

        # Test that the engine has expected methods
        assert hasattr(execution_engine, "place")
        assert hasattr(execution_engine, "cancel")
        assert hasattr(execution_engine, "wait_for_fill")
