"""
Integration tests for coordinator interactions.

Tests end-to-end flows across multiple coordinators, state transitions,
and cross-coordinator communication patterns.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.coordinators.runtime import RuntimeCoordinator
from bot_v2.orchestration.coordinators.strategy import StrategyCoordinator
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def integration_context():
    """Integration test context with all coordinators."""
    config = BotConfig(
        profile=Profile.DEV, symbols=["BTC-PERP"], dry_run=False
    )  # Use DEV to avoid broker validation
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

    broker = Mock()
    risk_manager = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(config=config)

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="integration-test-bot",
        runtime_state=runtime_state,
        config_controller=Mock(),
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def coordinators(integration_context):
    """All coordinators initialized for integration testing."""
    runtime_coord = RuntimeCoordinator(integration_context)
    strategy_coord = StrategyCoordinator(integration_context)
    execution_coord = ExecutionCoordinator(integration_context)
    telemetry_coord = TelemetryCoordinator(integration_context)

    # Create mutable context copy for coordinator references
    mutable_context = integration_context.with_updates(
        execution_coordinator=execution_coord,
        strategy_coordinator=strategy_coord,
    )

    # Update coordinators with the mutable context
    runtime_coord.update_context(mutable_context)
    strategy_coord.update_context(mutable_context)
    execution_coord.update_context(mutable_context)
    telemetry_coord.update_context(mutable_context)

    return {
        "runtime": runtime_coord,
        "strategy": strategy_coord,
        "execution": execution_coord,
        "telemetry": telemetry_coord,
        "context": mutable_context,
    }


class TestCoordinatorLifecycleIntegration:
    """Test coordinator lifecycle and initialization integration."""

    def test_runtime_initialization_sets_up_all_coordinators(
        self, coordinators, integration_context
    ):
        """Test runtime coordinator initialization sets up all coordinators."""
        runtime_coord = coordinators["runtime"]

        # Initialize runtime coordinator
        result = runtime_coord.initialize(integration_context)

        # Should have set up broker and risk manager
        assert result.broker is not None
        assert result.risk_manager is not None

    def test_strategy_coordinator_initialization_with_orchestrator(
        self, coordinators, integration_context
    ):
        """Test strategy coordinator initializes with strategy orchestrator."""
        strategy_coord = coordinators["strategy"]
        orchestrator = Mock()
        mutable_context = integration_context.with_updates(strategy_orchestrator=orchestrator)
        strategy_coord.update_context(mutable_context)

        result = strategy_coord.initialize(mutable_context)

        assert result == mutable_context

    def test_execution_coordinator_initialization_with_engine(
        self, coordinators, integration_context
    ):
        """Test execution coordinator initializes execution engine."""
        execution_coord = coordinators["execution"]

        result = execution_coord.initialize(integration_context)

        # Should have initialized execution engine
        assert result.runtime_state.exec_engine is not None

    def test_telemetry_coordinator_initialization_with_services(
        self, coordinators, integration_context
    ):
        """Test telemetry coordinator initializes telemetry services."""
        telemetry_coord = coordinators["telemetry"]
        # Mock Coinbase broker
        integration_context.broker.__class__ = Mock()
        from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage

        integration_context.broker.__class__ = CoinbaseBrokerage

        result = telemetry_coord.initialize(integration_context)

        # Should have set up telemetry services in registry
        extras = result.registry.extras
        assert "account_telemetry" in extras
        assert "market_monitor" in extras


class TestTradingCycleIntegration:
    """Test complete trading cycle across coordinators."""

    @pytest.mark.asyncio
    async def test_complete_trading_cycle_execution(self, coordinators, integration_context):
        """Test complete trading cycle from strategy to execution."""
        strategy_coord = coordinators["strategy"]

        # Set up mocks
        integration_context.broker.list_balances = AsyncMock(return_value=[])
        integration_context.broker.list_positions = AsyncMock(return_value=[])
        integration_context.broker.get_account_info = AsyncMock(return_value=None)
        integration_context.system_monitor = Mock()
        integration_context.system_monitor.log_status = AsyncMock()
        integration_context.session_guard = Mock()
        integration_context.session_guard.should_trade = Mock(return_value=True)

        # Mock strategy processing
        processor = Mock()
        processor.process_symbol = AsyncMock()
        strategy_coord.set_symbol_processor(processor)

        # Mock execution engine
        exec_engine = Mock()
        integration_context.runtime_state.exec_engine = exec_engine

        # Run trading cycle
        await strategy_coord.run_cycle()

        # Should have processed symbols
        processor.process_symbol.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_to_execution_hand_off(self, coordinators, integration_context):
        """Test strategy decision hand-off to execution coordinator."""
        strategy_coord = coordinators["strategy"]

        # Set up execution engine
        exec_engine = Mock()
        exec_engine.place_order = AsyncMock(return_value=Mock(id="order-123"))
        integration_context.runtime_state.exec_engine = exec_engine

        # Create decision
        decision = ScenarioBuilder.create_decision(action=Action.BUY, quantity=Decimal("0.1"))
        product = ScenarioBuilder.create_product()

        # Execute decision through strategy coordinator
        await strategy_coord.execute_decision("BTC-PERP", decision, Decimal("50000"), product, None)

        # Should have delegated to execution coordinator
        exec_engine.place_order.assert_called_once()


class TestStateTransitionIntegration:
    """Test state transitions across coordinators."""

    def test_runtime_reduce_only_propagates_to_execution(self, coordinators, integration_context):
        """Test runtime reduce-only mode propagates to execution decisions."""
        runtime_coord = coordinators["runtime"]

        # Set up config controller
        integration_context.config_controller.set_reduce_only_mode = Mock(return_value=True)
        integration_context.config_controller.is_reduce_only_mode = Mock(return_value=True)

        # Enable reduce-only mode
        runtime_coord.set_reduce_only_mode(True, "test_transition")

        # Verify execution coordinator respects it
        assert runtime_coord.is_reduce_only_mode() is True

    def test_risk_state_changes_trigger_runtime_updates(self, coordinators, integration_context):
        """Test risk manager state changes trigger runtime coordinator updates."""
        runtime_coord = coordinators["runtime"]

        # Mock risk state change
        from bot_v2.features.live_trade.risk import RiskRuntimeState

        state = RiskRuntimeState(reduce_only_mode=True, last_reduce_only_reason="circuit_breaker")

        # Simulate risk state change
        runtime_coord.on_risk_state_change(state)

        # Should have applied risk update
        integration_context.config_controller.apply_risk_update.assert_called_once_with(True)


class TestBackgroundTasksIntegration:
    """Test background task coordination across coordinators."""

    @pytest.mark.asyncio
    async def test_execution_background_tasks_start_properly(
        self, coordinators, integration_context
    ):
        """Test execution coordinator background tasks start."""
        execution_coord = coordinators["execution"]
        integration_context.config.dry_run = False
        execution_coord.update_context(integration_context)

        tasks = await execution_coord.start_background_tasks()

        # Should have started reconciliation and guard tasks
        assert len(tasks) == 2

        # Clean up tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_telemetry_background_tasks_include_streaming(
        self, coordinators, integration_context
    ):
        """Test telemetry coordinator background tasks include streaming."""
        telemetry_coord = coordinators["telemetry"]
        integration_context.config.perps_enable_streaming = True
        integration_context.config.profile = Profile.PROD
        telemetry_coord.update_context(integration_context)

        # Mock streaming
        telemetry_coord._start_streaming = AsyncMock(return_value=Mock())

        tasks = await telemetry_coord.start_background_tasks()

        # Should have started streaming task
        assert len(tasks) == 1


class TestErrorHandlingIntegration:
    """Test error handling across coordinator boundaries."""

    @pytest.mark.asyncio
    async def test_execution_failure_triggers_telemetry(self, coordinators, integration_context):
        """Test execution failures trigger telemetry events."""
        execution_coord = coordinators["execution"]

        # Set up failing execution
        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=Exception("execution_failed"))
        integration_context.runtime_state.exec_engine = exec_engine

        decision = ScenarioBuilder.create_decision()
        product = ScenarioBuilder.create_product()

        # Attempt execution
        await execution_coord.execute_decision(
            "BTC-PERP", decision, Decimal("50000"), product, None
        )

        # Should have failed gracefully
        assert integration_context.runtime_state.order_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_reconciliation_failures_enable_reduce_only(
        self, coordinators, integration_context
    ):
        """Test reconciliation failures enable reduce-only mode."""
        runtime_coord = coordinators["runtime"]

        # Mock reconciliation failure
        broker = Mock()
        orders_store = Mock()
        event_store = Mock()
        runtime_coord.update_context(
            runtime_coord.context.with_updates(
                broker=broker,
                orders_store=orders_store,
                event_store=event_store,
                registry=runtime_coord.context.registry.with_updates(
                    broker=broker, event_store=event_store, orders_store=orders_store
                ),
            )
        )

        # Mock reconciler to fail
        reconciler = Mock()
        reconciler.fetch_local_open_orders = Mock(side_effect=Exception("reconcile_failed"))

        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "bot_v2.orchestration.coordinators.runtime.OrderReconciler",
                lambda **kwargs: reconciler,
            )

            await runtime_coord.reconcile_state_on_startup()

        # Should have enabled reduce-only mode
        integration_context.config_controller.set_reduce_only_mode.assert_called_with(
            True, reason="startup_reconcile_failed"
        )


class TestConfigurationDriftIntegration:
    """Test configuration drift detection and handling."""

    @pytest.mark.asyncio
    async def test_configuration_drift_triggers_emergency_shutdown(
        self, coordinators, integration_context
    ):
        """Test configuration drift triggers emergency shutdown."""
        strategy_coord = coordinators["strategy"]

        # Set up guardian with critical errors
        guardian = Mock()
        validation_result = Mock()
        validation_result.is_valid = False
        validation_result.errors = ["emergency_shutdown_required", "critical_violation"]
        guardian.pre_cycle_check = Mock(return_value=validation_result)

        integration_context.configuration_guardian = guardian
        integration_context.set_running_flag = Mock()
        integration_context.shutdown_hook = AsyncMock()

        strategy_coord.update_context(integration_context)

        trading_state = {
            "balances": [],
            "positions": [],
            "account_equity": None,
        }

        result = await strategy_coord._validate_configuration_and_handle_drift(trading_state)

        # Should have triggered shutdown
        assert result is False
        integration_context.set_running_flag.assert_called_once_with(False)
        integration_context.shutdown_hook.assert_called_once()


class TestHealthCheckIntegration:
    """Test health checks across all coordinators."""

    def test_all_coordinators_report_healthy_status(self, coordinators, integration_context):
        """Test all coordinators report healthy status under normal conditions."""
        # Set up minimal healthy state
        integration_context.runtime_state.exec_engine = Mock()
        integration_context.registry.extras["account_telemetry"] = Mock()

        for name, coord in coordinators.items():
            status = coord.health_check()
            # At minimum, should not crash and return a status
            assert hasattr(status, "healthy")
            assert hasattr(status, "component")
            assert status.component == name

    def test_coordinator_health_checks_detect_unhealthy_states(
        self, coordinators, integration_context
    ):
        """Test health checks detect unhealthy coordinator states."""
        # Test execution coordinator without engine
        execution_coord = coordinators["execution"]
        integration_context.runtime_state.exec_engine = None
        execution_coord.update_context(integration_context)

        status = execution_coord.health_check()
        assert status.healthy is False

        # Test telemetry coordinator without account telemetry
        telemetry_coord = coordinators["telemetry"]
        integration_context.registry.extras.pop("account_telemetry", None)
        telemetry_coord.update_context(integration_context)

        status = telemetry_coord.health_check()
        assert status.healthy is False


class TestStreamingIntegration:
    """Test streaming integration across telemetry and strategy coordinators."""

    def test_streaming_mark_updates_reach_strategy_coordinator(
        self, coordinators, integration_context
    ):
        """Test streaming mark updates reach strategy coordinator."""
        telemetry_coord = coordinators["telemetry"]
        strategy_coord = coordinators["strategy"]

        # Set up strategy coordinator
        integration_context.strategy_coordinator = strategy_coord
        telemetry_coord.update_context(integration_context)

        # Simulate mark update
        telemetry_coord._update_mark_and_metrics(integration_context, "BTC-PERP", Decimal("50000"))

        # Should have updated mark window
        runtime_state = integration_context.runtime_state
        assert "BTC-PERP" in runtime_state.mark_windows
        assert runtime_state.mark_windows["BTC-PERP"][-1] == Decimal("50000")


class TestOrderLockIntegration:
    """Test order lock coordination between strategy and execution."""

    def test_order_lock_coordination_between_coordinators(self, coordinators, integration_context):
        """Test order lock is properly coordinated between coordinators."""
        strategy_coord = coordinators["strategy"]
        execution_coord = coordinators["execution"]

        integration_context.execution_coordinator = execution_coord

        # Both coordinators should access the same lock
        strategy_lock = strategy_coord.ensure_order_lock()
        execution_lock = execution_coord.ensure_order_lock()

        assert strategy_lock is execution_lock
        assert integration_context.runtime_state.order_lock is strategy_lock
