import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, PropertyMock
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.runtime_settings import RuntimeSettings

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


@pytest.mark.asyncio
async def test_start_background_tasks_initializes_reconciliation_and_guards(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
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
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test start_background_tasks skips in dry run mode."""
    base_context = base_context.with_updates(
        config=base_context.config.with_overrides(dry_run=True)
    )
    coordinator.update_context(base_context)

    tasks = await coordinator.start_background_tasks()

    assert tasks == []


@pytest.mark.asyncio
async def test_run_order_reconciliation_loop_with_custom_interval(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test _run_order_reconciliation_loop respects custom interval."""
    # Mock reconciler to avoid actual reconciliation
    coordinator._get_order_reconciler = Mock(return_value=Mock())
    coordinator._run_order_reconciliation_cycle = AsyncMock()

    task = asyncio.create_task(coordinator._run_order_reconciliation_loop(interval_seconds=0.01))
    await asyncio.sleep(0.05)  # Should run ~5 cycles
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

    # Should have run multiple cycles
    assert coordinator._run_order_reconciliation_cycle.call_count >= 2


def test_initialize_uses_advanced_engine_for_dynamic_sizing(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
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
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
) -> None:
    """Test initialize uses advanced engine when market impact guard enabled."""
    risk_manager = Mock()
    risk_manager.config = Mock(enable_market_impact_guard=True)
    base_context = base_context.with_updates(risk_manager=risk_manager)
    coordinator.update_context(base_context)

    result = coordinator.initialize(base_context)

    # Should have initialized AdvancedExecutionEngine
    assert isinstance(result.runtime_state.exec_engine, AdvancedExecutionEngine)


def test_health_check_returns_unhealthy_without_engine(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
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
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
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


def test_initialize_handles_runtime_settings_not_runtime_settings(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test initialize handles non-RuntimeSettings in registry."""
    # Create registry with non-RuntimeSettings object
    registry = Mock()
    registry.runtime_settings = Mock()  # Not RuntimeSettings instance
    registry.extras = {}
    
    def with_updates_side_effect(**kwargs):
        if "runtime_settings" in kwargs:
            registry.runtime_settings = kwargs["runtime_settings"]
        return registry
        
    registry.with_updates.side_effect = with_updates_side_effect
    base_context = base_context.with_updates(registry=registry)

    with patch(
        "bot_v2.orchestration.coordinators.execution.core.load_runtime_settings",
        return_value=RuntimeSettings(),
    ):
        result = coordinator.initialize(base_context)

        # Should have updated registry with proper RuntimeSettings
        assert isinstance(result.registry.runtime_settings, RuntimeSettings)


@pytest.mark.asyncio
async def test_run_order_reconciliation_loop_handles_reconciler_none(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test reconciliation loop handles None reconciler gracefully."""
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])
    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    # Mock _get_order_reconciler to return None
    with patch.object(coordinator, "_get_order_reconciler", return_value=None):
        with patch("bot_v2.orchestration.coordinators.execution.background.logger") as mock_logger:
            # Should log debug about reconciliation error
            with patch("asyncio.sleep", side_effect=asyncio.CancelledError):
                with pytest.raises(asyncio.CancelledError):
                    await coordinator._run_order_reconciliation_loop()

            mock_logger.debug.assert_called()
            args, _ = mock_logger.debug.call_args
            assert "Order reconciliation error" in args[0] or "Order reconciliation error" in str(mock_logger.debug.call_args)


@pytest.mark.asyncio
async def test_run_order_reconciliation_cycle_handles_runtime_state_exception(
    coordinator: ExecutionCoordinator, base_context: CoordinatorContext
):
    """Test reconciliation cycle handles runtime state exceptions."""
    runtime_state = Mock()
    runtime_state.order_stats = {"failed": 1}
    # Use MagicMock to allow setting attributes/properties if needed, 
    # but here we want to simulate an error on attribute access.
    # Since we can't easily patch __getattr__ on a Mock instance to raise,
    # we can use a PropertyMock if it was a property, or just rely on the fact
    # that accessing a non-existent attribute on a non-spec Mock returns a Mock.
    # To raise an error, we can configure the mock.
    
    # However, the code under test likely accesses `runtime_state.exec_engine` or similar.
    # Let's try to mock the specific attribute access to raise.
    p = PropertyMock(side_effect=AttributeError("Test error"))
    type(runtime_state).exec_engine = p

    # Clean up is handled by mock scope usually, but here we modified the type of the mock.
    # It's safer to just use a class that raises.
    class BrokenState:
        order_stats = {"failed": 1}
        @property
        def exec_engine(self):
            raise AttributeError("Test error")
            
    runtime_state = BrokenState()

    base_context = base_context.with_updates(runtime_state=runtime_state)
    coordinator.update_context(base_context)

    reconciler = Mock()
    reconciler.fetch_exchange_open_orders = AsyncMock(side_effect=RuntimeError("Reconciliation failed"))
    reconciler.fetch_local_open_orders.return_value = {}

    with patch("bot_v2.orchestration.coordinators.execution.background.logger") as mock_logger:
        # The method does not catch exceptions, so it should raise
        with pytest.raises(RuntimeError, match="Reconciliation failed"):
            await coordinator._run_order_reconciliation_cycle(reconciler)
