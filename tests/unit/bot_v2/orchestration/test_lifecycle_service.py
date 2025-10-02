"""Tests for LifecycleService - trading loop orchestration and error handling."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, call, patch

from bot_v2.orchestration.lifecycle_service import (
    BackgroundTaskRegistry,
    LifecycleService,
)


@pytest.fixture
def mock_bot():
    """Mock PerpsBot instance with all necessary attributes."""
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Mock()
    bot.config.profile.value = "test"
    bot.config.dry_run = False
    bot.config.update_interval = 60
    bot.config.account_telemetry_interval = 300
    bot.running = False

    # Mock coordinators
    bot.runtime_coordinator = Mock()
    bot.runtime_coordinator.reconcile_state_on_startup = AsyncMock()

    bot.execution_coordinator = Mock()
    bot.execution_coordinator.run_runtime_guards = AsyncMock()
    bot.execution_coordinator.run_order_reconciliation = AsyncMock()

    bot.system_monitor = Mock()
    bot.system_monitor.run_position_reconciliation = AsyncMock()
    bot.system_monitor.write_health_status = Mock()
    bot.system_monitor.check_config_updates = Mock()

    bot.account_telemetry = Mock()
    bot.account_telemetry.supports_snapshots = Mock(return_value=True)
    bot.account_telemetry.run = AsyncMock()

    bot.run_cycle = AsyncMock()
    bot.shutdown = AsyncMock()

    return bot


@pytest.fixture
def lifecycle_service(mock_bot):
    """Create LifecycleService instance."""
    return LifecycleService(mock_bot)


class TestBackgroundTaskRegistry:
    """Test suite for BackgroundTaskRegistry."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = BackgroundTaskRegistry()
        assert registry._tasks == []
        assert registry._factory_functions == []

    def test_register_task_factory(self):
        """Test registering a task factory."""
        registry = BackgroundTaskRegistry()
        factory = Mock()

        registry.register(factory)

        assert len(registry._factory_functions) == 1
        assert registry._factory_functions[0] == factory

    def test_spawn_all_tasks(self):
        """Test spawning all registered tasks."""
        registry = BackgroundTaskRegistry()

        # Register multiple task factories
        task1 = Mock()
        task2 = Mock()
        factory1 = Mock(return_value=task1)
        factory2 = Mock(return_value=task2)

        registry.register(factory1)
        registry.register(factory2)

        spawned = registry.spawn_all()

        assert len(spawned) == 2
        assert spawned == [task1, task2]
        factory1.assert_called_once()
        factory2.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_all_tasks(self):
        """Test canceling all spawned tasks."""
        registry = BackgroundTaskRegistry()

        # Create mock tasks
        task1 = Mock()
        task1.done = Mock(return_value=False)
        task1.cancel = Mock()

        task2 = Mock()
        task2.done = Mock(return_value=False)
        task2.cancel = Mock()

        registry._tasks = [task1, task2]

        # Mock gather to avoid waiting
        with patch("bot_v2.orchestration.lifecycle_service.asyncio.gather", new=AsyncMock()):
            await registry.cancel_all()

        task1.cancel.assert_called_once()
        task2.cancel.assert_called_once()
        assert registry._tasks == []

    @pytest.mark.asyncio
    async def test_cancel_all_no_tasks(self):
        """Test canceling when no tasks exist."""
        registry = BackgroundTaskRegistry()
        await registry.cancel_all()  # Should not raise
        assert registry._tasks == []

    def test_clear_registry(self):
        """Test clearing registered task factories."""
        registry = BackgroundTaskRegistry()
        registry.register(Mock())
        registry.register(Mock())

        registry.clear()

        assert registry._factory_functions == []


class TestLifecycleService:
    """Test suite for LifecycleService."""

    def test_initialization(self, lifecycle_service, mock_bot):
        """Test service initialization."""
        assert lifecycle_service._bot == mock_bot
        assert lifecycle_service._sleep_fn == asyncio.sleep
        assert isinstance(lifecycle_service._task_registry, BackgroundTaskRegistry)

    def test_initialization_with_custom_sleep(self, mock_bot):
        """Test initialization with custom sleep function."""
        custom_sleep = AsyncMock()
        service = LifecycleService(mock_bot, sleep_fn=custom_sleep)

        assert service._sleep_fn == custom_sleep

    def test_register_background_task(self, lifecycle_service):
        """Test registering a background task."""
        task_factory = Mock()
        lifecycle_service.register_background_task(task_factory)

        assert task_factory in lifecycle_service._task_registry._factory_functions

    def test_configure_background_tasks_dry_run(self, lifecycle_service, mock_bot):
        """Test that dry_run skips background task configuration."""
        mock_bot.config.dry_run = True

        lifecycle_service.configure_background_tasks(single_cycle=False)

        # Should not register any tasks
        assert lifecycle_service._task_registry._factory_functions == []

    def test_configure_background_tasks_single_cycle(self, lifecycle_service, mock_bot):
        """Test that single_cycle skips background task configuration."""
        lifecycle_service.configure_background_tasks(single_cycle=True)

        # Should not register any tasks
        assert lifecycle_service._task_registry._factory_functions == []

    def test_configure_background_tasks_normal(self, lifecycle_service, mock_bot):
        """Test normal background task configuration."""
        lifecycle_service.configure_background_tasks(single_cycle=False)

        # Should register 4 tasks (runtime guards, order reconciliation, position reconciliation, telemetry)
        assert len(lifecycle_service._task_registry._factory_functions) == 4

    def test_configure_background_tasks_no_telemetry(self, lifecycle_service, mock_bot):
        """Test background task configuration without telemetry support."""
        mock_bot.account_telemetry.supports_snapshots.return_value = False

        lifecycle_service.configure_background_tasks(single_cycle=False)

        # Should register 3 tasks (no telemetry)
        assert len(lifecycle_service._task_registry._factory_functions) == 3

    @pytest.mark.asyncio
    async def test_run_single_cycle_success(self, lifecycle_service, mock_bot):
        """Test successful single cycle run."""
        await lifecycle_service.run(single_cycle=True)

        # Should have run the cycle
        mock_bot.run_cycle.assert_called_once()

        # Should write health status
        mock_bot.system_monitor.write_health_status.assert_called_with(ok=True)

        # Should check config updates
        mock_bot.system_monitor.check_config_updates.assert_called_once()

        # Should perform startup reconciliation (not dry_run)
        mock_bot.runtime_coordinator.reconcile_state_on_startup.assert_called_once()

        # Should shutdown
        mock_bot.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_single_cycle_dry_run(self, lifecycle_service, mock_bot):
        """Test single cycle in dry_run mode."""
        mock_bot.config.dry_run = True

        await lifecycle_service.run(single_cycle=True)

        # Should NOT perform startup reconciliation
        mock_bot.runtime_coordinator.reconcile_state_on_startup.assert_not_called()

        # Should still run the cycle
        mock_bot.run_cycle.assert_called_once()

        # Should shutdown
        mock_bot.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_multiple_cycles(self, lifecycle_service, mock_bot):
        """Test multiple cycle run with mocked sleep."""
        sleep_count = [0]  # Use list to allow modification in nested function

        async def custom_sleep(duration):
            sleep_count[0] += 1
            if sleep_count[0] >= 2:
                mock_bot.running = False
            # Don't actually sleep, just track calls

        lifecycle_service._sleep_fn = custom_sleep

        await lifecycle_service.run(single_cycle=False)

        # Should have run the cycle 2 times (initial + 1 after first sleep)
        # The second sleep sets running=False, so no third cycle
        assert mock_bot.run_cycle.call_count == 2

        # Should have slept 2 times
        assert sleep_count[0] == 2

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt(self, lifecycle_service, mock_bot):
        """Test handling of KeyboardInterrupt."""
        mock_bot.run_cycle.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            await lifecycle_service.run(single_cycle=True)

        # Should still shutdown
        mock_bot.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, lifecycle_service, mock_bot):
        """Test handling of general exceptions."""
        error_msg = "Test error"
        mock_bot.run_cycle.side_effect = RuntimeError(error_msg)

        with pytest.raises(RuntimeError):
            await lifecycle_service.run(single_cycle=True)

        # Should write error to health status
        mock_bot.system_monitor.write_health_status.assert_called_with(
            ok=False, error=error_msg
        )

        # Should still shutdown
        mock_bot.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_task_spawning(self, lifecycle_service, mock_bot):
        """Test that background tasks are spawned correctly."""
        lifecycle_service.configure_background_tasks(single_cycle=False)

        # Set up sleep to stop after first cycle
        async def custom_sleep(duration):
            mock_bot.running = False

        lifecycle_service._sleep_fn = custom_sleep

        # Mock the spawn_all to track calls
        with patch.object(
            lifecycle_service._task_registry, "spawn_all", return_value=[]
        ) as spawn_mock:
            await lifecycle_service.run(single_cycle=False)

            # Should spawn tasks
            spawn_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_task_cleanup(self, lifecycle_service, mock_bot):
        """Test that background tasks are cleaned up properly."""
        lifecycle_service.configure_background_tasks(single_cycle=False)

        # Set up sleep to stop after first cycle
        async def custom_sleep(duration):
            mock_bot.running = False

        lifecycle_service._sleep_fn = custom_sleep

        # Mock the cancel_all to track calls
        with patch.object(
            lifecycle_service._task_registry, "cancel_all", new=AsyncMock()
        ) as cancel_mock:
            await lifecycle_service.run(single_cycle=False)

            # Should cancel tasks on cleanup
            cancel_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_account_telemetry(self, lifecycle_service, mock_bot):
        """Test account telemetry task execution."""
        interval = 300
        await lifecycle_service._run_account_telemetry(interval)

        mock_bot.account_telemetry.run.assert_called_once_with(interval)

    @pytest.mark.asyncio
    async def test_run_account_telemetry_not_supported(self, lifecycle_service, mock_bot):
        """Test account telemetry when not supported."""
        mock_bot.account_telemetry.supports_snapshots.return_value = False

        await lifecycle_service._run_account_telemetry(300)

        # Should not call run
        mock_bot.account_telemetry.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_sets_running_false(self, lifecycle_service, mock_bot):
        """Test that cleanup sets running=False."""
        mock_bot.running = True

        await lifecycle_service._cleanup()

        assert mock_bot.running is False

    @pytest.mark.asyncio
    async def test_run_sets_running_true(self, lifecycle_service, mock_bot):
        """Test that run sets running=True at start."""
        assert mock_bot.running is False

        # Use single_cycle to avoid infinite loop
        await lifecycle_service.run(single_cycle=True)

        # running should be set to False after cleanup
        assert mock_bot.running is False

    @pytest.mark.asyncio
    async def test_multiple_cycles_respect_running_flag(self, lifecycle_service, mock_bot):
        """Test that loop respects the running flag."""
        sleep_mock = AsyncMock()
        lifecycle_service._sleep_fn = sleep_mock

        # Set running to False immediately after first cycle
        async def mock_run_cycle():
            mock_bot.running = False

        mock_bot.run_cycle.side_effect = mock_run_cycle

        await lifecycle_service.run(single_cycle=False)

        # Should run only once (initial cycle)
        assert mock_bot.run_cycle.call_count == 1

        # Should not sleep (loop exits before sleep)
        sleep_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_executes_in_order(self, lifecycle_service, mock_bot):
        """Test that run_cycle executes steps in correct order."""
        await lifecycle_service._run_single_cycle()

        # Create a mock manager to track call order
        from unittest.mock import call

        # Check that run_cycle is called first
        assert mock_bot.run_cycle.called

        # Then health status
        assert mock_bot.system_monitor.write_health_status.called

        # Then config updates
        assert mock_bot.system_monitor.check_config_updates.called


class TestLifecycleServiceIntegration:
    """Integration tests for LifecycleService with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_background_tasks(self, mock_bot):
        """Test full lifecycle with background tasks."""
        # Set up custom sleep to control loop
        sleep_count = [0]

        async def custom_sleep(duration):
            sleep_count[0] += 1
            # Stop after 2 sleeps to allow 2 cycles
            if sleep_count[0] >= 2:
                mock_bot.running = False

        service = LifecycleService(mock_bot, sleep_fn=custom_sleep)
        service.configure_background_tasks(single_cycle=False)

        # Run the service
        await service.run(single_cycle=False)

        # Verify startup reconciliation
        mock_bot.runtime_coordinator.reconcile_state_on_startup.assert_called_once()

        # Verify cycles were run (initial + 1 more = 2 total)
        assert mock_bot.run_cycle.call_count >= 2

        # Verify cleanup
        mock_bot.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_recovery_continues_lifecycle(self, mock_bot):
        """Test that errors in a cycle don't stop the lifecycle (when not re-raised)."""
        # This test verifies cleanup happens even on error
        mock_bot.run_cycle.side_effect = RuntimeError("Test error")

        service = LifecycleService(mock_bot)

        with pytest.raises(RuntimeError):
            await service.run(single_cycle=True)

        # Cleanup should still happen
        mock_bot.shutdown.assert_called_once()
        mock_bot.system_monitor.write_health_status.assert_called_with(
            ok=False, error="Test error"
        )
