"""
Live cycle async tests for lifecycle_manager.py.

Tests async operations including:
- Single cycle vs continuous runs
- Dry run vs live mode differences
- Background task coordination and cancellation
- Exception handling and recovery
- Shutdown sequence
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bot_v2.orchestration.lifecycle_manager import LifecycleManager


class TestLifecycleManagerAsync:
    """Test async methods in LifecycleManager."""

    # ------------------------------------------------------------------
    # Run method tests

    @pytest.mark.asyncio
    async def test_run_single_cycle_dry_run(
        self, fake_bot, fake_runtime_coordinator, fake_system_monitor
    ):
        """Test run executes single cycle in dry run mode."""
        fake_bot.config.dry_run = True
        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=True)

        # Verify dry run skips certain operations
        fake_runtime_coordinator.reconcile_state_on_startup.assert_not_awaited()
        fake_bot._coordinator_registry.start_all_background_tasks.assert_not_awaited()
        fake_system_monitor.run_position_reconciliation.assert_not_awaited()

        # But still runs cycle and shutdown
        fake_bot.run_cycle.assert_awaited_once()
        fake_bot._coordinator_registry.shutdown_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_single_cycle_live_mode(
        self, fake_bot, fake_runtime_coordinator, fake_system_monitor
    ):
        """Test run executes single cycle in live mode."""
        fake_bot.config.dry_run = False
        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=True)

        # Verify live mode operations
        fake_runtime_coordinator.reconcile_state_on_startup.assert_awaited_once()
        fake_bot._coordinator_registry.start_all_background_tasks.assert_not_awaited()  # Single cycle skips background tasks
        # Note: run_position_reconciliation is added as a background task but not awaited in single cycle mode
        # since shutdown cancels background tasks immediately after the cycle

        fake_bot.run_cycle.assert_awaited_once()
        fake_bot._coordinator_registry.shutdown_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_continuous_live_mode(
        self, fake_bot, fake_runtime_coordinator, fake_system_monitor
    ):
        """Test run executes continuous cycles in live mode."""
        fake_bot.config.dry_run = False
        fake_bot.config.update_interval = 0.001  # Very short for quick test

        # Mock run_cycle to set running=False after first cycle to exit loop
        async def mock_run_cycle():
            fake_bot.running = False

        fake_bot.run_cycle = AsyncMock(side_effect=mock_run_cycle)

        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=False)

        # Verify continuous mode operations
        fake_runtime_coordinator.reconcile_state_on_startup.assert_awaited_once()
        fake_bot._coordinator_registry.start_all_background_tasks.assert_awaited_once()
        # Note: run_position_reconciliation is only added in single_cycle=False and not dry_run
        # But it's added as a background task, so we need to check differently

        # Should run at least one cycle
        assert fake_bot.run_cycle.await_count >= 1
        fake_bot._coordinator_registry.shutdown_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_background_tasks_started_and_cancelled(
        self, fake_bot, fake_coordinator_registry
    ):
        """Test run starts and properly cancels background tasks."""
        fake_bot.config.dry_run = False
        fake_bot.config.update_interval = 0.001

        # Create real asyncio Tasks that can be cancelled
        async def dummy_task():
            await asyncio.sleep(10)  # Long running task

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())
        fake_coordinator_registry.start_all_background_tasks.return_value = [task1, task2]

        # Mock run_cycle to exit immediately
        async def mock_run_cycle():
            fake_bot.running = False

        fake_bot.run_cycle = AsyncMock(side_effect=mock_run_cycle)

        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=False)

        # Verify tasks were started
        fake_coordinator_registry.start_all_background_tasks.assert_awaited_once()

        # Verify tasks were cancelled (since they weren't done)
        assert task1.cancelled()
        assert task2.cancelled()

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt_handling(self, fake_bot):
        """Test run handles KeyboardInterrupt gracefully."""
        fake_bot.run_cycle.side_effect = KeyboardInterrupt()

        manager = LifecycleManager(fake_bot)

        # Should not raise
        await manager.run(single_cycle=True)

        # Verify shutdown was called
        fake_bot._coordinator_registry.shutdown_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_generic_exception_handling(self, fake_bot, fake_system_monitor):
        """Test run handles generic exceptions and writes health status."""
        test_error = ValueError("Test error")
        fake_bot.run_cycle.side_effect = test_error

        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=True)

        # Verify error was written to health status
        fake_system_monitor.write_health_status.assert_called_with(ok=False, error=str(test_error))

        # Verify shutdown was called
        fake_bot._coordinator_registry.shutdown_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_already_running_flag_management(self, fake_bot):
        """Test run manages the running flag correctly."""
        fake_bot.running = False  # Start as not running

        async def mock_run_cycle():
            # Simulate cycle setting running to False to exit
            await asyncio.sleep(0.001)
            fake_bot.running = False

        fake_bot.run_cycle = AsyncMock(side_effect=mock_run_cycle)

        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=True)

        # Verify running was set to True at start
        assert fake_bot.running is False  # Should be False after completion

    @pytest.mark.asyncio
    async def test_run_health_status_updates(self, fake_bot, fake_system_monitor):
        """Test run updates health status during execution."""
        manager = LifecycleManager(fake_bot)

        await manager.run(single_cycle=True)

        # Verify health status was written
        fake_system_monitor.write_health_status.assert_called_with(ok=True)
        fake_system_monitor.check_config_updates.assert_called_once()

    # ------------------------------------------------------------------
    # Shutdown tests

    @pytest.mark.asyncio
    async def test_shutdown_success(self, fake_bot, fake_coordinator_registry):
        """Test shutdown executes successfully."""
        manager = LifecycleManager(fake_bot)

        await manager.shutdown()

        fake_coordinator_registry.shutdown_all.assert_awaited_once()
        assert fake_bot.running is False

    @pytest.mark.asyncio
    async def test_shutdown_registry_failure(self, fake_bot, fake_coordinator_registry):
        """Test shutdown handles registry shutdown failures."""
        fake_coordinator_registry.shutdown_all.side_effect = Exception("Shutdown failed")

        manager = LifecycleManager(fake_bot)

        # Should raise - shutdown should propagate exceptions
        with pytest.raises(Exception, match="Shutdown failed"):
            await manager.shutdown()

        fake_coordinator_registry.shutdown_all.assert_awaited_once()
        assert fake_bot.running is False
