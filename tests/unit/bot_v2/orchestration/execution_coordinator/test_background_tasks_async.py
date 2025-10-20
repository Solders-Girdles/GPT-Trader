"""
Tests for ExecutionCoordinator background tasks async functionality.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest


class TestExecutionCoordinatorBackgroundTasks:
    """Test ExecutionCoordinator background tasks async functionality."""

    @pytest.mark.asyncio
    async def test_start_background_tasks_dry_run_skips(
        self, execution_coordinator, execution_context
    ):
        """Test start_background_tasks skips in dry run mode."""
        execution_context.config.dry_run = True

        tasks = await execution_coordinator.start_background_tasks()

        assert tasks == []

    @pytest.mark.asyncio
    async def test_start_background_tasks_no_runtime_state(
        self, execution_coordinator, execution_context
    ):
        """Test start_background_tasks handles missing runtime state."""
        # Create new coordinator with runtime_state=None context
        from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator

        context_with_no_runtime = execution_context.with_updates(runtime_state=None)
        coordinator_no_runtime = ExecutionCoordinator(context_with_no_runtime)

        tasks = await coordinator_no_runtime.start_background_tasks()

        # Should return empty list when runtime_state is None
        assert tasks == []

    @pytest.mark.asyncio
    async def test_start_background_tasks_creates_tasks(
        self, execution_coordinator, execution_context
    ):
        """Test start_background_tasks creates background tasks."""
        execution_context.runtime_state.exec_engine = Mock()

        tasks = await execution_coordinator.start_background_tasks()

        assert len(tasks) == 2  # guards and reconciliation tasks
        assert all(hasattr(task, "cancel") for task in tasks)

    @pytest.mark.asyncio
    async def test_run_runtime_guards_loop_exception_handling(
        self, execution_coordinator, execution_context
    ):
        """Test _run_runtime_guards_loop handles exceptions."""
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.run_runtime_guards = Mock(
            side_effect=Exception("Guard error")
        )

        # Run for a short time then cancel
        import asyncio

        task = asyncio.create_task(execution_coordinator._run_runtime_guards_loop())

        await asyncio.sleep(0.1)  # Let it run one iteration
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_loop_exception_handling(
        self, execution_coordinator, execution_context
    ):
        """Test _run_order_reconciliation_loop handles exceptions."""
        from unittest.mock import patch

        with patch.object(execution_coordinator, "_get_order_reconciler") as mock_get_reconciler:
            mock_reconciler = Mock()
            mock_reconciler.fetch_local_open_orders = Mock(return_value={})
            mock_reconciler.fetch_exchange_open_orders = AsyncMock(
                side_effect=Exception("Reconciliation error")
            )
            mock_get_reconciler.return_value = mock_reconciler

            # Run for a short time then cancel
            import asyncio

            task = asyncio.create_task(
                execution_coordinator._run_order_reconciliation_loop(interval_seconds=0.1)
            )

            await asyncio.sleep(0.2)  # Let it run one iteration
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass  # Expected

    @pytest.mark.asyncio
    async def test_run_runtime_guards_missing_engine(
        self, execution_coordinator, execution_context
    ):
        """Test _run_runtime_guards_loop handles missing execution engine."""
        execution_context.runtime_state.exec_engine = None

        # Run for a short time then cancel
        import asyncio

        task = asyncio.create_task(execution_coordinator._run_runtime_guards_loop())

        await asyncio.sleep(0.1)  # Let it run one iteration
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_run_runtime_guards_engine_exception(
        self, execution_coordinator, execution_context
    ):
        """Test _run_runtime_guards_loop handles engine exceptions."""
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.run_runtime_guards = Mock(
            side_effect=Exception("Engine error")
        )

        # Run for a short time then cancel
        import asyncio

        task = asyncio.create_task(execution_coordinator._run_runtime_guards_loop())

        await asyncio.sleep(0.1)  # Let it run one iteration
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_health_check_with_background_tasks(
        self, execution_coordinator, execution_context
    ):
        """Test health_check includes background task count."""
        execution_coordinator._background_tasks = [asyncio.create_task(asyncio.sleep(1))]

        health = execution_coordinator.health_check()

        assert health.details["background_tasks"] == 1
