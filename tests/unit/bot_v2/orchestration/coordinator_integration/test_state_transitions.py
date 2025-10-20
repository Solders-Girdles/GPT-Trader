"""
State transition integration tests across coordinators.
"""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, Mock

import pytest


class TestStateTransitionIntegration:
    """Validate cross-coordinator state propagation."""

    def test_runtime_reduce_only_propagates_to_execution(self, coordinators, integration_context):
        runtime_coord = coordinators["runtime"]

        integration_context.config_controller.set_reduce_only_mode = Mock(return_value=True)
        integration_context.config_controller.is_reduce_only_mode = Mock(return_value=True)

        runtime_coord.set_reduce_only_mode(True, "test_transition")

        assert runtime_coord.is_reduce_only_mode() is True

    def test_risk_state_changes_trigger_runtime_updates(self, coordinators, integration_context):
        runtime_coord = coordinators["runtime"]

        from bot_v2.features.live_trade.risk import RiskRuntimeState

        state = RiskRuntimeState(reduce_only_mode=True, last_reduce_only_reason="circuit_breaker")

        runtime_coord.on_risk_state_change(state)

        integration_context.config_controller.apply_risk_update.assert_called_once_with(True)


class TestBackgroundTasksIntegration:
    """Ensure background tasks are started as expected."""

    @pytest.mark.asyncio
    async def test_execution_background_tasks_start_properly(
        self, coordinators, integration_context
    ):
        execution_coord = coordinators["execution"]
        integration_context.config.dry_run = False
        execution_coord.update_context(integration_context)

        if not hasattr(integration_context.orders_store, "get_open_orders"):
            integration_context.orders_store.get_open_orders = Mock(return_value=[])

        tasks = await execution_coord.start_background_tasks()

        assert len(tasks) == 2

        for task in tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_telemetry_background_tasks_include_streaming(
        self, coordinators, integration_context
    ):
        telemetry_coord = coordinators["telemetry"]
        from bot_v2.orchestration.configuration import Profile

        integration_context.config.perps_enable_streaming = True
        integration_context.config.profile = Profile.PROD
        telemetry_coord.update_context(integration_context)

        telemetry_coord._start_streaming = AsyncMock(return_value=Mock())

        tasks = await telemetry_coord.start_background_tasks()

        assert len(tasks) == 1
