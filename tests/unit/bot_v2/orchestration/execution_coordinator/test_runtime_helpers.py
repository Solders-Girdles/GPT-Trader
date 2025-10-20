"""
Helper utilities for ExecutionCoordinator (locks, stats, health).
"""

import asyncio
from unittest.mock import Mock, patch

import pytest


class TestExecutionCoordinatorOrderLock:
    """ensure ensure_order_lock covers edge cases."""

    def test_creates_lock_when_missing(self, execution_coordinator, execution_context):
        execution_context.runtime_state.order_lock = None

        lock = execution_coordinator.ensure_order_lock()

        assert isinstance(lock, asyncio.Lock)
        assert execution_context.runtime_state.order_lock is lock

    def test_returns_existing_lock(self, execution_coordinator, execution_context):
        existing_lock = asyncio.Lock()
        execution_context.runtime_state.order_lock = existing_lock

        lock = execution_coordinator.ensure_order_lock()

        assert lock is existing_lock

    def test_raises_when_runtime_state_missing(self, execution_coordinator, execution_context):
        execution_context = execution_context.with_updates(runtime_state=None)
        execution_coordinator.update_context(execution_context)

        with pytest.raises(RuntimeError, match="Runtime state is unavailable"):
            execution_coordinator.ensure_order_lock()

    def test_wraps_lock_creation_failures(self, execution_coordinator, execution_context):
        execution_context.runtime_state.order_lock = None

        with patch("asyncio.Lock", side_effect=RuntimeError("Lock creation failed")):
            with pytest.raises(RuntimeError, match="Lock creation failed"):
                execution_coordinator.ensure_order_lock()


class TestExecutionCoordinatorStats:
    """_increment_order_stat scenarios."""

    def test_updates_existing_stats(self, execution_coordinator, execution_context):
        execution_context.runtime_state.order_stats = {}

        execution_coordinator._increment_order_stat("successful")

        assert execution_context.runtime_state.order_stats["successful"] == 1

    def test_handles_missing_runtime_state(self, execution_coordinator, execution_context):
        execution_context = execution_context.with_updates(runtime_state=None)
        execution_coordinator.update_context(execution_context)

        execution_coordinator._increment_order_stat("successful")

    def test_handles_missing_stats_dict(self, execution_coordinator, execution_context):
        execution_context.runtime_state.order_stats = None

        execution_coordinator._increment_order_stat("successful")


class TestExecutionCoordinatorHealth:
    """health_check branch coverage."""

    def test_reports_healthy_with_engine(self, execution_coordinator, execution_context):
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.order_stats = {"attempted": 5, "successful": 4}

        health = execution_coordinator.health_check()

        assert health.healthy is True
        assert health.component == "execution"
        assert health.details["has_execution_engine"] is True
        assert health.details["order_stats"]["attempted"] == 5

    def test_reports_unhealthy_without_engine(self, execution_coordinator, execution_context):
        execution_context = execution_context.with_updates(runtime_state=None)
        execution_coordinator.update_context(execution_context)

        health = execution_coordinator.health_check()

        assert health.healthy is False
        assert health.component == "execution"
        assert health.details["has_execution_engine"] is False
