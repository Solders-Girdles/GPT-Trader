"""Tests for the health check runner and result types."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.app.health_server import HealthState
from gpt_trader.monitoring.health_checks import HealthCheckResult, HealthCheckRunner


class TestHealthCheckRunner:
    """Tests for HealthCheckRunner class."""

    def test_initialization(self) -> None:
        """Test runner initialization with default values."""
        health_state = HealthState()
        runner = HealthCheckRunner(health_state=health_state)

        assert runner._health_state is health_state
        assert runner._broker is None
        assert runner._degradation_state is None
        assert runner._risk_manager is None
        assert runner._market_data_service is None
        assert runner._interval == 30.0
        assert runner._running is False

    def test_set_broker(self) -> None:
        """Test setting broker after initialization."""
        runner = HealthCheckRunner(health_state=HealthState())
        broker = MagicMock()

        runner.set_broker(broker)

        assert runner._broker is broker

    def test_set_degradation_state(self) -> None:
        """Test setting degradation state."""
        runner = HealthCheckRunner(health_state=HealthState())
        state = MagicMock()

        runner.set_degradation_state(state)

        assert runner._degradation_state is state

    def test_run_checks_sync(self) -> None:
        """Test synchronous check execution."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 123}
        broker.get_ws_health.return_value = None

        degradation_state = MagicMock()
        degradation_state.get_status.return_value = {
            "global_paused": False,
            "global_reason": None,
            "paused_symbols": {},
            "global_remaining_seconds": 0,
        }

        runner = HealthCheckRunner(
            health_state=HealthState(),
            broker=broker,
            degradation_state=degradation_state,
        )

        results = runner.run_checks_sync()

        assert "broker" in results
        assert "websocket" in results
        assert "degradation" in results
        assert results["broker"][0] is True  # healthy
        assert results["degradation"][0] is True  # healthy

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Test async start and stop."""
        import asyncio

        runner = HealthCheckRunner(
            health_state=HealthState(),
            interval_seconds=60.0,
        )  # Long interval to avoid loop execution

        await runner.start()
        assert runner._running is True
        assert runner._task is not None

        # Give the task a moment to start the loop
        await asyncio.sleep(0.01)

        await runner.stop()
        assert runner._running is False
        assert runner._task is None

    @pytest.mark.asyncio
    async def test_execute_checks_updates_health_state(self) -> None:
        """Test that execute_checks updates the global health state."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 123}
        broker.get_ws_health.return_value = None

        health_state = MagicMock()
        runner = HealthCheckRunner(
            health_state=health_state,
            broker=broker,
            interval_seconds=60.0,
        )

        await runner._execute_checks()

        # Verify checks were added
        assert health_state.add_check.call_count >= 2
        call_args = [call[0][0] for call in health_state.add_check.call_args_list]
        assert "broker" in call_args
        assert "websocket" in call_args


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a health check result."""
        result = HealthCheckResult(healthy=True, details={"latency_ms": 50})

        assert result.healthy is True
        assert result.details["latency_ms"] == 50

    def test_unhealthy_result(self) -> None:
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            details={"error": "connection refused", "severity": "critical"},
        )

        assert result.healthy is False
        assert result.details["error"] == "connection refused"
