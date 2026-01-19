"""Tests for telemetry_health.health_check()."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.live_trade.engines.telemetry_health import health_check


class TestHealthCheck:
    """Tests for health_check function."""

    def _create_mock_coordinator(self) -> Mock:
        """Create a mock coordinator."""
        coordinator = Mock()
        coordinator.name = "telemetry"
        coordinator._market_monitor = None
        coordinator._stream_task = None
        coordinator._background_tasks = []
        coordinator.context.registry.extras = {}
        return coordinator

    def test_healthy_when_market_monitor_present(self) -> None:
        """Test returns healthy when market_monitor is present."""
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.component == "telemetry"
        assert result.details["has_market_monitor"] is True

    def test_unhealthy_when_no_monitor_or_streaming(self) -> None:
        """Test returns unhealthy when no market_monitor and no streaming."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.healthy is False
        assert result.details["has_market_monitor"] is False
        assert result.details["streaming_active"] is False

    def test_healthy_when_streaming_active(self) -> None:
        """Test returns healthy when streaming is active (even without monitor)."""
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.details["streaming_active"] is True

    def test_detects_market_monitor_from_coordinator(self) -> None:
        """Test detects market_monitor from coordinator attribute."""
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is True

    def test_no_market_monitor(self) -> None:
        """Test reports no market_monitor when missing."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.details["has_market_monitor"] is False

    def test_streaming_active_when_task_running(self) -> None:
        """Test reports streaming_active when task is running."""
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.details["streaming_active"] is True

    def test_streaming_inactive_when_task_done(self) -> None:
        """Test reports streaming_inactive when task is done."""
        coordinator = self._create_mock_coordinator()
        task = Mock()
        task.done.return_value = True
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.details["streaming_active"] is False

    def test_streaming_inactive_when_no_task(self) -> None:
        """Test reports streaming_inactive when no task."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert result.details["streaming_active"] is False

    def test_counts_background_tasks(self) -> None:
        """Test counts background tasks."""
        coordinator = self._create_mock_coordinator()
        coordinator._background_tasks = [Mock(), Mock(), Mock()]

        result = health_check(coordinator)

        assert result.details["background_tasks"] == 3

    def test_health_with_both_monitor_and_streaming(self) -> None:
        """Test health when both market_monitor and streaming are present."""
        coordinator = self._create_mock_coordinator()
        coordinator._market_monitor = Mock()
        task = Mock()
        task.done.return_value = False
        coordinator._stream_task = task

        result = health_check(coordinator)

        assert result.healthy is True
        assert result.details["has_market_monitor"] is True
        assert result.details["streaming_active"] is True

    def test_health_details_always_present(self) -> None:
        """Test that all expected keys are present in details."""
        coordinator = self._create_mock_coordinator()

        result = health_check(coordinator)

        assert "has_market_monitor" in result.details
        assert "streaming_active" in result.details
        assert "background_tasks" in result.details
