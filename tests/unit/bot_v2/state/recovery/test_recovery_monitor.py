"""
Unit tests for RecoveryMonitor

Tests monitoring lifecycle, failure detection, recovery initiation, and error handling.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.state.recovery.detection import FailureDetector
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryMode,
)
from bot_v2.state.recovery.monitor import RecoveryMonitor


@pytest.fixture
def mock_detector():
    """Create mock failure detector."""
    detector = Mock(spec=FailureDetector)
    detector.detect_failures = AsyncMock(return_value=[])
    return detector


@pytest.fixture
def mock_recovery_initiator():
    """Create mock recovery initiator."""
    return AsyncMock()


@pytest.fixture
def mock_is_critical():
    """Create mock critical checker."""
    return Mock(return_value=True)


@pytest.fixture
def mock_affected_components():
    """Create mock affected components getter."""
    return Mock(return_value=["system"])


@pytest.fixture
def mock_recovery_in_progress():
    """Create mock recovery in progress checker."""
    return Mock(return_value=False)


@pytest.fixture
def recovery_config():
    """Create recovery config with fast intervals for testing."""
    return RecoveryConfig(
        automatic_recovery_enabled=True,
        failure_detection_interval_seconds=0.01,  # Fast for tests
    )


@pytest.fixture
def monitor(
    mock_detector,
    mock_recovery_initiator,
    mock_is_critical,
    mock_affected_components,
    mock_recovery_in_progress,
    recovery_config,
):
    """Create monitor instance."""
    return RecoveryMonitor(
        detector=mock_detector,
        recovery_initiator=mock_recovery_initiator,
        is_critical_checker=mock_is_critical,
        affected_components_getter=mock_affected_components,
        recovery_in_progress_checker=mock_recovery_in_progress,
        config=recovery_config,
    )


class TestMonitorInit:
    """Test monitor initialization."""

    def test_init_with_all_dependencies(
        self,
        mock_detector,
        mock_recovery_initiator,
        mock_is_critical,
        mock_affected_components,
        mock_recovery_in_progress,
        recovery_config,
    ):
        """Should initialize with all dependencies."""
        monitor = RecoveryMonitor(
            detector=mock_detector,
            recovery_initiator=mock_recovery_initiator,
            is_critical_checker=mock_is_critical,
            affected_components_getter=mock_affected_components,
            recovery_in_progress_checker=mock_recovery_in_progress,
            config=recovery_config,
        )

        assert monitor.detector is mock_detector
        assert monitor.recovery_initiator is mock_recovery_initiator
        assert monitor.is_critical_checker is mock_is_critical
        assert monitor.affected_components_getter is mock_affected_components
        assert monitor.recovery_in_progress_checker is mock_recovery_in_progress
        assert monitor.config is recovery_config
        assert not monitor.is_running()

    def test_init_with_default_config(
        self,
        mock_detector,
        mock_recovery_initiator,
        mock_is_critical,
        mock_affected_components,
        mock_recovery_in_progress,
    ):
        """Should use default config if not provided."""
        monitor = RecoveryMonitor(
            detector=mock_detector,
            recovery_initiator=mock_recovery_initiator,
            is_critical_checker=mock_is_critical,
            affected_components_getter=mock_affected_components,
            recovery_in_progress_checker=mock_recovery_in_progress,
        )

        assert monitor.config is not None
        assert isinstance(monitor.config, RecoveryConfig)


class TestMonitorLifecycle:
    """Test monitor lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_creates_monitoring_task(self, monitor):
        """Should create monitoring task on start."""
        await monitor.start()

        assert monitor.is_running()
        assert monitor._monitoring_task is not None

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, monitor):
        """Should not create duplicate tasks if already running."""
        await monitor.start()
        first_task = monitor._monitoring_task

        await monitor.start()  # Second start
        second_task = monitor._monitoring_task

        assert first_task is second_task

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, monitor):
        """Should cancel monitoring task on stop."""
        await monitor.start()
        assert monitor.is_running()

        await monitor.stop()

        assert not monitor.is_running()
        assert monitor._monitoring_task is None

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, monitor):
        """Should handle multiple stop calls gracefully."""
        await monitor.start()
        await monitor.stop()
        await monitor.stop()  # Second stop should not raise

        assert not monitor.is_running()

    @pytest.mark.asyncio
    async def test_is_running_returns_false_when_not_started(self, monitor):
        """Should return False when not started."""
        assert not monitor.is_running()

    @pytest.mark.asyncio
    async def test_is_running_returns_true_when_active(self, monitor):
        """Should return True when monitoring active."""
        await monitor.start()
        assert monitor.is_running()
        await monitor.stop()


class TestFailureDetection:
    """Test failure detection and recovery initiation."""

    @pytest.mark.asyncio
    async def test_tick_detects_and_initiates_recovery(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should detect failures and initiate recovery."""
        # Setup
        mock_detector.detect_failures.return_value = [FailureType.REDIS_DOWN]

        # Execute
        count = await monitor.tick()

        # Verify
        assert count == 1
        mock_detector.detect_failures.assert_called_once()
        mock_recovery_initiator.assert_called_once()

        # Verify event structure
        call_args = mock_recovery_initiator.call_args
        event = call_args[0][0]
        mode = call_args[0][1]

        assert isinstance(event, FailureEvent)
        assert event.failure_type == FailureType.REDIS_DOWN
        assert event.severity == "critical"
        assert mode == RecoveryMode.AUTOMATIC

    @pytest.mark.asyncio
    async def test_tick_ignores_non_critical_failures(
        self, monitor, mock_detector, mock_recovery_initiator, mock_is_critical
    ):
        """Should not initiate recovery for non-critical failures."""
        # Setup
        mock_detector.detect_failures.return_value = [FailureType.S3_UNAVAILABLE]
        mock_is_critical.return_value = False

        # Execute
        count = await monitor.tick()

        # Verify
        assert count == 0
        mock_recovery_initiator.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_skips_if_recovery_in_progress(
        self, monitor, mock_detector, mock_recovery_initiator, mock_recovery_in_progress
    ):
        """Should not initiate recovery if one already in progress."""
        # Setup
        mock_detector.detect_failures.return_value = [FailureType.REDIS_DOWN]
        mock_recovery_in_progress.return_value = True

        # Execute
        count = await monitor.tick()

        # Verify
        assert count == 0
        mock_recovery_initiator.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_handles_multiple_critical_failures(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should initiate recovery for multiple critical failures."""
        # Setup
        mock_detector.detect_failures.return_value = [
            FailureType.REDIS_DOWN,
            FailureType.POSTGRES_DOWN,
        ]

        # Execute
        count = await monitor.tick()

        # Verify
        assert count == 2
        assert mock_recovery_initiator.call_count == 2

    @pytest.mark.asyncio
    async def test_tick_respects_automatic_recovery_disabled(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should not initiate recovery if automatic recovery disabled."""
        # Setup
        monitor.config.automatic_recovery_enabled = False
        mock_detector.detect_failures.return_value = [FailureType.REDIS_DOWN]

        # Execute
        count = await monitor.tick()

        # Verify
        assert count == 0
        mock_recovery_initiator.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_handles_detector_exceptions(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should handle detector exceptions gracefully."""
        # Setup
        mock_detector.detect_failures.side_effect = RuntimeError("Detector failed")

        # Execute
        count = await monitor.tick()

        # Verify
        assert count == 0
        mock_recovery_initiator.assert_not_called()


class TestMonitoringLoop:
    """Test continuous monitoring loop."""

    @pytest.mark.asyncio
    async def test_monitoring_loop_runs_periodically(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should run detection periodically."""
        # Setup
        mock_detector.detect_failures.return_value = []

        # Start monitoring
        await monitor.start()

        # Let it run a few cycles
        await asyncio.sleep(0.05)

        # Stop monitoring
        await monitor.stop()

        # Verify multiple detection calls
        assert mock_detector.detect_failures.call_count >= 2

    @pytest.mark.asyncio
    async def test_monitoring_loop_initiates_recovery_for_failures(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should initiate recovery when failures detected."""
        # Setup - return failure once, then empty
        mock_detector.detect_failures.side_effect = [
            [FailureType.REDIS_DOWN],
            [],
            [],
        ]

        # Start monitoring
        await monitor.start()

        # Let it run a few cycles
        await asyncio.sleep(0.05)

        # Stop monitoring
        await monitor.stop()

        # Verify recovery initiated
        mock_recovery_initiator.assert_called()

    @pytest.mark.asyncio
    async def test_monitoring_loop_handles_exceptions(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should continue running after exceptions."""
        # Setup - fail first time, succeed after
        mock_detector.detect_failures.side_effect = [
            RuntimeError("Detector failed"),
            [],
            [],
        ]

        # Start monitoring
        await monitor.start()

        # Let it run a few cycles (with backoff)
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop()

        # Verify loop continued after error
        assert mock_detector.detect_failures.call_count >= 2

    @pytest.mark.asyncio
    async def test_monitoring_loop_stops_cleanly_on_cancel(self, monitor, mock_detector):
        """Should stop cleanly when cancelled."""
        # Setup
        mock_detector.detect_failures.return_value = []

        # Start monitoring
        await monitor.start()

        # Let it run briefly
        await asyncio.sleep(0.02)

        # Stop should complete without hanging
        await monitor.stop()

        assert not monitor.is_running()


class TestAffectedComponents:
    """Test affected components tracking."""

    @pytest.mark.asyncio
    async def test_tick_includes_affected_components(
        self, monitor, mock_detector, mock_recovery_initiator, mock_affected_components
    ):
        """Should include affected components in failure event."""
        # Setup
        mock_detector.detect_failures.return_value = [FailureType.REDIS_DOWN]
        mock_affected_components.return_value = ["cache", "hot_storage"]

        # Execute
        await monitor.tick()

        # Verify
        call_args = mock_recovery_initiator.call_args
        event = call_args[0][0]

        assert event.affected_components == ["cache", "hot_storage"]
        mock_affected_components.assert_called_once_with(FailureType.REDIS_DOWN)


class TestEventStructure:
    """Test failure event structure."""

    @pytest.mark.asyncio
    async def test_tick_creates_proper_failure_event(
        self, monitor, mock_detector, mock_recovery_initiator
    ):
        """Should create properly structured failure event."""
        # Setup
        mock_detector.detect_failures.return_value = [FailureType.POSTGRES_DOWN]

        # Execute
        await monitor.tick()

        # Verify event
        call_args = mock_recovery_initiator.call_args
        event = call_args[0][0]
        mode = call_args[0][1]

        assert isinstance(event, FailureEvent)
        assert event.failure_type == FailureType.POSTGRES_DOWN
        assert event.severity == "critical"
        assert event.timestamp is not None
        assert "Automatic detection" in event.error_message
        assert mode == RecoveryMode.AUTOMATIC
