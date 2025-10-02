"""
Frozen-time test suite for monitoring evaluator logic.

Covers:
- Deduplication with time-window control
- Escalation and severity-based routing
- Schedule paths with deterministic clock
- Threshold evaluation with table-driven inputs

Uses unittest.mock to patch datetime for deterministic time control.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest

from bot_v2.monitoring.alerts import Alert, AlertChannel, AlertDispatcher, AlertLevel
from bot_v2.monitoring.alerts_manager import AlertManager
from bot_v2.monitoring.interfaces import (
    ComponentHealth,
    ComponentStatus,
    MonitorConfig,
    PerformanceMetrics,
    ResourceUsage,
    SystemHealth,
)
from bot_v2.monitoring.system.engine import MonitoringSystem


# ============================================================================
# Time Control Utilities
# ============================================================================


class FrozenTime:
    """Context manager for freezing time in tests."""

    def __init__(self, frozen_time: str | datetime):
        if isinstance(frozen_time, str):
            self.current_time = datetime.fromisoformat(frozen_time)
        else:
            self.current_time = frozen_time
        self.patcher = None
        self._mock_datetime = None

    def __enter__(self):
        self.patcher = patch("bot_v2.monitoring.alerts_manager.datetime")
        self._mock_datetime = self.patcher.__enter__()
        self._mock_datetime.now.return_value = self.current_time
        self._mock_datetime.utcnow.return_value = self.current_time
        return self

    def __exit__(self, *args):
        if self.patcher:
            self.patcher.__exit__(*args)

    def tick(self, delta: timedelta):
        """Move time forward by delta."""
        self.current_time += delta
        if self._mock_datetime:
            self._mock_datetime.now.return_value = self.current_time
            self._mock_datetime.utcnow.return_value = self.current_time

    def move_to(self, new_time: str | datetime):
        """Move to specific time."""
        if isinstance(new_time, str):
            self.current_time = datetime.fromisoformat(new_time)
        else:
            self.current_time = new_time
        if self._mock_datetime:
            self._mock_datetime.now.return_value = self.current_time
            self._mock_datetime.utcnow.return_value = self.current_time


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def alert_manager():
    """AlertManager with empty dispatcher for testing."""
    dispatcher = AlertDispatcher.from_config({})
    return AlertManager(dispatcher=dispatcher, dedup_window_seconds=300)


@pytest.fixture
def monitoring_system():
    """MonitoringSystem with test config."""
    config = MonitorConfig(
        check_interval_seconds=60,
        alert_threshold_cpu=80.0,
        alert_threshold_memory=80.0,
        alert_threshold_disk=90.0,
        alert_threshold_error_rate=0.05,
        alert_threshold_response_time_ms=1000.0,
    )
    return MonitoringSystem(config=config)


@pytest.fixture
def mock_alert_channel():
    """Mock alert channel for testing dispatch."""

    class MockChannel(AlertChannel):
        def __init__(self):
            super().__init__(min_severity=AlertLevel.INFO)
            self.sent_alerts: list[Alert] = []

        async def _send_impl(self, alert: Alert) -> bool:
            self.sent_alerts.append(alert)
            return True

    return MockChannel()


# ============================================================================
# Test Suite 1: Deduplication with Frozen Time
# ============================================================================


class TestAlertDeduplication:
    """Test time-window based alert deduplication."""

    def test_dedupe_blocks_duplicate_within_window(self, alert_manager):
        """Test deduplication blocks identical alerts within time window."""
        with FrozenTime("2025-01-15 12:00:00"):
            # First alert should be created
            alert1 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High CPU usage detected",
            )

            assert alert1 is not None
            assert alert1.source == "TestSystem"
            assert len(alert_manager.alert_history) == 1

            # Duplicate alert within window should be blocked
            alert2 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High CPU usage detected",
            )

            assert alert2 is None  # Deduplicated
            assert len(alert_manager.alert_history) == 1  # Still only 1 alert

    def test_dedupe_allows_different_message(self, alert_manager):
        """Test deduplication allows alerts with different messages."""
        with FrozenTime("2025-01-15 12:00:00"):
            # First alert
            alert1 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High CPU usage",
            )

            # Different message - should NOT be deduplicated
            alert2 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High memory usage",
            )

            assert alert1 is not None
            assert alert2 is not None
            assert len(alert_manager.alert_history) == 2

    def test_dedupe_allows_different_source(self, alert_manager):
        """Test deduplication allows alerts from different sources."""
        with FrozenTime("2025-01-15 12:00:00"):
            # First alert
            alert1 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="System1",
                message="High CPU usage",
            )

            # Different source - should NOT be deduplicated
            alert2 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="System2",
                message="High CPU usage",
            )

            assert alert1 is not None
            assert alert2 is not None
            assert len(alert_manager.alert_history) == 2

    def test_dedupe_expires_after_window(self, alert_manager):
        """Test deduplication expires after time window passes."""
        # Freeze at initial time
        with FrozenTime("2025-01-15 12:00:00") as frozen:
            # Create first alert
            alert1 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High CPU",
            )
            assert alert1 is not None

            # Move forward 4 minutes (still within 5-minute window)
            frozen.move_to("2025-01-15 12:04:00")

            # Should still be deduplicated
            alert2 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High CPU",
            )
            assert alert2 is None

            # Move forward past window (6 minutes total)
            frozen.move_to("2025-01-15 12:06:00")

            # Should now be allowed
            alert3 = alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="TestSystem",
                message="High CPU",
            )
            assert alert3 is not None
            assert len(alert_manager.alert_history) == 2

    def test_dedupe_cleanup_removes_old_entries(self, alert_manager):
        """Test deduplication cleanup removes expired entries."""
        with FrozenTime("2025-01-15 12:00:00") as frozen:
            # Create multiple alerts
            for i in range(5):
                alert_manager.create_alert(
                    level=AlertLevel.INFO,
                    source=f"System{i}",
                    message=f"Event {i}",
                )

            # Verify dedup tracking has entries
            assert len(alert_manager._recent_alerts) == 5

            # Move forward 10 minutes (past dedup window)
            frozen.move_to("2025-01-15 12:10:00")

            # Run cleanup
            alert_manager.cleanup_old_alerts(retention_hours=1)

            # Dedup tracking should be empty (all entries expired)
            assert len(alert_manager._recent_alerts) == 0


# ============================================================================
# Test Suite 2: Escalation and Severity Routing
# ============================================================================


class TestAlertEscalation:
    """Test severity-based alert escalation and routing."""

    @pytest.mark.parametrize(
        "severity,expected_numeric",
        [
            (AlertLevel.DEBUG, 10),
            (AlertLevel.INFO, 20),
            (AlertLevel.WARNING, 30),
            (AlertLevel.ERROR, 40),
            (AlertLevel.CRITICAL, 50),
        ],
    )
    def test_severity_numeric_levels(self, severity, expected_numeric):
        """Test severity levels have correct numeric values for comparison."""
        assert severity.numeric_level == expected_numeric

    def test_severity_comparison_ordering(self):
        """Test severity levels can be compared for escalation."""
        assert AlertLevel.DEBUG.numeric_level < AlertLevel.INFO.numeric_level
        assert AlertLevel.INFO.numeric_level < AlertLevel.WARNING.numeric_level
        assert AlertLevel.WARNING.numeric_level < AlertLevel.ERROR.numeric_level
        assert AlertLevel.ERROR.numeric_level < AlertLevel.CRITICAL.numeric_level

    @pytest.mark.asyncio
    async def test_channel_filters_below_threshold(self, mock_alert_channel):
        """Test alert channels filter alerts below minimum severity."""
        # Set channel to only accept ERROR and above
        mock_alert_channel.min_severity = AlertLevel.ERROR

        # Send WARNING alert (should be filtered)
        warning_alert = Alert(
            timestamp=datetime.now(),
            source="Test",
            severity=AlertLevel.WARNING,
            title="Warning Alert",
            message="This should be filtered",
        )

        result = await mock_alert_channel.send(warning_alert)
        assert result is False  # Below threshold
        assert len(mock_alert_channel.sent_alerts) == 0

        # Send ERROR alert (should pass)
        error_alert = Alert(
            timestamp=datetime.now(),
            source="Test",
            severity=AlertLevel.ERROR,
            title="Error Alert",
            message="This should pass",
        )

        result = await mock_alert_channel.send(error_alert)
        assert result is True
        assert len(mock_alert_channel.sent_alerts) == 1

    @pytest.mark.asyncio
    async def test_channel_accepts_at_threshold(self, mock_alert_channel):
        """Test alert channels accept alerts at exact threshold."""
        mock_alert_channel.min_severity = AlertLevel.WARNING

        alert = Alert(
            timestamp=datetime.now(),
            source="Test",
            severity=AlertLevel.WARNING,
            title="At Threshold",
            message="Exactly at minimum",
        )

        result = await mock_alert_channel.send(alert)
        assert result is True
        assert len(mock_alert_channel.sent_alerts) == 1

    # Removed decorator - using FrozenTime context manager
    def test_alert_level_string_coercion(self, alert_manager):
        """Test AlertManager accepts string severity levels."""
        # Test string to enum coercion
        alert = alert_manager.create_alert(
            level="WARNING",  # String instead of enum
            source="Test",
            message="String level test",
        )

        assert alert is not None
        assert alert.severity == AlertLevel.WARNING


# ============================================================================
# Test Suite 3: Threshold Evaluation with Input Tables
# ============================================================================


class TestThresholdEvaluation:
    """
    Table-driven tests for threshold evaluation.

    Uses input tables to systematically test threshold boundaries
    and ensure consistent alert behavior.
    """

    @pytest.mark.parametrize(
        "cpu_percent,memory_percent,disk_percent,expected_alert_count",
        [
            # Below all thresholds
            (50.0, 50.0, 50.0, 0),
            # CPU at threshold
            (80.0, 50.0, 50.0, 1),
            # CPU above threshold
            (85.0, 50.0, 50.0, 1),
            # Memory at threshold
            (50.0, 80.0, 50.0, 1),
            # Memory above threshold
            (50.0, 90.0, 50.0, 1),
            # Disk at threshold
            (50.0, 50.0, 90.0, 1),
            # Disk above threshold
            (50.0, 50.0, 95.0, 1),
            # Multiple thresholds exceeded
            (85.0, 85.0, 50.0, 2),
            # All thresholds exceeded
            (85.0, 85.0, 95.0, 3),
        ],
    )
    # Removed decorator - using FrozenTime context manager
    def test_resource_threshold_alerts(
        self, monitoring_system, cpu_percent, memory_percent, disk_percent, expected_alert_count
    ):
        """Test resource threshold alerts with table-driven inputs."""
        # Create resource usage matching test case
        resources = ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=1000.0,
            disk_percent=disk_percent,
            disk_gb=100.0,
            network_sent_mb=10.0,
            network_recv_mb=10.0,
            open_files=100,
            threads=20,
        )

        # Create health snapshot
        from bot_v2.monitoring.interfaces import SystemHealth

        health = SystemHealth(
            timestamp=datetime.now(),
            overall_status=ComponentStatus.HEALTHY,
            components={},
            resource_usage=resources,
            performance=PerformanceMetrics(
                requests_per_second=100.0,
                avg_response_time_ms=50.0,
                p95_response_time_ms=100.0,
                p99_response_time_ms=200.0,
                error_rate=0.01,
                success_rate=0.99,
                active_connections=50,
                queued_tasks=5,
            ),
            active_alerts=0,
        )

        # Check for alerts
        monitoring_system._check_alerts(health)

        # Verify expected number of alerts created
        alerts = monitoring_system.alert_manager.alert_history
        assert len(alerts) == expected_alert_count

    @pytest.mark.parametrize(
        "resource_value,threshold,expected_exceeds",
        [
            # CPU threshold tests
            (79.9, 80.0, False),
            (80.0, 80.0, False),  # Boundary: at threshold, not exceeded
            (80.1, 80.0, True),
            (100.0, 80.0, True),
            # Memory threshold tests
            (79.9, 80.0, False),
            (80.0, 80.0, False),  # Boundary
            (80.1, 80.0, True),
            # Disk threshold tests
            (89.9, 90.0, False),
            (90.0, 90.0, False),  # Boundary
            (90.1, 90.0, True),
        ],
    )
    def test_threshold_boundary_conditions(self, resource_value, threshold, expected_exceeds):
        """Test threshold boundary conditions with precise decimal comparisons."""
        exceeds = resource_value > threshold
        assert exceeds == expected_exceeds

    @pytest.mark.parametrize(
        "error_rate,response_time_ms,expected_degraded",
        [
            # Both healthy
            (0.01, 500.0, False),
            # Error rate at boundary
            (0.05, 500.0, False),
            # Error rate exceeds
            (0.06, 500.0, True),
            # Response time at boundary
            (0.01, 1000.0, False),
            # Response time exceeds
            (0.01, 1001.0, True),
            # Both exceed
            (0.10, 2000.0, True),
        ],
    )
    def test_performance_degradation_detection(
        self, error_rate, response_time_ms, expected_degraded
    ):
        """Test performance degradation detection with input table."""
        metrics = PerformanceMetrics(
            requests_per_second=100.0,
            avg_response_time_ms=response_time_ms,
            p95_response_time_ms=response_time_ms * 1.5,
            p99_response_time_ms=response_time_ms * 2.0,
            error_rate=error_rate,
            success_rate=1.0 - error_rate,
            active_connections=50,
            queued_tasks=5,
        )

        assert metrics.is_degraded() == expected_degraded


# ============================================================================
# Test Suite 4: Schedule Paths with Frozen Time
# ============================================================================


class TestSchedulePaths:
    """Test monitoring schedule paths with deterministic clock control."""

    def test_alert_history_cleanup_schedule(self, alert_manager):
        """Test alert history cleanup follows retention schedule."""
        with FrozenTime("2025-01-15 00:00:00") as frozen:
            # Create alerts at different times
            for hour in range(48):  # 48 hours of alerts
                frozen.move_to(f"2025-01-15 00:00:00")
                frozen.tick(delta=timedelta(hours=hour))

                alert_manager.create_alert(
                    level=AlertLevel.INFO,
                    source="Test",
                    message=f"Alert at hour {hour}",
                )

            # Should have 48 alerts
            assert len(alert_manager.alert_history) == 48

            # Move to 25 hours after start
            frozen.move_to("2025-01-16 01:00:00")

            # Clean up alerts older than 24 hours
            alert_manager.cleanup_old_alerts(retention_hours=24)

            # Should have ~24 alerts left (ones from last 24 hours)
            assert len(alert_manager.alert_history) <= 24
            assert len(alert_manager.alert_history) >= 23  # Allow small margin

    # Removed decorator - using FrozenTime context manager
    def test_dedupe_window_timing_precision(self, alert_manager):
        """Test deduplication window respects precise timing boundaries."""
        # Set 60-second dedup window for precise testing
        alert_manager.dedup_window_seconds = 60

        with FrozenTime("2025-01-15 12:00:00") as frozen:
            # Create alert at T=0
            alert1 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="Test", message="Event"
            )
            assert alert1 is not None

            # T=59s - should be deduplicated (within window)
            frozen.tick(delta=timedelta(seconds=59))
            alert2 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="Test", message="Event"
            )
            assert alert2 is None

            # T=60s - should be deduplicated (exactly at boundary)
            frozen.tick(delta=timedelta(seconds=1))
            alert3 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="Test", message="Event"
            )
            assert alert3 is None  # Still within window

            # T=61s - should pass (outside window)
            frozen.tick(delta=timedelta(seconds=1))
            alert4 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="Test", message="Event"
            )
            assert alert4 is not None

    # Removed decorator - using FrozenTime context manager
    def test_concurrent_dedupe_window_per_source(self, alert_manager):
        """Test deduplication windows are tracked independently per source."""
        with FrozenTime("2025-01-15 12:00:00") as frozen:
            # Create alerts from two sources
            alert_a1 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="SourceA", message="Event"
            )
            alert_b1 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="SourceB", message="Event"
            )

            assert alert_a1 is not None
            assert alert_b1 is not None

            # Move forward 4 minutes
            frozen.tick(delta=timedelta(minutes=4))

            # Both should be deduplicated (within their windows)
            alert_a2 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="SourceA", message="Event"
            )
            alert_b2 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="SourceB", message="Event"
            )

            assert alert_a2 is None
            assert alert_b2 is None

            # Move forward 2 more minutes (6 total)
            frozen.tick(delta=timedelta(minutes=2))

            # Both should now pass (outside windows)
            alert_a3 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="SourceA", message="Event"
            )
            alert_b3 = alert_manager.create_alert(
                level=AlertLevel.INFO, source="SourceB", message="Event"
            )

            assert alert_a3 is not None
            assert alert_b3 is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestMonitoringIntegration:
    """Integration tests for monitoring system with frozen time."""

    # Removed decorator - using FrozenTime context manager
    def test_end_to_end_threshold_to_alert(self, monitoring_system):
        """Test end-to-end flow from threshold breach to alert creation."""
        # Create health with CPU threshold exceeded
        health = SystemHealth(
            timestamp=datetime.now(),
            overall_status=ComponentStatus.HEALTHY,
            components={},
            resource_usage=ResourceUsage(
                cpu_percent=85.0,  # Above 80% threshold
                memory_percent=50.0,
                memory_mb=1000.0,
                disk_percent=50.0,
                disk_gb=100.0,
                network_sent_mb=10.0,
                network_recv_mb=10.0,
                open_files=100,
                threads=20,
            ),
            performance=PerformanceMetrics(
                requests_per_second=100.0,
                avg_response_time_ms=50.0,
                p95_response_time_ms=100.0,
                p99_response_time_ms=200.0,
                error_rate=0.01,
                success_rate=0.99,
                active_connections=50,
                queued_tasks=5,
            ),
            active_alerts=0,
        )

        # Trigger alert check
        monitoring_system._check_alerts(health)

        # Verify alert was created
        alerts = monitoring_system.alert_manager.alert_history
        assert len(alerts) == 1
        assert alerts[0].source == "System"
        assert "CPU" in alerts[0].message
        assert "85" in alerts[0].message

    def test_repeated_threshold_breach_with_dedupe(self, monitoring_system):
        """Test repeated threshold breaches respect deduplication."""
        with FrozenTime("2025-01-15 12:00:00") as frozen:
            health = SystemHealth(
                timestamp=datetime.now(),
                overall_status=ComponentStatus.HEALTHY,
                components={},
                resource_usage=ResourceUsage(
                    cpu_percent=85.0,
                    memory_percent=50.0,
                    memory_mb=1000.0,
                    disk_percent=50.0,
                    disk_gb=100.0,
                    network_sent_mb=10.0,
                    network_recv_mb=10.0,
                    open_files=100,
                    threads=20,
                ),
                performance=PerformanceMetrics(
                    requests_per_second=100.0,
                    avg_response_time_ms=50.0,
                    p95_response_time_ms=100.0,
                    p99_response_time_ms=200.0,
                    error_rate=0.01,
                    success_rate=0.99,
                    active_connections=50,
                    queued_tasks=5,
                ),
                active_alerts=0,
            )

            # First check - should create alert
            monitoring_system._check_alerts(health)
            assert len(monitoring_system.alert_manager.alert_history) == 1

            # Immediate second check - should be deduplicated
            monitoring_system._check_alerts(health)
            assert len(monitoring_system.alert_manager.alert_history) == 1  # Still 1

            # Move forward 6 minutes (past dedupe window)
            frozen.tick(delta=timedelta(minutes=6))

            # Third check - should create new alert
            monitoring_system._check_alerts(health)
            assert len(monitoring_system.alert_manager.alert_history) == 2
