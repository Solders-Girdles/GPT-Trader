"""Tests for execution telemetry collector core behavior and submission records."""

import pytest

from gpt_trader.tui.services.execution_telemetry import (
    ExecutionTelemetryCollector,
    SubmissionRecord,
    clear_execution_telemetry,
    get_execution_telemetry,
)


class TestSubmissionRecord:
    """Tests for SubmissionRecord dataclass."""

    def test_successful_submission(self):
        """Test creating a successful submission record."""
        record = SubmissionRecord(
            timestamp=1000.0,
            latency_ms=45.5,
            success=True,
        )
        assert record.success is True
        assert record.rejected is False
        assert record.retry_count == 0
        assert record.latency_ms == 45.5

    def test_failed_submission(self):
        """Test creating a failed submission record."""
        record = SubmissionRecord(
            timestamp=1000.0,
            latency_ms=100.0,
            success=False,
            failure_reason="Connection timeout",
        )
        assert record.success is False
        assert record.failure_reason == "Connection timeout"

    def test_rejected_submission(self):
        """Test creating a rejected submission record."""
        record = SubmissionRecord(
            timestamp=1000.0,
            latency_ms=30.0,
            success=False,
            rejected=True,
        )
        assert record.success is False
        assert record.rejected is True


class TestExecutionTelemetryCollector:
    """Tests for ExecutionTelemetryCollector."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear singleton before and after each test."""
        clear_execution_telemetry()
        yield
        clear_execution_telemetry()

    def test_singleton_pattern(self):
        """Test get_execution_telemetry returns same instance."""
        collector1 = get_execution_telemetry()
        collector2 = get_execution_telemetry()
        assert collector1 is collector2

    def test_clear_singleton(self):
        """Test clear_execution_telemetry creates new instance."""
        collector1 = get_execution_telemetry()
        clear_execution_telemetry()
        collector2 = get_execution_telemetry()
        assert collector1 is not collector2

    def test_empty_metrics(self):
        """Test metrics with no submissions."""
        collector = ExecutionTelemetryCollector()
        metrics = collector.get_metrics()
        assert metrics.submissions_total == 0
        assert metrics.success_rate == 100.0

    def test_record_successful_submission(self):
        """Test recording a successful submission."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.submissions_total == 1
        assert metrics.submissions_success == 1
        assert metrics.success_rate == 100.0
        assert metrics.avg_latency_ms == 50.0

    def test_record_failed_submission(self):
        """Test recording a failed submission."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=100.0,
            success=False,
            failure_reason="Timeout",
        )

        metrics = collector.get_metrics()
        assert metrics.submissions_total == 1
        assert metrics.submissions_failed == 1
        assert metrics.success_rate == 0.0
        assert metrics.last_failure_reason == "Timeout"

    def test_record_rejected_submission(self):
        """Test recording a rejected submission."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0,
            success=False,
            rejected=True,
        )

        metrics = collector.get_metrics()
        assert metrics.submissions_total == 1
        assert metrics.submissions_rejected == 1

    def test_multiple_submissions(self):
        """Test recording multiple submissions."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(latency_ms=40.0, success=True)
        collector.record_submission(latency_ms=60.0, success=True)
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.submissions_total == 3
        assert metrics.submissions_success == 3
        assert metrics.avg_latency_ms == 50.0  # (40+60+50)/3
        assert metrics.p50_latency_ms == 50.0

    def test_mixed_success_and_failure(self):
        """Test metrics with mixed success and failure."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(latency_ms=50.0, success=True)
        collector.record_submission(latency_ms=50.0, success=True)
        collector.record_submission(latency_ms=100.0, success=False)
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.submissions_total == 4
        assert metrics.submissions_success == 3
        assert metrics.submissions_failed == 1
        assert metrics.success_rate == 75.0

    def test_retry_tracking(self):
        """Test retry count tracking."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(latency_ms=50.0, success=True, retry_count=2)
        collector.record_submission(latency_ms=50.0, success=True, retry_count=1)
        collector.record_submission(latency_ms=50.0, success=True, retry_count=0)

        metrics = collector.get_metrics()
        assert metrics.retry_total == 3
        assert metrics.retry_rate == 1.0  # 3 retries / 3 submissions

    def test_rolling_window(self):
        """Test that collector uses rolling window."""
        collector = ExecutionTelemetryCollector(window_size=5)

        for _ in range(5):
            collector.record_submission(latency_ms=50.0, success=True)

        for _ in range(5):
            collector.record_submission(latency_ms=50.0, success=False)

        metrics = collector.get_metrics()
        assert metrics.submissions_total == 5  # Window size
        assert metrics.submissions_success == 0  # All pushed out
        assert metrics.submissions_failed == 5

    def test_clear(self):
        """Test clearing metrics."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(latency_ms=50.0, success=True)
        collector.record_submission(latency_ms=50.0, success=False)

        collector.clear()
        metrics = collector.get_metrics()

        assert metrics.submissions_total == 0

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        collector = ExecutionTelemetryCollector()

        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for lat in latencies:
            collector.record_submission(latency_ms=float(lat), success=True)

        metrics = collector.get_metrics()
        assert metrics.p50_latency_ms == 55.0  # Median of 10 values
        assert metrics.p95_latency_ms == 100.0  # 95th percentile

    def test_thread_safety(self):
        """Test that collector is thread-safe."""
        import threading

        collector = ExecutionTelemetryCollector()
        errors = []

        def record_submissions():
            try:
                for _ in range(100):
                    collector.record_submission(latency_ms=50.0, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_submissions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        metrics = collector.get_metrics()
        assert metrics.submissions_total > 0
