"""Tests for execution telemetry collector."""

import pytest

from gpt_trader.tui.services.execution_telemetry import (
    ExecutionTelemetryCollector,
    SubmissionRecord,
    clear_execution_telemetry,
    get_execution_telemetry,
)
from gpt_trader.tui.types import ExecutionMetrics


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


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_default_values(self):
        """Test default values for execution metrics."""
        metrics = ExecutionMetrics()
        assert metrics.submissions_total == 0
        assert metrics.submissions_success == 0
        assert metrics.success_rate == 100.0  # No submissions = 100%
        assert metrics.is_healthy is True
        assert metrics.rejection_reasons == {}
        assert metrics.retry_reasons == {}

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ExecutionMetrics(
            submissions_total=10,
            submissions_success=8,
        )
        assert metrics.success_rate == 80.0

    def test_success_rate_zero_submissions(self):
        """Test success rate with zero submissions."""
        metrics = ExecutionMetrics(submissions_total=0)
        assert metrics.success_rate == 100.0

    def test_is_healthy_good_metrics(self):
        """Test is_healthy with good metrics."""
        metrics = ExecutionMetrics(
            submissions_total=100,
            submissions_success=98,
            retry_rate=0.2,
        )
        assert metrics.is_healthy is True

    def test_is_healthy_low_success_rate(self):
        """Test is_healthy with low success rate."""
        metrics = ExecutionMetrics(
            submissions_total=100,
            submissions_success=80,  # 80% < 95%
            retry_rate=0.2,
        )
        assert metrics.is_healthy is False

    def test_is_healthy_high_retry_rate(self):
        """Test is_healthy with high retry rate."""
        metrics = ExecutionMetrics(
            submissions_total=100,
            submissions_success=98,
            retry_rate=0.6,  # > 0.5
        )
        assert metrics.is_healthy is False

    def test_top_rejection_reasons_sorted(self):
        """Test top_rejection_reasons returns sorted by count."""
        metrics = ExecutionMetrics(
            rejection_reasons={"rate_limit": 5, "insufficient_funds": 2, "timeout": 8}
        )
        top = metrics.top_rejection_reasons
        assert top[0] == ("timeout", 8)
        assert top[1] == ("rate_limit", 5)
        assert top[2] == ("insufficient_funds", 2)

    def test_top_rejection_reasons_empty(self):
        """Test top_rejection_reasons with no rejections."""
        metrics = ExecutionMetrics()
        assert metrics.top_rejection_reasons == []

    def test_top_retry_reasons_sorted(self):
        """Test top_retry_reasons returns sorted by count."""
        metrics = ExecutionMetrics(retry_reasons={"timeout": 3, "network": 7, "rate_limit": 1})
        top = metrics.top_retry_reasons
        assert top[0] == ("network", 7)
        assert top[1] == ("timeout", 3)
        assert top[2] == ("rate_limit", 1)


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

        # Add 5 successful submissions
        for _ in range(5):
            collector.record_submission(latency_ms=50.0, success=True)

        # Add 5 failed submissions (should push out successful ones)
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

        # Add submissions with known latencies
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
        # Should have recorded all submissions (window may cap total)
        assert metrics.submissions_total > 0


class TestReasonTracking:
    """Tests for rejection and retry reason tracking."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear singleton before and after each test."""
        clear_execution_telemetry()
        yield
        clear_execution_telemetry()

    def test_record_rejection_reason(self):
        """Test recording submission with rejection reason."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0,
            success=False,
            rejected=True,
            rejection_reason="rate_limit",
        )

        metrics = collector.get_metrics()
        assert metrics.submissions_rejected == 1
        assert metrics.rejection_reasons == {"rate_limit": 1}

    def test_multiple_rejection_reasons(self):
        """Test aggregating multiple rejection reasons."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0, success=False, rejected=True, rejection_reason="rate_limit"
        )
        collector.record_submission(
            latency_ms=30.0,
            success=False,
            rejected=True,
            rejection_reason="insufficient_funds",
        )
        collector.record_submission(
            latency_ms=30.0, success=False, rejected=True, rejection_reason="rate_limit"
        )

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"rate_limit": 2, "insufficient_funds": 1}
        # Verify sorted order via property
        top = metrics.top_rejection_reasons
        assert top[0] == ("rate_limit", 2)
        assert top[1] == ("insufficient_funds", 1)

    def test_record_retry_with_reason(self):
        """Test recording retry with reason."""
        collector = ExecutionTelemetryCollector()
        collector.record_retry(reason="timeout")
        collector.record_retry(reason="network")
        collector.record_retry(reason="timeout")

        metrics = collector.get_metrics()
        # Note: get_metrics returns empty if no submissions
        # Add a submission to get metrics
        collector.record_submission(latency_ms=50.0, success=True)
        metrics = collector.get_metrics()
        assert metrics.retry_reasons == {"timeout": 2, "network": 1}

    def test_retry_reasons_sorted(self):
        """Test retry reasons are sorted by count."""
        collector = ExecutionTelemetryCollector()
        collector.record_retry(reason="network")
        collector.record_retry(reason="network")
        collector.record_retry(reason="network")
        collector.record_retry(reason="timeout")
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        top = metrics.top_retry_reasons
        assert top[0] == ("network", 3)
        assert top[1] == ("timeout", 1)

    def test_empty_reason_not_tracked(self):
        """Test that empty reasons are not tracked."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0, success=False, rejected=True, rejection_reason=""
        )
        collector.record_retry(reason="")

        collector.record_submission(latency_ms=50.0, success=True)
        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {}
        assert metrics.retry_reasons == {}

    def test_reasons_in_rolling_window(self):
        """Test that reasons respect rolling window."""
        collector = ExecutionTelemetryCollector(window_size=3)

        # Add 3 rejections with "rate_limit"
        for _ in range(3):
            collector.record_submission(
                latency_ms=30.0,
                success=False,
                rejected=True,
                rejection_reason="rate_limit",
            )

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"rate_limit": 3}

        # Add 3 more with "timeout" - should push out rate_limit
        for _ in range(3):
            collector.record_submission(
                latency_ms=30.0,
                success=False,
                rejected=True,
                rejection_reason="timeout",
            )

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"timeout": 3}
        assert "rate_limit" not in metrics.rejection_reasons

    def test_clear_clears_reasons(self):
        """Test that clear() also clears reason tracking."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0, success=False, rejected=True, rejection_reason="rate_limit"
        )
        collector.record_retry(reason="timeout")

        collector.clear()
        collector.record_submission(latency_ms=50.0, success=True)
        metrics = collector.get_metrics()

        assert metrics.rejection_reasons == {}
        assert metrics.retry_reasons == {}
