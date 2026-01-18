"""Tests for execution telemetry rejection/retry reason tracking."""

import pytest

from gpt_trader.tui.services.execution_telemetry import (
    ExecutionTelemetryCollector,
    clear_execution_telemetry,
)


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
        top = metrics.top_rejection_reasons
        assert top[0] == ("rate_limit", 2)
        assert top[1] == ("insufficient_funds", 1)

    def test_record_retry_with_reason(self):
        """Test recording retry with reason."""
        collector = ExecutionTelemetryCollector()
        collector.record_retry(reason="timeout")
        collector.record_retry(reason="network")
        collector.record_retry(reason="timeout")

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

        for _ in range(3):
            collector.record_submission(
                latency_ms=30.0,
                success=False,
                rejected=True,
                rejection_reason="rate_limit",
            )

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"rate_limit": 3}

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
        assert metrics.recent_rejections == []
        assert metrics.recent_retries == []
