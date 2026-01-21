"""Tests for execution telemetry metrics types and issue tracking."""

import pytest

from gpt_trader.tui.services.execution_telemetry import (
    ExecutionTelemetryCollector,
    clear_execution_telemetry,
)
from gpt_trader.tui.types import ExecutionMetrics


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
        assert metrics.recent_rejections == []
        assert metrics.recent_retries == []

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


class TestIssueTracking:
    """Tests for recent rejection/retry issue tracking."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear singleton before and after each test."""
        clear_execution_telemetry()
        yield
        clear_execution_telemetry()

    def test_rejection_issue_records_context(self):
        """Test that rejection issues capture order context."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0,
            success=False,
            rejected=True,
            rejection_reason="rate_limit",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.5,
            price=30000.0,
        )

        metrics = collector.get_metrics()
        assert len(metrics.recent_rejections) == 1
        issue = metrics.recent_rejections[0]
        assert issue.symbol == "BTC-USD"
        assert issue.side == "BUY"
        assert issue.quantity == 0.5
        assert issue.price == 30000.0
        assert issue.reason == "rate_limit"
        assert issue.is_retry is False

    def test_failed_submission_records_issue(self):
        """Test failed submissions also record an issue."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=80.0,
            success=False,
            failure_reason="Timeout",
            symbol="ETH-USD",
            side="SELL",
            quantity=1.0,
            price=2000.0,
        )

        metrics = collector.get_metrics()
        assert len(metrics.recent_rejections) == 1
        assert metrics.recent_rejections[0].reason == "Timeout"

    def test_retry_issue_records_context(self):
        """Test that retry issues capture order context."""
        collector = ExecutionTelemetryCollector()
        collector.record_retry(
            reason="timeout",
            symbol="SOL-USD",
            side="SELL",
            quantity=2.0,
            price=95.0,
        )
        collector.record_submission(latency_ms=20.0, success=True)

        metrics = collector.get_metrics()
        assert len(metrics.recent_retries) == 1
        issue = metrics.recent_retries[0]
        assert issue.symbol == "SOL-USD"
        assert issue.side == "SELL"
        assert issue.quantity == 2.0
        assert issue.price == 95.0
        assert issue.reason == "timeout"
        assert issue.is_retry is True

    def test_issue_ordering_is_most_recent_first(self):
        """Test that recent issues are ordered newest-first."""
        collector = ExecutionTelemetryCollector()
        collector.record_submission(
            latency_ms=30.0,
            success=False,
            rejected=True,
            rejection_reason="older",
        )
        collector.record_submission(
            latency_ms=30.0,
            success=False,
            rejected=True,
            rejection_reason="newer",
        )

        metrics = collector.get_metrics()
        assert metrics.recent_rejections[0].reason == "newer"
        assert metrics.recent_rejections[1].reason == "older"

    def test_empty_retry_reason_skips_issue(self):
        """Test that empty retry reasons don't create issues."""
        collector = ExecutionTelemetryCollector()
        collector.record_retry(reason="")
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.recent_retries == []
