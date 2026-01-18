"""Tests for execution telemetry issue tracking."""

import pytest

from gpt_trader.tui.services.execution_telemetry import (
    ExecutionTelemetryCollector,
    clear_execution_telemetry,
)


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
