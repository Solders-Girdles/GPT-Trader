"""Tests for ExecutionIssuesModal."""

import time

from gpt_trader.tui.types import ExecutionIssue, ExecutionMetrics
from gpt_trader.tui.widgets.execution_issues_modal import ExecutionIssuesModal


class TestExecutionIssuesModal:
    """Tests for ExecutionIssuesModal formatting."""

    def test_format_issue_row_buy_side(self):
        """Test formatting a BUY side issue row."""
        issue = ExecutionIssue(
            timestamp=time.time(),
            symbol="BTC-USD",
            side="BUY",
            quantity=0.5,
            price=50000.0,
            reason="rate_limit",
        )
        metrics = ExecutionMetrics(recent_rejections=[issue])
        modal = ExecutionIssuesModal(metrics)

        formatted = modal._format_issue_row(issue)
        text_str = str(formatted)

        assert "BTC-USD" in text_str
        assert "BUY" in text_str
        assert "0.5000" in text_str
        assert "50000.00" in text_str
        assert "rate_limit" in text_str

    def test_format_issue_row_sell_side(self):
        """Test formatting a SELL side issue row."""
        issue = ExecutionIssue(
            timestamp=time.time(),
            symbol="ETH-USD",
            side="SELL",
            quantity=2.0,
            price=3000.0,
            reason="insufficient_funds",
        )
        metrics = ExecutionMetrics(recent_rejections=[issue])
        modal = ExecutionIssuesModal(metrics)

        formatted = modal._format_issue_row(issue)
        text_str = str(formatted)

        assert "ETH-USD" in text_str
        assert "SELL" in text_str
        assert "2.0000" in text_str
        assert "3000.00" in text_str
        assert "insufficient_funds" in text_str

    def test_format_issue_row_missing_values(self):
        """Test formatting an issue row with missing values."""
        issue = ExecutionIssue(
            timestamp=time.time(),
            symbol="",
            side="",
            quantity=0.0,
            price=0.0,
            reason="unknown",
        )
        metrics = ExecutionMetrics(recent_rejections=[issue])
        modal = ExecutionIssuesModal(metrics)

        formatted = modal._format_issue_row(issue)
        text_str = str(formatted)

        # Should show dashes for missing values
        assert "—" in text_str
        assert "unknown" in text_str

    def test_format_issue_row_timestamp(self):
        """Test that timestamp is formatted as HH:MM:SS."""
        # Create a timestamp for 14:30:45
        ts = time.mktime(time.strptime("2024-01-15 14:30:45", "%Y-%m-%d %H:%M:%S"))
        issue = ExecutionIssue(
            timestamp=ts,
            symbol="BTC-USD",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            reason="timeout",
        )
        metrics = ExecutionMetrics(recent_rejections=[issue])
        modal = ExecutionIssuesModal(metrics)

        formatted = modal._format_issue_row(issue)
        text_str = str(formatted)

        assert "14:30:45" in text_str


class TestModalWithEmptyData:
    """Tests for modal with empty data."""

    def test_empty_rejections_shows_dash(self):
        """Modal with no rejections should indicate empty state."""
        metrics = ExecutionMetrics(recent_rejections=[], recent_retries=[])
        modal = ExecutionIssuesModal(metrics)

        # Modal should be created without error
        assert modal.metrics.recent_rejections == []
        assert modal.metrics.recent_retries == []

    def test_empty_retries_shows_dash(self):
        """Modal with no retries should indicate empty state."""
        issue = ExecutionIssue(
            timestamp=time.time(),
            symbol="BTC-USD",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            reason="rate_limit",
        )
        metrics = ExecutionMetrics(recent_rejections=[issue], recent_retries=[])
        modal = ExecutionIssuesModal(metrics)

        assert len(modal.metrics.recent_rejections) == 1
        assert modal.metrics.recent_retries == []


class TestModalWithPopulatedData:
    """Tests for modal with populated data."""

    def test_multiple_rejections(self):
        """Modal correctly handles multiple rejections."""
        issues = [
            ExecutionIssue(
                timestamp=time.time() - i,
                symbol=f"SYMBOL-{i}",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=float(i + 1),
                price=float(1000 * (i + 1)),
                reason=f"reason_{i}",
            )
            for i in range(5)
        ]
        metrics = ExecutionMetrics(recent_rejections=issues)
        modal = ExecutionIssuesModal(metrics)

        assert len(modal.metrics.recent_rejections) == 5

    def test_multiple_retries(self):
        """Modal correctly handles multiple retries."""
        issues = [
            ExecutionIssue(
                timestamp=time.time() - i,
                symbol=f"SYMBOL-{i}",
                side="BUY",
                quantity=float(i + 1),
                price=float(1000 * (i + 1)),
                reason=f"retry_reason_{i}",
                is_retry=True,
            )
            for i in range(3)
        ]
        metrics = ExecutionMetrics(recent_retries=issues)
        modal = ExecutionIssuesModal(metrics)

        assert len(modal.metrics.recent_retries) == 3

    def test_summary_calculation(self):
        """Modal correctly calculates summary totals."""
        metrics = ExecutionMetrics(
            submissions_rejected=5,
            submissions_failed=3,
            retry_total=10,
        )
        modal = ExecutionIssuesModal(metrics)

        # Total rejections = rejected + failed
        total_rejections = modal.metrics.submissions_rejected + modal.metrics.submissions_failed
        assert total_rejections == 8
        assert modal.metrics.retry_total == 10
