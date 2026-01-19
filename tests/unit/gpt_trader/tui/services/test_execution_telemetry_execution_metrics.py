"""Tests for execution telemetry metrics types."""

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
