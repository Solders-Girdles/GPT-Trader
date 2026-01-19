"""Tests for execution telemetry submission records."""

from gpt_trader.tui.services.execution_telemetry import SubmissionRecord


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
