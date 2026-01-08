"""Unit tests for the profiling module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from gpt_trader.monitoring.profiling import (
    ProfileSample,
    profile_span,
    record_profile,
)


class TestProfileSample:
    """Tests for ProfileSample dataclass."""

    def test_profile_sample_creation(self) -> None:
        """Test creating a ProfileSample with required fields."""
        sample = ProfileSample(name="test_op", duration_ms=100.5)
        assert sample.name == "test_op"
        assert sample.duration_ms == 100.5
        assert sample.labels is None
        assert isinstance(sample.timestamp, datetime)

    def test_profile_sample_with_labels(self) -> None:
        """Test creating a ProfileSample with labels."""
        labels = {"symbol": "BTC-USD", "result": "ok"}
        sample = ProfileSample(name="fetch", duration_ms=50.0, labels=labels)
        assert sample.name == "fetch"
        assert sample.labels == labels

    def test_profile_sample_to_dict(self) -> None:
        """Test serialization to dictionary."""
        timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        sample = ProfileSample(
            name="test",
            duration_ms=25.5,
            timestamp=timestamp,
            labels={"key": "value"},
        )
        result = sample.to_dict()
        assert result["name"] == "test"
        assert result["duration_ms"] == 25.5
        assert result["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert result["labels"] == {"key": "value"}


class TestRecordProfile:
    """Tests for record_profile function."""

    def test_record_profile_calls_histogram(self) -> None:
        """Test that record_profile calls record_histogram."""
        with patch("gpt_trader.monitoring.profiling.record_histogram") as mock_histogram:
            sample = record_profile("my_operation", 150.0)

            mock_histogram.assert_called_once_with(
                "gpt_trader_profile_duration_seconds",
                0.15,  # 150ms -> 0.15s
                labels={"phase": "my_operation"},
            )
            assert sample.name == "my_operation"
            assert sample.duration_ms == 150.0

    def test_record_profile_with_labels(self) -> None:
        """Test record_profile merges custom labels with phase label."""
        with patch("gpt_trader.monitoring.profiling.record_histogram") as mock_histogram:
            sample = record_profile(
                "order_submit",
                75.5,
                labels={"symbol": "ETH-USD"},
            )

            mock_histogram.assert_called_once_with(
                "gpt_trader_profile_duration_seconds",
                0.0755,
                labels={"phase": "order_submit", "symbol": "ETH-USD"},
            )
            assert sample.labels == {"symbol": "ETH-USD"}


class TestProfileSpan:
    """Tests for profile_span context manager."""

    def test_profile_span_records_duration(self) -> None:
        """Test that profile_span records duration on exit."""
        with patch("gpt_trader.monitoring.profiling.record_histogram") as mock_histogram:
            with profile_span("test_span") as sample:
                assert sample is not None
                assert sample.duration_ms == 0.0  # Not yet updated
                # Do some trivial work to ensure non-zero duration
                _ = sum(range(100))

            # After exiting, duration should be updated (even if tiny)
            assert sample.duration_ms >= 0
            # Histogram should be called
            assert mock_histogram.called
            call_args = mock_histogram.call_args
            assert call_args[0][0] == "gpt_trader_profile_duration_seconds"
            assert call_args[1]["labels"] == {"phase": "test_span"}

    def test_profile_span_with_labels(self) -> None:
        """Test profile_span with custom labels."""
        with patch("gpt_trader.monitoring.profiling.record_histogram") as mock_histogram:
            with profile_span("fetch", {"symbol": "BTC-USD"}):
                pass

            call_args = mock_histogram.call_args
            assert call_args[1]["labels"] == {"phase": "fetch", "symbol": "BTC-USD"}

    def test_profile_span_tolerates_exceptions(self) -> None:
        """Test that profile_span still records metrics when exception occurs."""
        with patch("gpt_trader.monitoring.profiling.record_histogram") as mock_histogram:
            with pytest.raises(ValueError, match="test error"):
                with profile_span("failing_op") as sample:
                    raise ValueError("test error")

            # Histogram should still be called even after exception
            assert mock_histogram.called
            call_args = mock_histogram.call_args
            assert call_args[0][0] == "gpt_trader_profile_duration_seconds"
            assert call_args[1]["labels"]["phase"] == "failing_op"
            # Duration should be recorded
            assert sample is not None
            assert sample.duration_ms > 0

    def test_profile_span_updates_sample_timestamp(self) -> None:
        """Test that sample timestamp is updated on exit."""
        with patch("gpt_trader.monitoring.profiling.record_histogram"):
            before = datetime.now(timezone.utc)
            with profile_span("time_test") as sample:
                # Do trivial work
                _ = sum(range(100))
            after = datetime.now(timezone.utc)

            assert sample is not None
            assert before <= sample.timestamp <= after

    def test_profile_span_measures_duration(self) -> None:
        """Test that duration measurement captures elapsed time."""
        with patch("gpt_trader.monitoring.profiling.record_histogram"):
            with profile_span("duration_test") as sample:
                # Do some work to ensure measurable duration
                _ = [x**2 for x in range(1000)]

            assert sample is not None
            # Duration should be non-negative
            assert sample.duration_ms >= 0
