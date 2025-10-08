"""Tests for datetime helper utilities."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from bot_v2.utilities.datetime_helpers import (
    normalize_to_utc,
    to_iso_utc,
    utc_now,
    utc_now_iso,
    utc_timestamp,
)


class TestUtcNow:
    """Test utc_now function."""

    def test_utc_now_returns_timezone_aware(self) -> None:
        """Test utc_now returns timezone-aware datetime."""
        result = utc_now()

        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_utc_now_is_recent(self) -> None:
        """Test utc_now returns recent time."""
        before = datetime.now(UTC)
        result = utc_now()
        after = datetime.now(UTC)

        assert before <= result <= after

    def test_utc_now_consistency(self) -> None:
        """Test utc_now is consistent across calls."""
        result1 = utc_now()
        result2 = utc_now()

        # Should be close (within 1 second), but may be the same if called quickly
        time_diff = abs((result2 - result1).total_seconds())
        assert time_diff < 1.0


class TestUtcNowIso:
    """Test utc_now_iso function."""

    def test_utc_now_iso_returns_string(self) -> None:
        """Test utc_now_iso returns string."""
        result = utc_now_iso()

        assert isinstance(result, str)

    def test_utc_now_iso_format(self) -> None:
        """Test utc_now_iso returns correct ISO format."""
        result = utc_now_iso()

        # Should end with UTC timezone indicator
        assert result.endswith("+00:00")

        # Should be parseable by datetime.fromisoformat
        parsed = datetime.fromisoformat(result.replace("+00:00", "+00:00"))
        assert isinstance(parsed, datetime)
        assert parsed.tzinfo is not None

    def test_utc_now_iso_consistency(self) -> None:
        """Test utc_now_iso is consistent with utc_now."""
        iso_result = utc_now_iso()
        dt_result = utc_now()

        iso_parsed = datetime.fromisoformat(iso_result.replace("+00:00", "+00:00"))

        # Should be very close (within a few seconds)
        time_diff = abs((dt_result - iso_parsed).total_seconds())
        assert time_diff < 1.0


class TestUtcTimestamp:
    """Test utc_timestamp function."""

    def test_utc_timestamp_returns_float(self) -> None:
        """Test utc_timestamp returns float."""
        result = utc_timestamp()

        assert isinstance(result, float)

    def test_utc_timestamp_is_recent(self) -> None:
        """Test utc_timestamp returns recent timestamp."""
        before = datetime.now(UTC).timestamp()
        result = utc_timestamp()
        after = datetime.now(UTC).timestamp()

        assert before <= result <= after

    def test_utc_timestamp_consistency(self) -> None:
        """Test utc_timestamp is consistent with utc_now."""
        ts_result = utc_timestamp()
        dt_result = utc_now()

        dt_timestamp = dt_result.timestamp()

        # Should be very close
        time_diff = abs(ts_result - dt_timestamp)
        assert time_diff < 1.0


class TestToIsoUtc:
    """Test to_iso_utc function."""

    def test_to_iso_utc_with_utc_datetime(self) -> None:
        """Test to_iso_utc with UTC datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        result = to_iso_utc(dt)

        assert result == "2024-01-15T10:30:45+00:00"

    def test_to_iso_utc_with_naive_datetime(self) -> None:
        """Test to_iso_utc with naive datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = to_iso_utc(dt)

        assert result == "2024-01-15T10:30:45+00:00"

    def test_to_iso_utc_with_different_timezone(self) -> None:
        """Test to_iso_utc with different timezone."""
        # EST (UTC-5)
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=est)
        result = to_iso_utc(dt)

        # 10:30 EST = 15:30 UTC
        assert result == "2024-01-15T15:30:45+00:00"

    def test_to_iso_utc_with_positive_timezone(self) -> None:
        """Test to_iso_utc with positive timezone offset."""
        # JST (UTC+9)
        jst = timezone(timedelta(hours=9))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=jst)
        result = to_iso_utc(dt)

        # 10:30 JST = 01:30 UTC (previous day)
        assert result == "2024-01-15T01:30:45+00:00"

    def test_to_iso_utc_with_microseconds(self) -> None:
        """Test to_iso_utc preserves microseconds."""
        dt = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        result = to_iso_utc(dt)

        assert result == "2024-01-15T10:30:45.123456+00:00"

    def test_to_iso_utc_edge_cases(self) -> None:
        """Test to_iso_utc with edge case datetimes."""
        # Epoch start
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)
        result = to_iso_utc(dt)
        assert result == "1970-01-01T00:00:00+00:00"

        # Far future
        dt = datetime(2100, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
        result = to_iso_utc(dt)
        assert result == "2100-12-31T23:59:59.999999+00:00"


class TestNormalizeToUtc:
    """Test normalize_to_utc function."""

    def test_normalize_to_utc_with_utc_datetime(self) -> None:
        """Test normalize_to_utc with UTC datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        result = normalize_to_utc(dt)

        assert result == dt
        assert result.tzinfo == UTC

    def test_normalize_to_utc_with_naive_datetime(self) -> None:
        """Test normalize_to_utc with naive datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = normalize_to_utc(dt)

        assert result == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        assert result.tzinfo == UTC

    def test_normalize_to_utc_with_different_timezone(self) -> None:
        """Test normalize_to_utc with different timezone."""
        # EST (UTC-5)
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=est)
        result = normalize_to_utc(dt)

        # 10:30 EST = 15:30 UTC
        expected = datetime(2024, 1, 15, 15, 30, 45, tzinfo=UTC)
        assert result == expected
        assert result.tzinfo == UTC

    def test_normalize_to_utc_with_positive_timezone(self) -> None:
        """Test normalize_to_utc with positive timezone offset."""
        # JST (UTC+9)
        jst = timezone(timedelta(hours=9))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=jst)
        result = normalize_to_utc(dt)

        # 10:30 JST = 01:30 UTC (previous day)
        expected = datetime(2024, 1, 15, 1, 30, 45, tzinfo=UTC)
        assert result == expected
        assert result.tzinfo == UTC

    def test_normalize_to_utc_preserves_microseconds(self) -> None:
        """Test normalize_to_utc preserves microseconds."""
        dt = datetime(2024, 1, 15, 10, 30, 45, 123456)
        result = normalize_to_utc(dt)

        expected = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        assert result == expected
        assert result.microsecond == 123456

    def test_normalize_to_utc_does_not_modify_original(self) -> None:
        """Test normalize_to_utc doesn't modify original datetime."""
        original_dt = datetime(2024, 1, 15, 10, 30, 45)
        result = normalize_to_utc(original_dt)

        # Original should remain unchanged
        assert original_dt.tzinfo is None
        assert result.tzinfo == UTC


class TestIntegrationWorkflow:
    """Test integration between different functions."""

    def test_roundtrip_datetime_to_iso_and_back(self) -> None:
        """Test roundtrip conversion: datetime -> iso -> datetime."""
        original = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)

        # Convert to ISO
        iso_str = to_iso_utc(original)

        # Parse back from ISO
        parsed = datetime.fromisoformat(iso_str.replace("+00:00", "+00:00"))

        # Normalize to UTC
        normalized = normalize_to_utc(parsed)

        assert normalized == original

    def test_all_functions_use_utc(self) -> None:
        """Test all functions consistently use UTC."""
        # Get current time using different methods
        dt_now = utc_now()
        iso_now = utc_now_iso()
        ts_now = utc_timestamp()

        # Parse ISO back to datetime
        dt_from_iso = datetime.fromisoformat(iso_now.replace("+00:00", "+00:00"))
        dt_from_ts = datetime.fromtimestamp(ts_now, tz=UTC)

        # Normalize all to UTC
        dt_from_iso_norm = normalize_to_utc(dt_from_iso)

        # All should be timezone-aware and in UTC
        assert dt_now.tzinfo == UTC
        assert dt_from_iso_norm.tzinfo == UTC
        assert dt_from_ts.tzinfo == UTC

        # All should be close in time
        assert abs((dt_from_iso_norm - dt_now).total_seconds()) < 1.0
        assert abs((dt_from_ts - dt_now).total_seconds()) < 1.0

    def test_timezone_conversion_consistency(self) -> None:
        """Test timezone conversion is consistent across functions."""
        # Create datetime in different timezone
        est = timezone(timedelta(hours=-5))
        original = datetime(2024, 1, 15, 10, 30, 45, tzinfo=est)

        # Convert using different methods
        iso_result = to_iso_utc(original)
        normalized_result = normalize_to_utc(original)
        iso_from_normalized = to_iso_utc(normalized_result)

        # All should represent the same moment in time
        assert iso_result == iso_from_normalized
        assert normalized_result.tzinfo == UTC

        # Parse ISO and compare
        parsed_iso = datetime.fromisoformat(iso_result.replace("+00:00", "+00:00"))
        parsed_iso_norm = normalize_to_utc(parsed_iso)

        assert parsed_iso_norm == normalized_result
