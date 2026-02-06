"""Tests for datetime helper utilities."""

from datetime import UTC, datetime, timedelta, timezone

from gpt_trader.utilities.datetime_helpers import (
    age_since_timestamp_seconds,
    normalize_to_utc,
    to_epoch_seconds,
    to_iso_utc,
    utc_now,
    utc_now_iso,
    utc_timestamp,
)


class TestDateTimeHelpers:
    """Test consistency between datetime helper functions."""

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
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=est)
        result = normalize_to_utc(dt)

        expected = datetime(2024, 1, 15, 15, 30, 45, tzinfo=UTC)
        assert result == expected
        assert result.tzinfo == UTC

    def test_normalize_to_utc_with_positive_timezone(self) -> None:
        """Test normalize_to_utc with positive timezone offset."""
        jst = timezone(timedelta(hours=9))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=jst)
        result = normalize_to_utc(dt)

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

        assert original_dt.tzinfo is None
        assert result.tzinfo == UTC


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
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=est)
        result = to_iso_utc(dt)

        assert result == "2024-01-15T15:30:45+00:00"

    def test_to_iso_utc_with_positive_timezone(self) -> None:
        """Test to_iso_utc with positive timezone offset."""
        jst = timezone(timedelta(hours=9))
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=jst)
        result = to_iso_utc(dt)

        assert result == "2024-01-15T01:30:45+00:00"

    def test_to_iso_utc_with_microseconds(self) -> None:
        """Test to_iso_utc preserves microseconds."""
        dt = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        result = to_iso_utc(dt)

        assert result == "2024-01-15T10:30:45.123456+00:00"

    def test_to_iso_utc_edge_cases(self) -> None:
        """Test to_iso_utc with edge case datetimes."""
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)
        result = to_iso_utc(dt)
        assert result == "1970-01-01T00:00:00+00:00"

        dt = datetime(2100, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
        result = to_iso_utc(dt)
        assert result == "2100-12-31T23:59:59.999999+00:00"


class TestTimestampCoercionHelpers:
    def test_to_epoch_seconds_handles_various_types(self) -> None:
        """Test epoch coercion for numbers, datetimes, and ISO strings."""
        assert to_epoch_seconds(123.5) == 123.5
        assert to_epoch_seconds(42) == 42.0

        dt = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        assert to_epoch_seconds(dt) == dt.timestamp()

        iso = "2024-05-10T12:34:56Z"
        assert (
            to_epoch_seconds(iso) == datetime.fromisoformat("2024-05-10T12:34:56+00:00").timestamp()
        )

        assert to_epoch_seconds("invalid") is None
        assert to_epoch_seconds(None) is None
        assert to_epoch_seconds(False) is None

    def test_age_since_timestamp_seconds_returns_expected_age(self) -> None:
        """Test age calculations use provided `now_seconds` and missing values."""
        now = 1000.0
        timestamp_value = 900.0
        timestamp, age = age_since_timestamp_seconds(timestamp_value, now_seconds=now)
        assert timestamp == timestamp_value
        assert age == 100.0

        iso = "2024-01-01T00:00:00Z"
        timestamp_iso, age_iso = age_since_timestamp_seconds(
            iso,
            now_seconds=1700000000.0,
        )
        expected_iso = datetime.fromisoformat("2024-01-01T00:00:00+00:00").timestamp()
        assert timestamp_iso == expected_iso
        assert age_iso == 1700000000.0 - expected_iso

        missing_timestamp, missing_age = age_since_timestamp_seconds(
            None,
            now_seconds=now,
            missing_value=999.0,
        )
        assert missing_timestamp is None
        assert missing_age == 999.0
