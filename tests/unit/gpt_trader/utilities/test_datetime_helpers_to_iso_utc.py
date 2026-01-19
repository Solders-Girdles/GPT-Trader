"""Tests for datetime helper utilities (ISO conversion)."""

from datetime import UTC, datetime, timedelta, timezone

from gpt_trader.utilities.datetime_helpers import to_iso_utc


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
