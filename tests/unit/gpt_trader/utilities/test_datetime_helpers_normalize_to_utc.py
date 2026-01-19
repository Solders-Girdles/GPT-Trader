"""Tests for datetime helper utilities (timezone normalization)."""

from datetime import UTC, datetime, timedelta, timezone

from gpt_trader.utilities.datetime_helpers import normalize_to_utc


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
