"""Tests for datetime helper utilities (current time helpers)."""

from datetime import UTC, datetime

from gpt_trader.utilities.datetime_helpers import utc_now, utc_now_iso, utc_timestamp


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
