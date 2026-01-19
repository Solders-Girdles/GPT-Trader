"""Tests for DataTable timestamp formatting utilities."""

from gpt_trader.tui.utilities.table_formatting import (
    format_timestamp,
    get_age_seconds,
    parse_timestamp_to_epoch,
)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_iso_with_milliseconds_and_z(self) -> None:
        """Parse full ISO 8601 format with Z suffix."""
        result = format_timestamp("2024-01-15T10:30:45.123Z")
        assert result == "10:30:45"

    def test_iso_with_milliseconds(self) -> None:
        """Parse ISO format with milliseconds."""
        result = format_timestamp("2024-01-15T10:30:45.123")
        assert result == "10:30:45"

    def test_iso_without_milliseconds(self) -> None:
        """Parse ISO format without milliseconds."""
        result = format_timestamp("2024-01-15T10:30:45")
        assert result == "10:30:45"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        result = format_timestamp("")
        assert result == ""

    def test_no_t_separator(self) -> None:
        """String without T returns original if parsing fails."""
        result = format_timestamp("not-a-timestamp")
        assert result == "not-a-timestamp"

    def test_already_formatted_time(self) -> None:
        """Already formatted time returns as-is if no T."""
        result = format_timestamp("10:30:45")
        assert result == "10:30:45"


class TestParseTimestampToEpoch:
    """Tests for parse_timestamp_to_epoch function."""

    def test_float_passthrough(self) -> None:
        """Float epoch timestamp passes through unchanged."""
        result = parse_timestamp_to_epoch(1705312245.0)
        assert result == 1705312245.0

    def test_int_passthrough(self) -> None:
        """Int epoch timestamp converts to float."""
        result = parse_timestamp_to_epoch(1705312245)
        assert result == 1705312245.0

    def test_iso_with_z_suffix(self) -> None:
        """Parse ISO timestamp with Z suffix."""
        result = parse_timestamp_to_epoch("2024-01-15T10:30:45Z")
        assert result is not None
        assert isinstance(result, float)
        # Just verify it's a reasonable timestamp (after 2024)
        assert result > 1704067200  # 2024-01-01 00:00:00 UTC

    def test_iso_with_microseconds_and_z(self) -> None:
        """Parse ISO timestamp with microseconds and Z suffix."""
        result = parse_timestamp_to_epoch("2024-01-15T10:30:45.123456Z")
        assert result is not None
        assert isinstance(result, float)

    def test_iso_without_z(self) -> None:
        """Parse ISO timestamp without Z suffix."""
        result = parse_timestamp_to_epoch("2024-01-15T10:30:45")
        assert result is not None
        assert isinstance(result, float)

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        result = parse_timestamp_to_epoch(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        result = parse_timestamp_to_epoch("")
        assert result is None

    def test_invalid_string_returns_none(self) -> None:
        """Invalid string returns None."""
        result = parse_timestamp_to_epoch("not-a-timestamp")
        assert result is None


class TestGetAgeSeconds:
    """Tests for get_age_seconds function."""

    def test_recent_timestamp(self) -> None:
        """Recent timestamp returns small positive age."""
        import time

        recent = time.time() - 30  # 30 seconds ago
        result = get_age_seconds(recent)
        assert result is not None
        # Allow some tolerance for test execution time
        assert 28 <= result <= 35

    def test_old_timestamp(self) -> None:
        """Old timestamp returns correct age."""
        import time

        old = time.time() - 3600  # 1 hour ago
        result = get_age_seconds(old)
        assert result is not None
        # Allow some tolerance
        assert 3598 <= result <= 3605

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        result = get_age_seconds(None)
        assert result is None

    def test_zero_returns_none(self) -> None:
        """Zero timestamp returns None (invalid)."""
        result = get_age_seconds(0)
        assert result is None

    def test_negative_returns_none(self) -> None:
        """Negative timestamp returns None (invalid)."""
        result = get_age_seconds(-100)
        assert result is None

    def test_future_timestamp_returns_zero(self) -> None:
        """Future timestamp returns 0 (not negative)."""
        import time

        future = time.time() + 100  # 100 seconds in the future
        result = get_age_seconds(future)
        assert result == 0
