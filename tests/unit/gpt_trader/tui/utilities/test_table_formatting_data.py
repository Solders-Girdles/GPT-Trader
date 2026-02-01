"""Tests for DataTable sorting and timestamp utilities."""

from datetime import datetime, timezone
from decimal import Decimal

from gpt_trader.tui.utilities.table_formatting import (
    format_timestamp,
    get_age_seconds,
    get_sort_indicator,
    parse_timestamp_to_epoch,
    sort_table_data,
)


class TestSortTableData:
    """Tests for sort_table_data function."""

    def test_sort_by_string_column_ascending(self) -> None:
        """Sort by string column ascending."""
        data = [
            {"name": "Charlie", "age": 30},
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 35},
        ]
        result = sort_table_data(data, "name", ascending=True)
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"
        assert result[2]["name"] == "Charlie"

    def test_sort_by_string_column_descending(self) -> None:
        """Sort by string column descending."""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Charlie", "age": 30},
            {"name": "Bob", "age": 35},
        ]
        result = sort_table_data(data, "name", ascending=False)
        assert result[0]["name"] == "Charlie"
        assert result[1]["name"] == "Bob"
        assert result[2]["name"] == "Alice"

    def test_sort_by_numeric_column(self) -> None:
        """Sort by numeric column."""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Charlie", "age": 30},
            {"name": "Bob", "age": 35},
        ]
        result = sort_table_data(data, "age", ascending=True, numeric_columns={"age"})
        assert result[0]["age"] == 25
        assert result[1]["age"] == 30
        assert result[2]["age"] == 35

    def test_sort_empty_list(self) -> None:
        """Sort empty list returns empty list."""
        result = sort_table_data([], "name", ascending=True)
        assert result == []

    def test_sort_missing_column(self) -> None:
        """Sort by missing column doesn't crash."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = sort_table_data(data, "nonexistent", ascending=True)
        assert len(result) == 2

    def test_sort_decimal_values(self) -> None:
        """Sort Decimal values correctly."""
        data = [
            {"price": Decimal("100.50")},
            {"price": Decimal("50.25")},
            {"price": Decimal("200.75")},
        ]
        result = sort_table_data(data, "price", ascending=True, numeric_columns={"price"})
        assert result[0]["price"] == Decimal("50.25")
        assert result[2]["price"] == Decimal("200.75")


class TestGetSortIndicator:
    """Tests for get_sort_indicator function."""

    def test_no_sort(self) -> None:
        """No sort indicator when column not sorted."""
        result = get_sort_indicator("name", None, True)
        assert result == ""

    def test_ascending_indicator(self) -> None:
        """Ascending indicator for sorted column."""
        result = get_sort_indicator("name", "name", True)
        assert "▲" in result or "asc" in result.lower() or result != ""

    def test_descending_indicator(self) -> None:
        """Descending indicator for sorted column."""
        result = get_sort_indicator("name", "name", False)
        assert "▼" in result or "desc" in result.lower() or result != ""

    def test_different_column_no_indicator(self) -> None:
        """No indicator when different column is sorted."""
        result = get_sort_indicator("name", "age", True)
        assert result == ""


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
        expected = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc).timestamp()
        assert result is not None
        assert isinstance(result, float)
        assert result == expected

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
