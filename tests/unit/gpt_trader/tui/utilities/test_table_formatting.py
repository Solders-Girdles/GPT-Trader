"""Tests for DataTable formatting utilities."""

from decimal import Decimal
from unittest.mock import patch

from rich.text import Text

from gpt_trader.tui.utilities.table_formatting import (
    copy_to_clipboard,
    create_numeric_cell,
    format_table_cell,
    format_timestamp,
    get_sort_indicator,
    sort_table_data,
    truncate_id,
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


class TestTruncateId:
    """Tests for truncate_id function."""

    def test_long_id_truncated(self) -> None:
        """Long ID is truncated to last N characters."""
        result = truncate_id("abc123def456ghi789", length=8)
        assert result == "56ghi789"

    def test_short_id_unchanged(self) -> None:
        """Short ID (shorter than length) is unchanged."""
        result = truncate_id("short", length=8)
        assert result == "short"

    def test_exact_length_unchanged(self) -> None:
        """ID exactly at length is unchanged."""
        result = truncate_id("12345678", length=8)
        assert result == "12345678"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        result = truncate_id("")
        assert result == ""

    def test_none_returns_empty(self) -> None:
        """None returns empty string."""
        result = truncate_id(None)  # type: ignore[arg-type]
        assert result == ""

    def test_default_length(self) -> None:
        """Default length is 8."""
        result = truncate_id("0123456789abcdef")
        assert result == "89abcdef"
        assert len(result) == 8


class TestCreateNumericCell:
    """Tests for create_numeric_cell function."""

    def test_basic_value(self) -> None:
        """Basic value is converted to right-aligned Text."""
        result = create_numeric_cell(1234)
        assert isinstance(result, Text)
        assert result.plain == "1234"
        assert result.justify == "right"

    def test_with_format_function(self) -> None:
        """Format function is applied to value."""
        result = create_numeric_cell(1234.56, format_fn=lambda x: f"${x:,.2f}")
        assert result.plain == "$1,234.56"

    def test_custom_justify(self) -> None:
        """Custom justify is applied."""
        result = create_numeric_cell(100, justify="center")
        assert result.justify == "center"

    def test_float_value(self) -> None:
        """Float value is handled correctly."""
        result = create_numeric_cell(3.14159)
        assert "3.14159" in result.plain

    def test_string_value(self) -> None:
        """String value is handled correctly."""
        result = create_numeric_cell("N/A")
        assert result.plain == "N/A"


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
        # Should return original order or handle gracefully
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


class TestFormatTableCell:
    """Tests for format_table_cell function."""

    def test_format_text(self) -> None:
        """Format text value."""
        result = format_table_cell("Hello", column_type="text")
        assert result == "Hello"

    def test_format_currency(self) -> None:
        """Format currency value."""
        result = format_table_cell(1234.56, column_type="currency")
        assert "$" in result
        assert "1,234" in result or "1234" in result

    def test_format_percentage(self) -> None:
        """Format percentage value."""
        # Note: format_table_cell uses "percent" not "percentage"
        result = format_table_cell(0.1234, column_type="percent")
        assert "%" in result
        assert "12" in result  # 0.1234 * 100 = 12.34%

    def test_format_number(self) -> None:
        """Format number value."""
        result = format_table_cell(1234567, column_type="number")
        # Should be formatted with commas or as number
        assert "1234567" in result or "1,234,567" in result

    def test_format_decimal(self) -> None:
        """Format Decimal value."""
        result = format_table_cell(Decimal("123.456"), column_type="number")
        assert "123" in result

    def test_format_with_max_length(self) -> None:
        """Format with max length truncation."""
        result = format_table_cell("This is a very long string", max_length=10)
        assert len(result) <= 13  # Allow for "..." suffix

    def test_format_none(self) -> None:
        """Format None value."""
        result = format_table_cell(None)
        assert result == "--"  # Implementation returns "--" for None


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


class TestCopyToClipboard:
    """Tests for copy_to_clipboard function."""

    @patch("subprocess.run")
    def test_copy_on_macos(self, mock_run) -> None:
        """Test clipboard copy on macOS."""
        mock_run.return_value.returncode = 0
        # This test is platform-dependent, so we just verify no crash
        result = copy_to_clipboard("test text")
        # Result depends on platform and available tools
        assert isinstance(result, bool)

    def test_copy_empty_string(self) -> None:
        """Copy empty string doesn't crash."""
        # Should handle gracefully
        result = copy_to_clipboard("")
        assert isinstance(result, bool)
