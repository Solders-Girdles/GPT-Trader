"""Tests for DataTable formatting utilities."""

from rich.text import Text

from gpt_trader.tui.utilities.table_formatting import (
    create_numeric_cell,
    format_timestamp,
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
