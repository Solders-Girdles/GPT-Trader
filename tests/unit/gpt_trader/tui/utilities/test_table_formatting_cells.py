"""Tests for DataTable cell formatting utilities."""

from decimal import Decimal

from rich.text import Text

from gpt_trader.tui.utilities.table_formatting import create_numeric_cell, format_table_cell


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
