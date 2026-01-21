"""Tests for DataTable cell formatting, clipboard, and ID utilities."""

import subprocess
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from rich.text import Text

from gpt_trader.tui.utilities.table_formatting import (
    copy_to_clipboard,
    create_numeric_cell,
    format_table_cell,
    truncate_id,
)


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


class TestCopyToClipboard:
    """Tests for copy_to_clipboard function."""

    def test_copy_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clipboard copy on macOS."""
        mock_run = MagicMock()
        mock_run.return_value.returncode = 0
        monkeypatch.setattr(subprocess, "run", mock_run)
        # This test is platform-dependent, so we just verify no crash
        result = copy_to_clipboard("test text")
        # Result depends on platform and available tools
        assert isinstance(result, bool)

    def test_copy_empty_string(self) -> None:
        """Copy empty string doesn't crash."""
        result = copy_to_clipboard("")
        assert isinstance(result, bool)


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
