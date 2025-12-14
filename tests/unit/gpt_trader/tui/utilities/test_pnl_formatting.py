"""Tests for P&L and directional color formatting utilities."""

from decimal import Decimal

from rich.text import Text

from gpt_trader.tui.utilities.pnl_formatting import (
    format_direction_colored,
    format_leverage_colored,
    format_pnl_colored,
    format_side_colored,
    get_sparkline_color,
)


class TestFormatPnlColored:
    """Tests for format_pnl_colored function."""

    def test_positive_value_uses_success_color(self) -> None:
        """Positive P&L should use green/success color."""
        result = format_pnl_colored(100.50, "$100.50")
        assert isinstance(result, Text)
        # Check that markup contains success color
        markup = result.markup
        assert "100.50" in markup

    def test_negative_value_uses_error_color(self) -> None:
        """Negative P&L should use red/error color."""
        result = format_pnl_colored(-50.25, "-$50.25")
        assert isinstance(result, Text)
        markup = result.markup
        assert "50.25" in markup

    def test_none_value_uses_dim_style(self) -> None:
        """None P&L should use dim styling."""
        result = format_pnl_colored(None, "N/A")
        assert isinstance(result, Text)
        assert result.plain == "N/A"

    def test_zero_value_is_neutral(self) -> None:
        """Zero P&L should have no color styling."""
        result = format_pnl_colored(0, "$0.00")
        assert isinstance(result, Text)
        assert result.plain == "$0.00"

    def test_default_display_string(self) -> None:
        """When no display_str provided, uses str(value)."""
        result = format_pnl_colored(100.0)
        assert "100.0" in result.plain

    def test_none_value_default_display(self) -> None:
        """None value with no display_str shows N/A."""
        result = format_pnl_colored(None)
        assert result.plain == "N/A"

    def test_decimal_value(self) -> None:
        """Works with Decimal values."""
        result = format_pnl_colored(Decimal("123.45"), "$123.45")
        assert isinstance(result, Text)

    def test_justify_parameter(self) -> None:
        """Justify parameter is applied to Text."""
        result = format_pnl_colored(100.0, "100", justify="left")
        assert result.justify == "left"


class TestFormatSideColored:
    """Tests for format_side_colored function."""

    def test_buy_uses_success_color(self) -> None:
        """BUY should use green/success color."""
        result = format_side_colored("BUY")
        assert "BUY" in result
        # Should contain color markup
        assert "[" in result and "]" in result

    def test_sell_uses_error_color(self) -> None:
        """SELL should use red/error color."""
        result = format_side_colored("SELL")
        assert "SELL" in result
        assert "[" in result and "]" in result

    def test_other_value_uses_error_color(self) -> None:
        """Non-BUY values default to error color."""
        result = format_side_colored("UNKNOWN")
        assert "UNKNOWN" in result


class TestFormatLeverageColored:
    """Tests for format_leverage_colored function."""

    def test_low_leverage_green(self) -> None:
        """Leverage below 2x should be green."""
        result = format_leverage_colored(1.5)
        assert isinstance(result, Text)
        assert "1.5x" in result.plain
        assert "green" in result.markup

    def test_medium_leverage_yellow(self) -> None:
        """Leverage between 2x and 5x should be yellow."""
        result = format_leverage_colored(3.0)
        assert "3.0x" in result.plain
        assert "yellow" in result.markup

    def test_high_leverage_red(self) -> None:
        """Leverage above 5x should be red."""
        result = format_leverage_colored(10.0)
        assert "10.0x" in result.plain
        assert "red" in result.markup

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        # With low_threshold=1.0, 1.5 should be yellow
        result = format_leverage_colored(1.5, low_threshold=1.0, medium_threshold=3.0)
        assert "yellow" in result.markup

    def test_justify_parameter(self) -> None:
        """Justify parameter is applied."""
        result = format_leverage_colored(2.0, justify="center")
        assert result.justify == "center"


class TestFormatDirectionColored:
    """Tests for format_direction_colored function."""

    def test_positive_direction_green(self) -> None:
        """Positive values should be green."""
        result = format_direction_colored(5.2, "+5.2%")
        assert isinstance(result, Text)
        assert "5.2" in result.plain

    def test_negative_direction_red(self) -> None:
        """Negative values should be red."""
        result = format_direction_colored(-3.1, "-3.1%")
        assert "3.1" in result.plain

    def test_zero_is_neutral(self) -> None:
        """Zero should have no color."""
        result = format_direction_colored(0, "0.0%")
        assert result.plain == "0.0%"

    def test_decimal_value(self) -> None:
        """Works with Decimal values."""
        result = format_direction_colored(Decimal("2.5"), "+2.5%")
        assert isinstance(result, Text)


class TestGetSparklineColor:
    """Tests for get_sparkline_color function."""

    def test_uptrend_returns_success(self) -> None:
        """Uptrend (last >= first) returns success color."""
        color = get_sparkline_color([1, 2, 3, 4, 5])
        # Should be success color
        assert color is not None

    def test_downtrend_returns_error(self) -> None:
        """Downtrend (last < first) returns error color."""
        color = get_sparkline_color([5, 4, 3, 2, 1])
        # Should be error color
        assert color is not None

    def test_flat_returns_success(self) -> None:
        """Flat (last == first) returns success color."""
        color = get_sparkline_color([3, 3, 3])
        assert color is not None

    def test_empty_list_returns_success(self) -> None:
        """Empty list returns success as default."""
        color = get_sparkline_color([])
        assert color is not None

    def test_single_value_returns_success(self) -> None:
        """Single value returns success as default."""
        color = get_sparkline_color([5])
        assert color is not None
