"""Tests for StrategyDetailScreen backtest display and status badges."""

from gpt_trader.tui.screens.strategy_detail_screen import StrategyDetailScreen
from gpt_trader.tui.types import StrategyPerformance


class TestBuildBacktestDisplay:
    """Tests for _build_backtest_display helper."""

    def test_build_backtest_display_placeholder(self):
        """None performance returns placeholder values and note."""
        screen = StrategyDetailScreen()

        values, note = screen._build_backtest_display(None)

        assert values == {
            "win_rate": "--",
            "trades": "--",
            "profit_factor": "--",
            "drawdown": "--",
        }
        assert note == "No backtest data available"

    def test_build_backtest_display_values(self):
        """Performance with trades returns formatted values."""
        screen = StrategyDetailScreen()
        performance = StrategyPerformance(
            win_rate=0.652,
            profit_factor=1.85,
            max_drawdown_pct=-3.2,
            total_trades=23,
        )

        values, note = screen._build_backtest_display(performance)

        assert values["win_rate"] == "65.2%"
        assert values["trades"] == "23"
        assert values["profit_factor"] == "1.85"
        assert values["drawdown"] == "-3.2%"
        assert note == ""


class TestFormatDelta:
    """Tests for _format_delta helper."""

    def test_positive_delta_shows_green_with_plus(self):
        """Positive delta shows green color with plus sign."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=65.5, previous=60.0, suffix="%")
        assert "+5.5%" in result
        assert "green" in result

    def test_negative_delta_shows_red(self):
        """Negative delta shows red color."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=55.0, previous=60.0, suffix="%")
        assert "-5.0%" in result
        assert "red" in result

    def test_no_previous_returns_empty(self):
        """No previous value returns empty string."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=65.5, previous=None)
        assert result == ""

    def test_tiny_delta_ignored(self):
        """Deltas smaller than 0.01 are ignored."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=65.005, previous=65.0)
        assert result == ""

    def test_precision_parameter(self):
        """Precision parameter controls decimal places."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=1.85, previous=1.50, precision=2)
        assert "+0.35" in result

    def test_suffix_parameter(self):
        """Suffix parameter appends to formatted value."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=10.0, previous=5.0, suffix="%")
        assert "+5.0%" in result


class TestEntryExitBadgeDetailScreen:
    """Tests for _get_entry_exit_badge method on StrategyDetailScreen."""

    def test_buy_action_returns_entry_badge(self):
        """BUY action returns ENTRY badge."""
        screen = StrategyDetailScreen()
        badge = screen._get_entry_exit_badge("BUY")
        assert "ENTRY" in badge
        assert "cyan" in badge

    def test_close_action_returns_exit_badge(self):
        """CLOSE action returns EXIT badge."""
        screen = StrategyDetailScreen()
        badge = screen._get_entry_exit_badge("CLOSE")
        assert "EXIT" in badge
        assert "magenta" in badge

    def test_hold_action_returns_empty(self):
        """HOLD action returns empty string."""
        screen = StrategyDetailScreen()
        badge = screen._get_entry_exit_badge("HOLD")
        assert badge == ""
