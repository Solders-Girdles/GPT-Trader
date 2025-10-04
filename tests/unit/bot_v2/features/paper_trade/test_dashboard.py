"""
Comprehensive tests for paper trading dashboard.

Tests cover:
- Dashboard initialization
- Screen clearing and formatting
- Metrics calculation
- Display sections (header, portfolio, positions, performance, trades)
- Continuous display
- HTML report generation
- Edge cases and error handling
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.paper_trade.dashboard import PaperTradingDashboard


# ============================================================================
# Helper Functions and Fixtures
# ============================================================================


def create_mock_engine(
    initial_capital=100000,
    cash=100000,
    bot_id="test_bot",
    positions=None,
    trades=None,
):
    """Create mock execution engine for testing."""
    engine = Mock()
    engine.initial_capital = initial_capital
    engine.cash = cash
    engine.bot_id = bot_id
    engine.positions = positions or {}
    engine.trades = trades or []

    # Mock calculate_equity
    def calc_equity():
        positions_value = sum(
            pos.quantity
            * (
                pos.current_price
                if hasattr(pos, "current_price") and pos.current_price > 0
                else pos.entry_price
            )
            for pos in engine.positions.values()
        )
        return engine.cash + positions_value

    engine.calculate_equity = Mock(side_effect=calc_equity)

    return engine


def create_mock_position(symbol, quantity, entry_price, current_price=None):
    """Create mock position."""
    pos = Mock()
    pos.symbol = symbol
    pos.quantity = quantity
    pos.entry_price = entry_price
    pos.current_price = current_price if current_price is not None else entry_price
    return pos


def create_mock_trade(symbol, side, quantity, price, timestamp=None, pnl=None):
    """Create mock trade."""
    trade = Mock()
    trade.symbol = symbol
    trade.side = side
    trade.quantity = quantity
    trade.price = price
    trade.timestamp = timestamp or datetime.now()
    trade.pnl = pnl
    return trade


# ============================================================================
# Test: Dashboard Initialization
# ============================================================================


class TestDashboardInitialization:
    """Test dashboard initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.engine == engine
        assert dashboard.refresh_interval == 5
        assert dashboard.initial_equity == 100000
        assert isinstance(dashboard.start_time, datetime)

    def test_initialization_custom_refresh_interval(self):
        """Test initialization with custom refresh interval."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine, refresh_interval=10)

        assert dashboard.refresh_interval == 10

    def test_initialization_captures_initial_equity(self):
        """Test that initialization captures initial equity."""
        engine = create_mock_engine(initial_capital=50000)
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.initial_equity == 50000


# ============================================================================
# Test: Formatting Functions
# ============================================================================


class TestFormattingFunctions:
    """Test formatting utility functions."""

    def test_format_currency_positive(self):
        """Test currency formatting with positive values."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_currency(1234.56) == "$1,234.56"
        assert dashboard.format_currency(1000000) == "$1,000,000.00"

    def test_format_currency_negative(self):
        """Test currency formatting with negative values."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_currency(-1234.56) == "$-1,234.56"

    def test_format_currency_zero(self):
        """Test currency formatting with zero."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_currency(0) == "$0.00"

    def test_format_pct_positive(self):
        """Test percentage formatting with positive values."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_pct(5.67) == "+5.67%"
        assert dashboard.format_pct(0.01) == "+0.01%"

    def test_format_pct_negative(self):
        """Test percentage formatting with negative values."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_pct(-5.67) == "-5.67%"

    def test_format_pct_zero(self):
        """Test percentage formatting with zero."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_pct(0) == "0.00%"


# ============================================================================
# Test: Screen Clearing
# ============================================================================


class TestScreenClearing:
    """Test screen clearing functionality."""

    @patch("os.system")
    @patch("os.name", "posix")
    def test_clear_screen_posix(self, mock_system):
        """Test screen clearing on POSIX systems."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        dashboard.clear_screen()

        mock_system.assert_called_once_with("clear")

    @patch("os.system")
    @patch("os.name", "nt")
    def test_clear_screen_windows(self, mock_system):
        """Test screen clearing on Windows."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        dashboard.clear_screen()

        mock_system.assert_called_once_with("cls")


# ============================================================================
# Test: Metrics Calculation
# ============================================================================


class TestMetricsCalculation:
    """Test metrics calculation."""

    def test_calculate_metrics_initial_state(self):
        """Test metrics calculation at initial state."""
        engine = create_mock_engine(initial_capital=100000, cash=100000)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["equity"] == 100000
        assert metrics["cash"] == 100000
        assert metrics["positions_value"] == 0
        assert metrics["returns_pct"] == 0.0
        assert metrics["total_trades"] == 0
        assert metrics["positions_count"] == 0

    def test_calculate_metrics_with_profit(self):
        """Test metrics calculation with profit."""
        engine = create_mock_engine(initial_capital=100000, cash=110000)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["equity"] == 110000
        assert metrics["returns_pct"] == 10.0

    def test_calculate_metrics_with_loss(self):
        """Test metrics calculation with loss."""
        engine = create_mock_engine(initial_capital=100000, cash=90000)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["equity"] == 90000
        assert metrics["returns_pct"] == -10.0

    def test_calculate_metrics_with_positions(self):
        """Test metrics calculation with open positions."""
        positions = {
            "AAPL": create_mock_position("AAPL", 100, 150.0, 155.0),
            "MSFT": create_mock_position("MSFT", 50, 300.0, 310.0),
        }
        engine = create_mock_engine(initial_capital=100000, cash=50000, positions=positions)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        # Positions value = (100 * 155) + (50 * 310) = 15500 + 15500 = 31000
        assert metrics["positions_value"] == 31000
        assert metrics["equity"] == 81000  # 50000 cash + 31000 positions
        assert metrics["positions_count"] == 2

    def test_calculate_metrics_exposure(self):
        """Test exposure calculation."""
        positions = {"AAPL": create_mock_position("AAPL", 100, 100.0, 100.0)}
        engine = create_mock_engine(initial_capital=100000, cash=90000, positions=positions)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        # Exposure = positions_value / equity * 100 = 10000 / 100000 * 100 = 10%
        assert metrics["exposure_pct"] == pytest.approx(10.0, rel=1e-6)

    def test_calculate_metrics_win_rate(self):
        """Test win rate calculation."""
        trades = [
            create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None),
            create_mock_trade("AAPL", "sell", 100, 155.0, pnl=500.0),  # Win
            create_mock_trade("MSFT", "buy", 50, 300.0, pnl=None),
            create_mock_trade("MSFT", "sell", 50, 295.0, pnl=-250.0),  # Loss
        ]
        engine = create_mock_engine(trades=trades)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 50.0
        assert metrics["total_trades"] == 4

    def test_calculate_metrics_no_trades(self):
        """Test metrics with no trades."""
        engine = create_mock_engine(trades=[])
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["win_rate"] == 0.0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0

    def test_calculate_metrics_all_winning_trades(self):
        """Test metrics with all winning trades."""
        trades = [
            create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None),
            create_mock_trade("AAPL", "sell", 100, 155.0, pnl=500.0),
            create_mock_trade("MSFT", "buy", 50, 300.0, pnl=None),
            create_mock_trade("MSFT", "sell", 50, 310.0, pnl=500.0),
        ]
        engine = create_mock_engine(trades=trades)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["win_rate"] == 100.0


# ============================================================================
# Test: Display Functions
# ============================================================================


class TestDisplayFunctions:
    """Test individual display functions."""

    @patch("builtins.print")
    def test_print_header(self, mock_print):
        """Test header printing."""
        engine = create_mock_engine(bot_id="test_bot_123")
        dashboard = PaperTradingDashboard(engine)

        dashboard.print_header()

        # Should print multiple lines
        assert mock_print.call_count >= 5

    @patch("builtins.print")
    def test_print_portfolio_summary(self, mock_print):
        """Test portfolio summary printing."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        metrics = {
            "equity": 105000,
            "cash": 50000,
            "positions_value": 55000,
            "returns_pct": 5.0,
            "drawdown_pct": -2.0,
            "exposure_pct": 52.38,
        }

        dashboard.print_portfolio_summary(metrics)

        # Should print multiple lines
        assert mock_print.call_count >= 4

    @patch("builtins.print")
    def test_print_positions_empty(self, mock_print):
        """Test printing positions when none exist."""
        engine = create_mock_engine(positions={})
        dashboard = PaperTradingDashboard(engine)

        dashboard.print_positions()

        # Should indicate no positions
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("No open positions" in call for call in print_calls)

    @patch("builtins.print")
    def test_print_positions_with_positions(self, mock_print):
        """Test printing positions with open positions."""
        positions = {
            "AAPL": create_mock_position("AAPL", 100, 150.0, 155.0),
            "MSFT": create_mock_position("MSFT", 50, 300.0, 295.0),
        }
        engine = create_mock_engine(positions=positions)
        dashboard = PaperTradingDashboard(engine)

        dashboard.print_positions()

        # Should print header and position rows
        assert mock_print.call_count >= 4

    @patch("builtins.print")
    def test_print_performance(self, mock_print):
        """Test performance metrics printing."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        metrics = {
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "win_rate": 60.0,
        }

        dashboard.print_performance(metrics)

        # Should print multiple lines
        assert mock_print.call_count >= 5

    @patch("builtins.print")
    def test_print_recent_trades_empty(self, mock_print):
        """Test printing recent trades when none exist."""
        engine = create_mock_engine(trades=[])
        dashboard = PaperTradingDashboard(engine)

        dashboard.print_recent_trades()

        # Should indicate no trades
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("No trades" in call for call in print_calls)

    @patch("builtins.print")
    def test_print_recent_trades_with_trades(self, mock_print):
        """Test printing recent trades."""
        trades = [
            create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None),
            create_mock_trade("AAPL", "sell", 100, 155.0, pnl=500.0),
            create_mock_trade("MSFT", "buy", 50, 300.0, pnl=None),
        ]
        engine = create_mock_engine(trades=trades)
        dashboard = PaperTradingDashboard(engine)

        dashboard.print_recent_trades(limit=5)

        # Should print header and trade rows
        assert mock_print.call_count >= 5

    @patch("builtins.print")
    def test_print_recent_trades_limit(self, mock_print):
        """Test recent trades respects limit."""
        trades = [create_mock_trade("AAPL", "buy", 100, 150.0 + i) for i in range(10)]
        engine = create_mock_engine(trades=trades)
        dashboard = PaperTradingDashboard(engine)

        dashboard.print_recent_trades(limit=3)

        # Should only show 3 trades
        # Count of lines should be limited
        assert mock_print.call_count <= 10  # Header + 3 trades + separators


# ============================================================================
# Test: Display Once
# ============================================================================


class TestDisplayOnce:
    """Test single display functionality."""

    @patch("builtins.print")
    def test_display_once_calls_all_sections(self, mock_print):
        """Test that display_once calls all display sections."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        dashboard.display_once()

        # Should print many lines (header, portfolio, positions, performance, trades)
        assert mock_print.call_count >= 10

    @patch("builtins.print")
    def test_display_once_with_data(self, mock_print):
        """Test display_once with actual data."""
        positions = {"AAPL": create_mock_position("AAPL", 100, 150.0, 155.0)}
        trades = [create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None)]
        engine = create_mock_engine(positions=positions, trades=trades)
        dashboard = PaperTradingDashboard(engine)

        dashboard.display_once()

        # Should complete without error
        assert mock_print.call_count >= 10


# ============================================================================
# Test: Continuous Display
# ============================================================================


class TestContinuousDisplay:
    """Test continuous display functionality."""

    @patch("bot_v2.features.paper_trade.dashboard.display_controller.time.sleep")
    @patch("builtins.print")
    @patch("bot_v2.features.paper_trade.dashboard.display_controller.os.system")
    def test_display_continuous_with_duration(self, mock_clear, mock_print, mock_sleep):
        """Test continuous display with duration limit."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine, refresh_interval=1)

        # Mock time to simulate duration passing
        with patch(
            "bot_v2.features.paper_trade.dashboard.display_controller.time.time"
        ) as mock_time:
            mock_time.side_effect = [0, 0, 1.5]  # Start, check, exit

            dashboard.display_continuous(duration=1)

        # Should clear screen and display
        assert mock_clear.call_count >= 1

    @patch(
        "bot_v2.features.paper_trade.dashboard.display_controller.time.sleep",
        side_effect=KeyboardInterrupt,
    )
    @patch("builtins.print")
    @patch("bot_v2.features.paper_trade.dashboard.display_controller.os.system")
    def test_display_continuous_keyboard_interrupt(self, mock_clear, mock_print, mock_sleep):
        """Test continuous display handles keyboard interrupt."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        # Should handle KeyboardInterrupt gracefully
        dashboard.display_continuous()

        # Should have attempted to display
        assert mock_clear.call_count >= 1


# ============================================================================
# Test: HTML Report Generation
# ============================================================================


class TestHTMLReportGeneration:
    """Test HTML report generation."""

    def test_generate_html_summary_creates_file(self):
        """Test that HTML summary creates a file."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_report.html")
            result_path = dashboard.generate_html_summary(output_path)

            assert result_path == output_path
            assert Path(output_path).exists()

    def test_generate_html_summary_content(self):
        """Test HTML summary content."""
        positions = {"AAPL": create_mock_position("AAPL", 100, 150.0, 155.0)}
        trades = [
            create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None),
            create_mock_trade("AAPL", "sell", 100, 155.0, pnl=500.0),
        ]
        engine = create_mock_engine(positions=positions, trades=trades)
        dashboard = PaperTradingDashboard(engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_report.html")
            dashboard.generate_html_summary(output_path)

            with open(output_path) as f:
                content = f.read()

            # Should contain key elements
            assert "<!DOCTYPE html>" in content
            assert "Paper Trading Summary Report" in content
            assert "AAPL" in content
            assert "Portfolio Metrics" in content

    def test_generate_html_summary_default_path(self):
        """Test HTML summary with default path."""
        engine = create_mock_engine(bot_id="test_bot")
        dashboard = PaperTradingDashboard(engine)

        result_path = dashboard.generate_html_summary()

        # Should create file with default naming
        assert "paper_trading_summary_test_bot" in result_path
        assert result_path.endswith(".html")

        # Cleanup
        if Path(result_path).exists():
            Path(result_path).unlink()

    def test_generate_html_summary_empty_positions(self):
        """Test HTML generation with no positions."""
        engine = create_mock_engine(positions={}, trades=[])
        dashboard = PaperTradingDashboard(engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_report.html")
            dashboard.generate_html_summary(output_path)

            with open(output_path) as f:
                content = f.read()

            # Should handle empty state
            assert "No open positions" in content
            assert "No trades executed" in content

    def test_generate_html_summary_formatting(self):
        """Test HTML formatting of values."""
        engine = create_mock_engine(initial_capital=100000, cash=105000)
        dashboard = PaperTradingDashboard(engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_report.html")
            dashboard.generate_html_summary(output_path)

            with open(output_path) as f:
                content = f.read()

            # Should contain formatted currency
            assert "$105,000.00" in content or "$105000.00" in content

    def test_generate_html_creates_directories(self):
        """Test that HTML generation creates parent directories."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "nested" / "dir" / "test_report.html")
            result_path = dashboard.generate_html_summary(output_path)

            assert Path(output_path).exists()
            assert Path(output_path).parent.exists()


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_metrics_zero_equity(self):
        """Test metrics calculation with zero equity."""
        engine = create_mock_engine(initial_capital=0, cash=0)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        # Should handle zero equity gracefully
        assert metrics["equity"] == 0
        assert metrics["exposure_pct"] == 0.0

    def test_metrics_negative_equity(self):
        """Test metrics calculation with negative equity."""
        engine = create_mock_engine(initial_capital=100000, cash=-10000)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["equity"] == -10000
        assert metrics["returns_pct"] == -110.0

    def test_position_with_zero_entry_price(self):
        """Test position handling with zero entry price."""
        positions = {"AAPL": create_mock_position("AAPL", 100, 0.0, 155.0)}
        engine = create_mock_engine(positions=positions)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        # Should handle gracefully
        assert "positions_value" in metrics

    def test_position_without_current_price(self):
        """Test position handling when current_price is missing."""
        pos = Mock()
        pos.symbol = "AAPL"
        pos.quantity = 100
        pos.entry_price = 150.0
        pos.current_price = 0  # Default/invalid

        positions = {"AAPL": pos}
        engine = create_mock_engine(positions=positions)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        # Should use entry_price as fallback
        assert metrics["positions_value"] == 15000

    def test_trade_without_pnl(self):
        """Test trade handling when pnl is None."""
        trades = [create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None)]
        engine = create_mock_engine(trades=trades)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        # Should handle None pnl
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        engine = create_mock_engine(initial_capital=1e10, cash=1.5e10)
        dashboard = PaperTradingDashboard(engine)

        metrics = dashboard.calculate_metrics()

        assert metrics["equity"] == 1.5e10
        # Should format correctly
        formatted = dashboard.format_currency(metrics["equity"])
        assert "$" in formatted

    def test_very_small_percentages(self):
        """Test formatting of very small percentages."""
        engine = create_mock_engine()
        dashboard = PaperTradingDashboard(engine)

        assert dashboard.format_pct(0.0001) == "+0.00%"
        assert dashboard.format_pct(-0.0001) == "-0.00%"

    @patch("builtins.print")
    def test_display_with_none_values(self, mock_print):
        """Test display handles None values gracefully."""
        trade = Mock()
        trade.symbol = "AAPL"
        trade.side = "buy"
        trade.quantity = 100
        trade.price = 150.0
        trade.timestamp = datetime.now()
        trade.pnl = None

        engine = create_mock_engine(trades=[trade])
        dashboard = PaperTradingDashboard(engine)

        # Should not raise error
        dashboard.print_recent_trades()


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch("builtins.print")
    def test_complete_dashboard_flow(self, mock_print):
        """Test complete dashboard display flow."""
        # Set up realistic state
        positions = {
            "AAPL": create_mock_position("AAPL", 100, 150.0, 155.0),
            "MSFT": create_mock_position("MSFT", 50, 300.0, 310.0),
        }
        trades = [
            create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None),
            create_mock_trade("AAPL", "sell", 50, 155.0, pnl=250.0),
            create_mock_trade("MSFT", "buy", 50, 300.0, pnl=None),
        ]
        engine = create_mock_engine(
            initial_capital=100000, cash=50000, positions=positions, trades=trades
        )
        dashboard = PaperTradingDashboard(engine)

        # Display complete dashboard
        dashboard.display_once()

        # Should print all sections
        assert mock_print.call_count >= 20

    def test_html_generation_comprehensive(self):
        """Test comprehensive HTML generation."""
        # Set up complete state
        positions = {
            "AAPL": create_mock_position("AAPL", 100, 150.0, 155.0),
            "MSFT": create_mock_position("MSFT", 50, 300.0, 295.0),
        }
        trades = [
            create_mock_trade("AAPL", "buy", 100, 150.0, pnl=None),
            create_mock_trade("AAPL", "sell", 100, 155.0, pnl=500.0),
            create_mock_trade("MSFT", "buy", 50, 300.0, pnl=None),
            create_mock_trade("MSFT", "sell", 50, 295.0, pnl=-250.0),
        ]
        engine = create_mock_engine(
            initial_capital=100000,
            cash=50000,
            bot_id="test_bot",
            positions=positions,
            trades=trades,
        )
        dashboard = PaperTradingDashboard(engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "comprehensive_report.html")
            dashboard.generate_html_summary(output_path)

            with open(output_path) as f:
                content = f.read()

            # Verify comprehensive content
            assert "AAPL" in content
            assert "MSFT" in content
            assert "buy" in content.lower()
            assert "sell" in content.lower()
            assert "$" in content
