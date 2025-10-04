"""
Tests for console renderer.

Tests cover:
- Header rendering
- Portfolio summary rendering
- Positions rendering
- Performance metrics rendering
- Recent trades rendering
- Edge cases
"""

from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.paper_trade.dashboard.console_renderer import ConsoleRenderer
from bot_v2.features.paper_trade.dashboard.formatters import DashboardFormatter


# ============================================================================
# Test: Header Rendering
# ============================================================================


class TestConsoleRendererHeader:
    """Test header rendering."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_header(self, mock_stdout):
        """Test that header is rendered correctly."""
        formatter = DashboardFormatter()
        start_time = datetime.now() - timedelta(hours=2, minutes=30)
        renderer = ConsoleRenderer(formatter, start_time)

        renderer.render_header(bot_id="test-bot-123")

        output = mock_stdout.getvalue()
        assert "PAPER TRADING DASHBOARD" in output
        assert "Bot ID: test-bot-123" in output
        assert "Runtime:" in output
        assert "Updated:" in output
        assert "=" * 80 in output
        assert "-" * 80 in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_header_with_special_characters(self, mock_stdout):
        """Test header with special characters in bot_id."""
        formatter = DashboardFormatter()
        start_time = datetime.now()
        renderer = ConsoleRenderer(formatter, start_time)

        renderer.render_header(bot_id="bot-with-dashes_and_underscores")

        output = mock_stdout.getvalue()
        assert "Bot ID: bot-with-dashes_and_underscores" in output


# ============================================================================
# Test: Portfolio Summary Rendering
# ============================================================================


class TestConsoleRendererPortfolioSummary:
    """Test portfolio summary rendering."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_portfolio_summary(self, mock_stdout):
        """Test portfolio summary rendering."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        metrics = {
            "equity": 105000.50,
            "cash": 50000.25,
            "positions_value": 55000.25,
            "returns_pct": 5.0,
            "drawdown_pct": -2.5,
            "exposure_pct": 52.38,
        }

        renderer.render_portfolio_summary(metrics)

        output = mock_stdout.getvalue()
        assert "PORTFOLIO SUMMARY" in output
        assert "$105,000.50" in output
        assert "$50,000.25" in output
        assert "$55,000.25" in output
        assert "+5.00%" in output
        assert "-2.50%" in output
        assert "+52.38%" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_portfolio_summary_negative_returns(self, mock_stdout):
        """Test portfolio summary with negative returns."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        metrics = {
            "equity": 95000.0,
            "cash": 50000.0,
            "positions_value": 45000.0,
            "returns_pct": -5.0,
            "drawdown_pct": -10.0,
            "exposure_pct": 47.37,
        }

        renderer.render_portfolio_summary(metrics)

        output = mock_stdout.getvalue()
        assert "$95,000.00" in output
        assert "-5.00%" in output
        assert "-10.00%" in output


# ============================================================================
# Test: Positions Rendering
# ============================================================================


class TestConsoleRendererPositions:
    """Test positions rendering."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_positions_empty(self, mock_stdout):
        """Test rendering with no open positions."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        renderer.render_positions({})

        output = mock_stdout.getvalue()
        assert "OPEN POSITIONS" in output
        assert "No open positions" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_positions_with_data(self, mock_stdout):
        """Test rendering with open positions."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        position = Mock()
        position.quantity = 10.0
        position.entry_price = 150.0
        position.current_price = 155.0

        positions = {"AAPL": position}

        renderer.render_positions(positions)

        output = mock_stdout.getvalue()
        assert "OPEN POSITIONS" in output
        assert "AAPL" in output
        assert "10.000000" in output
        assert "$150.00" in output
        assert "$155.00" in output
        assert "$50.00" in output  # P&L = (155-150)*10

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_positions_with_multiple_symbols(self, mock_stdout):
        """Test rendering with multiple positions."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        position1 = Mock()
        position1.quantity = 10.0
        position1.entry_price = 150.0
        position1.current_price = 155.0

        position2 = Mock()
        position2.quantity = 5.0
        position2.entry_price = 200.0
        position2.current_price = 195.0

        positions = {"AAPL": position1, "MSFT": position2}

        renderer.render_positions(positions)

        output = mock_stdout.getvalue()
        assert "AAPL" in output
        assert "MSFT" in output
        assert "10.000000" in output
        assert "5.000000" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_positions_with_zero_current_price(self, mock_stdout):
        """Test rendering when current_price is zero (uses entry_price)."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        position = Mock()
        position.quantity = 10.0
        position.entry_price = 150.0
        position.current_price = 0.0  # Zero current price

        positions = {"AAPL": position}

        renderer.render_positions(positions)

        output = mock_stdout.getvalue()
        assert "AAPL" in output
        # Should use entry_price when current_price is 0
        assert "$150.00" in output


# ============================================================================
# Test: Performance Rendering
# ============================================================================


class TestConsoleRendererPerformance:
    """Test performance metrics rendering."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_performance(self, mock_stdout):
        """Test performance metrics rendering."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        metrics = {
            "total_trades": 100,
            "winning_trades": 60,
            "losing_trades": 40,
            "win_rate": 60.0,
        }

        renderer.render_performance(metrics)

        output = mock_stdout.getvalue()
        assert "PERFORMANCE METRICS" in output
        assert "Total Trades:" in output
        assert "100" in output
        assert "Winning Trades:" in output
        assert "60" in output
        assert "Losing Trades:" in output
        assert "40" in output
        assert "Win Rate:" in output
        assert "60.0%" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_performance_zero_trades(self, mock_stdout):
        """Test performance metrics with zero trades."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        metrics = {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate": 0.0}

        renderer.render_performance(metrics)

        output = mock_stdout.getvalue()
        assert "0" in output
        assert "0.0%" in output


# ============================================================================
# Test: Recent Trades Rendering
# ============================================================================


class TestConsoleRendererRecentTrades:
    """Test recent trades rendering."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_recent_trades_empty(self, mock_stdout):
        """Test rendering with no trades."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        renderer.render_recent_trades([])

        output = mock_stdout.getvalue()
        assert "RECENT TRADES" in output
        assert "No trades executed" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_recent_trades_with_data(self, mock_stdout):
        """Test rendering with trades."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        trade1 = Mock()
        trade1.timestamp = datetime(2025, 10, 3, 10, 30, 0)
        trade1.symbol = "AAPL"
        trade1.side = "buy"
        trade1.quantity = 10.0
        trade1.price = 150.0
        trade1.pnl = None

        trade2 = Mock()
        trade2.timestamp = datetime(2025, 10, 3, 11, 30, 0)
        trade2.symbol = "MSFT"
        trade2.side = "sell"
        trade2.quantity = 5.0
        trade2.price = 200.0
        trade2.pnl = 50.0

        trades = [trade1, trade2]

        renderer.render_recent_trades(trades)

        output = mock_stdout.getvalue()
        assert "RECENT TRADES" in output
        assert "AAPL" in output
        assert "MSFT" in output
        assert "buy" in output
        assert "sell" in output
        assert "$150.00" in output
        assert "$200.00" in output
        assert "$50.00" in output  # P&L for second trade

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_recent_trades_with_limit(self, mock_stdout):
        """Test rendering with limit parameter."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        trades = []
        for i in range(10):
            trade = Mock()
            trade.timestamp = datetime(2025, 10, 3, 10, i, 0)
            trade.symbol = f"SYM{i}"
            trade.side = "buy"
            trade.quantity = float(i)
            trade.price = 100.0 + i
            trade.pnl = None
            trades.append(trade)

        renderer.render_recent_trades(trades, limit=3)

        output = mock_stdout.getvalue()
        assert "Last 3" in output
        # Should show last 3 trades (reversed)
        assert "SYM9" in output
        assert "SYM8" in output
        assert "SYM7" in output
        # Should NOT show earlier trades
        assert "SYM0" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_render_recent_trades_fewer_than_limit(self, mock_stdout):
        """Test rendering when trades are fewer than limit."""
        formatter = DashboardFormatter()
        renderer = ConsoleRenderer(formatter, datetime.now())

        trade = Mock()
        trade.timestamp = datetime(2025, 10, 3, 10, 30, 0)
        trade.symbol = "AAPL"
        trade.side = "buy"
        trade.quantity = 10.0
        trade.price = 150.0
        trade.pnl = None

        trades = [trade]

        renderer.render_recent_trades(trades, limit=5)

        output = mock_stdout.getvalue()
        assert "Last 5" in output
        assert "AAPL" in output
