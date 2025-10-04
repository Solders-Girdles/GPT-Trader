"""
Tests for HTML report generator.

Tests cover:
- HTML generation with default path
- HTML generation with custom path
- Empty positions handling
- Empty trades handling
- Directory creation
- Content verification
- CSS styles presence
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from bot_v2.features.paper_trade.dashboard.formatters import DashboardFormatter
from bot_v2.features.paper_trade.dashboard.html_report_generator import HTMLReportGenerator


# ============================================================================
# Helper Functions
# ============================================================================


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
# Test: HTML Generation with Paths
# ============================================================================


class TestHTMLReportGeneratorPaths:
    """Test HTML generation with different paths."""

    def test_generate_with_custom_path(self):
        """Test HTML generation with custom output path."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "custom_report.html")
            result_path = generator.generate(
                bot_id="test-bot",
                metrics=metrics,
                positions={},
                trades=[],
                output_path=output_path,
            )

            assert result_path == output_path
            assert Path(output_path).exists()

    def test_generate_with_default_path(self):
        """Test HTML generation with default path."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        result_path = generator.generate(
            bot_id="test-bot", metrics=metrics, positions={}, trades=[], output_path=None
        )

        # Should create file with default naming
        assert "paper_trading_summary_test-bot" in result_path
        assert result_path.endswith(".html")
        assert Path(result_path).exists()

        # Cleanup
        if Path(result_path).exists():
            Path(result_path).unlink()

    def test_generate_creates_directories(self):
        """Test that HTML generation creates parent directories."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "nested" / "dir" / "report.html")
            result_path = generator.generate(
                bot_id="test-bot",
                metrics=metrics,
                positions={},
                trades=[],
                output_path=output_path,
            )

            assert Path(output_path).exists()
            assert Path(output_path).parent.exists()


# ============================================================================
# Test: HTML Content Verification
# ============================================================================


class TestHTMLReportGeneratorContent:
    """Test HTML content generation."""

    def test_html_contains_bot_id(self):
        """Test HTML contains bot ID."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot-123",
                metrics=metrics,
                positions={},
                trades=[],
                output_path=output_path,
            )

            with open(output_path) as f:
                content = f.read()

            assert "test-bot-123" in content
            assert "Bot ID:" in content

    def test_html_contains_metrics(self):
        """Test HTML contains portfolio metrics."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 105000.50,
            "cash": 50000.25,
            "returns_pct": 5.0,
            "drawdown_pct": -2.5,
            "win_rate": 60.0,
            "total_trades": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot", metrics=metrics, positions={}, trades=[], output_path=output_path
            )

            with open(output_path) as f:
                content = f.read()

            assert "$105,000.50" in content
            assert "$50,000.25" in content
            assert "+5.00%" in content
            assert "-2.50%" in content
            assert "60.0%" in content
            assert "10" in content
            assert "Portfolio Metrics" in content

    def test_html_contains_positions(self):
        """Test HTML contains position data."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        positions = {
            "AAPL": create_mock_position("AAPL", 100, 150.0, 155.0),
            "MSFT": create_mock_position("MSFT", 50, 300.0, 310.0),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot",
                metrics=metrics,
                positions=positions,
                trades=[],
                output_path=output_path,
            )

            with open(output_path) as f:
                content = f.read()

            assert "AAPL" in content
            assert "MSFT" in content
            assert "100.000000" in content  # AAPL quantity
            assert "50.000000" in content  # MSFT quantity
            assert "$150.00" in content  # AAPL entry
            assert "$155.00" in content  # AAPL current
            assert "Open Positions" in content

    def test_html_contains_trades(self):
        """Test HTML contains trade data."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 2,
        }

        trades = [
            create_mock_trade(
                "AAPL", "buy", 100, 150.0, datetime(2025, 10, 3, 10, 30, 0), pnl=None
            ),
            create_mock_trade(
                "AAPL", "sell", 100, 155.0, datetime(2025, 10, 3, 11, 30, 0), pnl=500.0
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot",
                metrics=metrics,
                positions={},
                trades=trades,
                output_path=output_path,
            )

            with open(output_path) as f:
                content = f.read()

            assert "AAPL" in content
            assert "BUY" in content
            assert "SELL" in content
            assert "$150.00" in content
            assert "$155.00" in content
            assert "$500.00" in content
            assert "Recent Trades" in content

    def test_html_structure(self):
        """Test HTML has correct structure."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot", metrics=metrics, positions={}, trades=[], output_path=output_path
            )

            with open(output_path) as f:
                content = f.read()

            # Check structure
            assert "<!DOCTYPE html>" in content
            assert "<html>" in content
            assert "</html>" in content
            assert "<head>" in content
            assert "<body>" in content
            assert "<style>" in content
            assert "Paper Trading Summary Report" in content


# ============================================================================
# Test: Empty State Handling
# ============================================================================


class TestHTMLReportGeneratorEmptyStates:
    """Test HTML generation with empty states."""

    def test_html_with_empty_positions(self):
        """Test HTML handles empty positions."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot", metrics=metrics, positions={}, trades=[], output_path=output_path
            )

            with open(output_path) as f:
                content = f.read()

            assert "No open positions" in content

    def test_html_with_empty_trades(self):
        """Test HTML handles empty trades."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot", metrics=metrics, positions={}, trades=[], output_path=output_path
            )

            with open(output_path) as f:
                content = f.read()

            assert "No trades executed" in content


# ============================================================================
# Test: CSS Styles
# ============================================================================


class TestHTMLReportGeneratorStyles:
    """Test CSS styles in generated HTML."""

    def test_html_contains_css_styles(self):
        """Test HTML contains CSS styles."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot", metrics=metrics, positions={}, trades=[], output_path=output_path
            )

            with open(output_path) as f:
                content = f.read()

            # Check CSS classes
            assert ".container" in content
            assert ".metrics" in content
            assert ".positive" in content
            assert ".negative" in content
            assert "table" in content
            assert ".footer" in content


# ============================================================================
# Test: Trade Limit
# ============================================================================


class TestHTMLReportGeneratorTradeLimit:
    """Test recent trades limit."""

    def test_html_limits_trades_to_ten(self):
        """Test HTML shows only last 10 trades."""
        formatter = DashboardFormatter()
        generator = HTMLReportGenerator(formatter)

        metrics = {
            "equity": 100000,
            "cash": 100000,
            "returns_pct": 0.0,
            "drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 15,
        }

        # Create 15 trades
        trades = [create_mock_trade(f"SYM{i}", "buy", 100, 100.0 + i, pnl=None) for i in range(15)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")
            generator.generate(
                bot_id="test-bot",
                metrics=metrics,
                positions={},
                trades=trades,
                output_path=output_path,
            )

            with open(output_path) as f:
                content = f.read()

            # Should show last 10 trades (SYM5 through SYM14, reversed)
            assert "SYM14" in content
            assert "SYM5" in content
            # Should NOT show first 5 trades
            assert "SYM0" not in content
            assert "SYM4" not in content
