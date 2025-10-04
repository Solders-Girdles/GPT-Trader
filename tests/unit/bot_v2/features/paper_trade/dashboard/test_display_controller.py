"""
Tests for display controller.

Tests cover:
- Screen clearing
- Single display pass
- Continuous display with duration
- Continuous display with keyboard interrupt
- Refresh interval behavior
"""

from io import StringIO
from unittest.mock import Mock, patch, call

import pytest

from bot_v2.features.paper_trade.dashboard.display_controller import DisplayController


# ============================================================================
# Test: Screen Clearing
# ============================================================================


class TestDisplayControllerScreenClearing:
    """Test screen clearing functionality."""

    @patch("os.system")
    def test_clear_screen_posix(self, mock_system):
        """Test screen clearing on POSIX systems."""
        dashboard = Mock()
        controller = DisplayController(dashboard, refresh_interval=5)

        with patch("os.name", "posix"):
            controller.clear_screen()

        mock_system.assert_called_once_with("clear")

    @patch("os.system")
    def test_clear_screen_windows(self, mock_system):
        """Test screen clearing on Windows systems."""
        dashboard = Mock()
        controller = DisplayController(dashboard, refresh_interval=5)

        with patch("os.name", "nt"):
            controller.clear_screen()

        mock_system.assert_called_once_with("cls")


# ============================================================================
# Test: Single Display Pass
# ============================================================================


class TestDisplayControllerSingleDisplay:
    """Test single display pass functionality."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_display_once_calls_all_methods(self, mock_stdout):
        """Test that display_once calls all dashboard print methods."""
        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {
            "equity": 100000,
            "returns_pct": 0.0,
            "cash": 100000,
        }

        controller = DisplayController(dashboard, refresh_interval=5)
        controller.display_once()

        # Verify all dashboard methods were called
        dashboard.calculate_metrics.assert_called_once()
        dashboard.print_header.assert_called_once()
        dashboard.print_portfolio_summary.assert_called_once()
        dashboard.print_positions.assert_called_once()
        dashboard.print_performance.assert_called_once()
        dashboard.print_recent_trades.assert_called_once()

        # Verify metrics passed to summary and performance
        metrics = dashboard.calculate_metrics.return_value
        dashboard.print_portfolio_summary.assert_called_with(metrics)
        dashboard.print_performance.assert_called_with(metrics)

    @patch("sys.stdout", new_callable=StringIO)
    def test_display_once_prints_footer(self, mock_stdout):
        """Test that display_once prints footer separator."""
        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {}

        controller = DisplayController(dashboard, refresh_interval=5)
        controller.display_once()

        output = mock_stdout.getvalue()
        assert "=" * 80 in output


# ============================================================================
# Test: Continuous Display
# ============================================================================


class TestDisplayControllerContinuousDisplay:
    """Test continuous display functionality."""

    @patch("time.sleep")
    @patch("time.time")
    @patch("os.system")
    @patch("sys.stdout", new_callable=StringIO)
    def test_display_continuous_with_duration(
        self, mock_stdout, mock_system, mock_time, mock_sleep
    ):
        """Test continuous display stops after duration."""
        # Mock time: start=0, check at 0 (continue), check at 11 (break)
        mock_time.side_effect = [0, 0, 11]

        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {}

        controller = DisplayController(dashboard, refresh_interval=5)
        controller.display_continuous(duration=10)

        # Should have 2 iterations: 1st at 0s (sleep), 2nd at 11s (break before sleep)
        assert mock_system.call_count == 2
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(5)

    @patch("time.sleep", side_effect=KeyboardInterrupt)
    @patch("time.time", return_value=0)
    @patch("os.system")
    @patch("sys.stdout", new_callable=StringIO)
    def test_display_continuous_keyboard_interrupt(
        self, mock_stdout, mock_system, mock_time, mock_sleep
    ):
        """Test continuous display handles keyboard interrupt gracefully."""
        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {}

        controller = DisplayController(dashboard, refresh_interval=5)
        controller.display_continuous()

        # Should display once, then get interrupted during sleep
        assert mock_system.call_count == 1
        output = mock_stdout.getvalue()
        assert "Dashboard stopped by user" in output

    @patch("time.sleep")
    @patch("time.time", return_value=0)
    @patch("os.system")
    @patch("sys.stdout", new_callable=StringIO)
    def test_display_continuous_shows_countdown(
        self, mock_stdout, mock_system, mock_time, mock_sleep
    ):
        """Test continuous display shows refresh countdown."""
        # Make sleep raise exception after first iteration to stop loop
        mock_sleep.side_effect = [None, KeyboardInterrupt]

        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {}

        controller = DisplayController(dashboard, refresh_interval=10)
        controller.display_continuous()

        output = mock_stdout.getvalue()
        assert "Refreshing in 10 seconds" in output
        assert "Press Ctrl+C to stop" in output

    @patch("time.sleep")
    @patch("time.time")
    @patch("os.system")
    @patch("sys.stdout", new_callable=StringIO)
    def test_display_continuous_clears_screen_each_iteration(
        self, mock_stdout, mock_system, mock_time, mock_sleep
    ):
        """Test continuous display clears screen before each display."""
        # Mock time: start=0, check at 0 (continue), check at 11 (break)
        mock_time.side_effect = [0, 0, 11]

        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {}

        controller = DisplayController(dashboard, refresh_interval=5)
        controller.display_continuous(duration=10)

        # Should clear screen twice (once per iteration)
        assert mock_system.call_count == 2

    @patch("time.sleep")
    @patch("time.time", return_value=0)
    @patch("os.system")
    def test_display_continuous_none_duration_runs_until_interrupt(
        self, mock_system, mock_time, mock_sleep
    ):
        """Test continuous display with None duration runs until interrupted."""
        # Run 3 iterations then interrupt
        mock_sleep.side_effect = [None, None, None, KeyboardInterrupt]

        dashboard = Mock()
        dashboard.calculate_metrics.return_value = {}

        controller = DisplayController(dashboard, refresh_interval=5)
        controller.display_continuous(duration=None)

        # Should have 4 iterations (3 sleeps + 1 interrupted)
        assert mock_system.call_count == 4


# ============================================================================
# Test: Refresh Interval
# ============================================================================


class TestDisplayControllerRefreshInterval:
    """Test refresh interval configuration."""

    def test_controller_uses_custom_refresh_interval(self):
        """Test controller respects custom refresh interval."""
        dashboard = Mock()
        controller = DisplayController(dashboard, refresh_interval=10)

        assert controller.refresh_interval == 10

    def test_controller_default_refresh_interval(self):
        """Test controller uses default refresh interval."""
        dashboard = Mock()
        controller = DisplayController(dashboard)

        # Default should be 5 (from parameter default)
        assert controller.refresh_interval == 5
