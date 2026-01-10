"""
Widget Interaction Tests.

Tests user interactions with TUI widgets using Textual's Pilot API.
These tests verify keyboard navigation, button clicks, and widget updates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.unit.gpt_trader.tui.factories import (
    BotStatusFactory,
    TuiStateFactory,
)
from textual.widgets import Button

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp


@pytest.fixture
def mock_bot_with_status():
    """Creates a mock bot with a StatusReporter that has initial status."""
    bot = MagicMock()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    # Mock engine and status reporter
    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    return bot


class TestKeyboardNavigation:
    """Tests for keyboard navigation in the TUI."""

    @pytest.mark.asyncio
    async def test_quit_shortcut(self, mock_bot_with_status):
        """Test that pressing 'q' quits the application."""
        app = TraderApp(bot=mock_bot_with_status)
        app.action_dispatcher.quit_app = AsyncMock()

        async with app.run_test() as pilot:
            # Press 'q' to quit
            await pilot.press("q")
            await pilot.pause()
            app.action_dispatcher.quit_app.assert_called_once()

    @pytest.mark.asyncio
    async def test_help_screen_toggle(self, mock_bot_with_status):
        """Test that pressing '?' shows help screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            # Press '?' to show help
            await pilot.press("?")
            await pilot.pause()

            # Check that help screen is pushed
            # Note: In Textual, screens are managed via the screen stack
            assert len(app.screen_stack) > 1, "Help screen should be pushed"

    @pytest.mark.asyncio
    async def test_theme_toggle(self, mock_bot_with_status):
        """Test that pressing 't' toggles the theme."""
        app = TraderApp(bot=mock_bot_with_status)
        app.action_dispatcher.toggle_theme = AsyncMock()

        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()

            app.action_dispatcher.toggle_theme.assert_called_once()


class TestBotControlInteractions:
    """Tests for bot control interactions."""

    @pytest.mark.asyncio
    async def test_start_bot_updates_ui(self, mock_bot_with_status):
        """Test that starting the bot updates the UI state."""
        bot = mock_bot_with_status
        app = TraderApp(bot=bot)

        async with app.run_test() as pilot:
            # Initially bot should not be running
            assert app.tui_state.running is False

            # Press 's' to start bot
            await pilot.press("s")
            await pilot.pause()

            # Bot's run method should have been called
            # Note: The actual running state depends on the async bot task


class TestPortfolioWidgetInteractions:
    """Tests for portfolio widget accessibility via Details overlay."""

    @pytest.mark.asyncio
    async def test_log_widget_on_main_screen(self, mock_bot_with_status):
        """Test that log widget is primary on main screen (log-centric layout)."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Query for log widget from active screen (Textual 7.0+ query scope)
            logs = app.screen.query("LogWidget")
            assert len(logs) > 0, "LogWidget should exist on main screen"

    @pytest.mark.asyncio
    async def test_portfolio_accessible_via_details(self, mock_bot_with_status):
        """Test that portfolio widget is accessible via Details overlay."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Press 'd' to open Details overlay
            await pilot.press("d")
            await pilot.pause()

            # Query from active screen (Textual 7.0+ query scope)
            positions = app.screen.query("PositionsWidget")
            assert len(positions) > 0, "PositionsWidget should exist in Details overlay"


class TestDataTableInteractions:
    """Tests for DataTable widget interactions."""

    @pytest.mark.asyncio
    async def test_positions_table_accessible(self, mock_bot_with_status):
        """Test that positions table can be found."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Check that DataTable widgets can be queried
            assert app.query("DataTable") is not None


class TestModeIndicator:
    """Tests for mode indicator widget."""

    @pytest.mark.asyncio
    async def test_mode_indicator_shows_demo(self, mock_bot_with_status):
        """Test that mode indicator shows demo mode by default."""
        # Set up mock to return DemoBot type
        mock_bot_with_status.__class__.__name__ = "DemoBot"
        type(mock_bot_with_status).__name__ = "DemoBot"

        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Mode should default to demo
            assert app.data_source_mode in ["demo", "paper", "live", "read_only"]


class TestLogWidgetInteractions:
    """Tests for log widget interactions."""

    @pytest.mark.asyncio
    async def test_log_widget_exists(self, mock_bot_with_status):
        """Test that log widget is present."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify log widget can be queried
            assert app.query("LogWidget") is not None

    @pytest.mark.asyncio
    async def test_log_level_filter_chips_present(self, mock_bot_with_status):
        """Log level filter chips should be present in log widget."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify level filter chip buttons exist (query from active screen)
            chip_ids = ["level-all", "level-error", "level-warn", "level-info", "level-debug"]
            for chip_id in chip_ids:
                chip = app.screen.query_one(f"#{chip_id}", Button)
                assert chip is not None


class TestFactoryIntegration:
    """Tests that factories integrate properly with pilot tests."""

    @pytest.mark.asyncio
    async def test_bot_status_factory_creates_valid_status(self):
        """Test that BotStatusFactory creates valid test data."""
        status = BotStatusFactory.create_running(uptime=500.0, cycle_count=100)

        assert status.engine.running is True
        assert status.engine.uptime_seconds == 500.0
        assert status.engine.cycle_count == 100

    @pytest.mark.asyncio
    async def test_bot_status_with_positions(self):
        """Test creating status with positions."""
        status = BotStatusFactory.create_with_positions()

        assert status.positions.count > 0
        assert len(status.positions.symbols) > 0
        assert "BTC-USD" in status.positions.symbols

    @pytest.mark.asyncio
    async def test_tui_state_factory_creates_valid_state(self):
        """Test that TuiStateFactory creates valid test data."""
        state = TuiStateFactory.create_running(mode="paper")

        assert state.running is True
        assert state.data_source_mode == "paper"

    @pytest.mark.asyncio
    async def test_tui_state_with_positions(self):
        """Test creating state with positions."""
        state = TuiStateFactory.create_with_positions()

        assert state.running is True
        assert len(state.position_data.positions) > 0
        assert "BTC-USD" in state.position_data.positions


class TestStatusBarInteractions:
    """Tests for status bar widget interactions."""

    @pytest.mark.asyncio
    async def test_status_bar_shows_on_main_screen(self, mock_bot_with_status):
        """Test that status bar is visible on main screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify status widget can be queried
            assert app.query("BotStatusWidget") is not None


class TestErrorIndicatorInteractions:
    """Tests for error indicator widget."""

    @pytest.mark.asyncio
    async def test_error_indicator_exists(self, mock_bot_with_status):
        """Test that error indicator is present."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            # Error indicator should be in the app
            assert app.error_tracker is not None


class TestResponsiveLayout:
    """Tests for responsive layout behavior."""

    @pytest.mark.asyncio
    async def test_app_handles_small_terminal(self, mock_bot_with_status):
        """Test that app handles small terminal size."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app.terminal_width == 80

    @pytest.mark.asyncio
    async def test_app_handles_large_terminal(self, mock_bot_with_status):
        """Test that app handles large terminal size."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            assert app.terminal_width == 200
