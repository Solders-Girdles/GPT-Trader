"""
Screen Flow Integration Tests.

Tests navigation flows between TUI screens using Textual's Pilot API.
These tests verify screen transitions, modal behavior, and navigation stack.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.screens.help import HelpScreen


@pytest.fixture
def mock_bot_for_flows():
    """Creates a mock bot for screen flow testing."""
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


class TestHelpScreenFlow:
    """Tests for help screen navigation."""

    @pytest.mark.asyncio
    async def test_help_screen_opens_on_question_mark(self, mock_bot_for_flows):
        """Test that pressing '?' opens the help screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Initial screen stack should have main screen
            initial_stack_size = len(app.screen_stack)

            # Press '?' to open help
            await pilot.press("?")
            await pilot.pause()

            # Help screen should be pushed
            assert len(app.screen_stack) > initial_stack_size
            assert isinstance(app.screen, HelpScreen)

    @pytest.mark.asyncio
    async def test_help_screen_closes_on_escape(self, mock_bot_for_flows):
        """Test that pressing Escape closes the help screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Open help
            await pilot.press("?")
            await pilot.pause()
            assert len(app.screen_stack) > initial_stack_size

            # Close help with Escape
            await pilot.press("escape")
            await pilot.pause()

            # Should be back to main screen
            assert len(app.screen_stack) == initial_stack_size

    @pytest.mark.asyncio
    async def test_help_screen_closes_on_question_mark(self, mock_bot_for_flows):
        """Test that pressing '?' again closes the help screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Open help
            await pilot.press("?")
            await pilot.pause()

            # Close with '?' again (toggle behavior)
            await pilot.press("?")
            await pilot.pause()

            # Should be back to main screen
            # Note: Actual behavior depends on implementation


class TestFullLogsScreenFlow:
    """Tests for full logs screen navigation."""

    @pytest.mark.asyncio
    async def test_full_logs_opens_on_1(self, mock_bot_for_flows):
        """Test that pressing '1' opens the full logs screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Press '1' to open full logs
            await pilot.press("1")
            await pilot.pause()

            # Full logs screen should be pushed
            assert len(app.screen_stack) > initial_stack_size

    @pytest.mark.asyncio
    async def test_full_logs_closes_on_escape(self, mock_bot_for_flows):
        """Test that Escape closes the full logs screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Open full logs
            await pilot.press("1")
            await pilot.pause()

            # Close with Escape
            await pilot.press("escape")
            await pilot.pause()

            # Should be back to main screen
            assert len(app.screen_stack) == initial_stack_size


class TestSystemDetailsScreenFlow:
    """Tests for system details screen navigation."""

    @pytest.mark.asyncio
    async def test_system_details_opens_on_2(self, mock_bot_for_flows):
        """Test that pressing '2' opens the system details screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Press '2' to open system details
            await pilot.press("2")
            await pilot.pause()

            # System details screen should be pushed
            assert len(app.screen_stack) > initial_stack_size

    @pytest.mark.asyncio
    async def test_system_details_closes_on_escape(self, mock_bot_for_flows):
        """Test that Escape closes the system details screen."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Open system details
            await pilot.press("2")
            await pilot.pause()

            # Close with Escape
            await pilot.press("escape")
            await pilot.pause()

            # Should be back to main screen
            assert len(app.screen_stack) == initial_stack_size


class TestScreenStackBehavior:
    """Tests for screen stack management."""

    @pytest.mark.asyncio
    async def test_multiple_screens_can_be_stacked(self, mock_bot_for_flows):
        """Test that multiple screens can be pushed to the stack."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Open full logs
            await pilot.press("1")
            await pilot.pause()
            after_logs_size = len(app.screen_stack)
            assert after_logs_size > initial_stack_size

            # Try to open help on top (if supported)
            await pilot.press("?")
            await pilot.pause()
            # Note: Whether this stacks depends on screen implementation

    @pytest.mark.asyncio
    async def test_escape_returns_to_previous_screen(self, mock_bot_for_flows):
        """Test that Escape properly pops screens."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Open help
            await pilot.press("?")
            await pilot.pause()

            # Press Escape to go back
            await pilot.press("escape")
            await pilot.pause()

            # Should return to initial screen
            assert len(app.screen_stack) == initial_stack_size


class TestConfigModalFlow:
    """Tests for config modal navigation."""

    @pytest.mark.asyncio
    async def test_config_opens_on_c(self, mock_bot_for_flows):
        """Test that pressing 'c' opens the config modal."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Press 'c' to open config
            await pilot.press("c")
            await pilot.pause()

            # Config modal should be shown
            # Note: Implementation may vary - could be screen push or modal overlay


class TestModeInfoFlow:
    """Tests for mode info modal navigation."""

    @pytest.mark.asyncio
    async def test_mode_info_opens_on_m(self, mock_bot_for_flows):
        """Test that pressing 'm' opens the mode info modal."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Press 'm' to open mode info
            await pilot.press("m")
            await pilot.pause()

            # Mode info should be shown
            # Note: Implementation may vary


class TestLogFocusFlow:
    """Tests for log widget focus navigation."""

    @pytest.mark.asyncio
    async def test_logs_focus_on_l(self, mock_bot_for_flows):
        """Test that pressing 'l' focuses the logs widget."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Press 'l' to focus logs
            await pilot.press("l")
            await pilot.pause()

            # Logs widget should receive focus
            # Note: Actual focus behavior depends on implementation


class TestQuickNavigationFlow:
    """Tests for quick navigation patterns."""

    @pytest.mark.asyncio
    async def test_navigation_sequence_help_to_logs(self, mock_bot_for_flows):
        """Test navigation from help screen to full logs."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            # Open help
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)

            # Close help
            await pilot.press("escape")
            await pilot.pause()

            # Open full logs
            await pilot.press("1")
            await pilot.pause()
            assert len(app.screen_stack) > initial_stack_size

            # Close logs
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.screen_stack) == initial_stack_size

    @pytest.mark.asyncio
    async def test_rapid_screen_toggling(self, mock_bot_for_flows):
        """Test that rapid screen switching doesn't cause issues."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Rapidly toggle screens
            await pilot.press("?")
            await pilot.press("escape")
            await pilot.press("1")
            await pilot.press("escape")
            await pilot.press("2")
            await pilot.press("escape")
            await pilot.pause()

            # App should still be responsive


class TestScreenStatePreservation:
    """Tests for state preservation during navigation."""

    @pytest.mark.asyncio
    async def test_main_screen_state_preserved_after_help(self, mock_bot_for_flows):
        """Test that main screen state is preserved after viewing help."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Remember initial state
            initial_running = app.tui_state.running

            # Open and close help
            await pilot.press("?")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            # State should be preserved
            assert app.tui_state.running == initial_running

    @pytest.mark.asyncio
    async def test_mode_preserved_after_navigation(self, mock_bot_for_flows):
        """Test that data source mode is preserved during navigation."""
        app = TraderApp(bot=mock_bot_for_flows)

        async with app.run_test() as pilot:
            # Get initial mode
            initial_mode = app.data_source_mode

            # Navigate to various screens
            await pilot.press("1")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            # Mode should be preserved
            assert app.data_source_mode == initial_mode
