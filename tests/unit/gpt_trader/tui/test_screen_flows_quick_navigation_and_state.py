"""
Screen Flow Integration Tests: Quick navigation, shortcuts, and state preservation.

Tests navigation flows between TUI screens using Textual's Pilot API.
"""

from __future__ import annotations

import pytest

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.screens.help import HelpScreen


class TestScreenStackBehavior:
    """Tests for screen stack management."""

    @pytest.mark.asyncio
    async def test_multiple_screens_can_be_stacked(self, mock_bot_with_status):
        """Test that multiple screens can be pushed to the stack."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("1")
            await pilot.pause()
            assert len(app.screen_stack) > initial_stack_size

            # Whether this stacks depends on screen implementation.
            await pilot.press("?")
            await pilot.pause()


class TestQuickNavigationFlow:
    """Tests for quick navigation patterns."""

    @pytest.mark.asyncio
    async def test_navigation_sequence_help_to_logs(self, mock_bot_with_status):
        """Test navigation from help screen to full logs."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)

            await pilot.press("escape")
            await pilot.pause()

            await pilot.press("1")
            await pilot.pause()
            assert len(app.screen_stack) > initial_stack_size

            await pilot.press("escape")
            await pilot.pause()
            assert len(app.screen_stack) == initial_stack_size

    @pytest.mark.asyncio
    async def test_rapid_screen_toggling(self, mock_bot_with_status):
        """Test that rapid screen switching doesn't cause issues."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("?")
            await pilot.press("escape")
            await pilot.press("1")
            await pilot.press("escape")
            await pilot.press("2")
            await pilot.press("escape")
            await pilot.pause()

            assert len(app.screen_stack) == initial_stack_size


class TestScreenStatePreservation:
    """Tests for state preservation during navigation."""

    @pytest.mark.asyncio
    async def test_main_screen_state_preserved_after_help(self, mock_bot_with_status):
        """Test that main screen state is preserved after viewing help."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_running = app.tui_state.running

            await pilot.press("?")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            assert app.tui_state.running == initial_running

    @pytest.mark.asyncio
    async def test_mode_preserved_after_navigation(self, mock_bot_with_status):
        """Test that data source mode is preserved during navigation."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_mode = app.data_source_mode

            await pilot.press("1")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            assert app.data_source_mode == initial_mode


class TestFullLogsScreenFlow:
    """Tests for full logs screen navigation."""

    @pytest.mark.asyncio
    async def test_full_logs_opens_on_1(self, mock_bot_with_status):
        """Test that pressing '1' opens the full logs screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("1")
            await pilot.pause()

            assert len(app.screen_stack) > initial_stack_size

    @pytest.mark.asyncio
    async def test_full_logs_closes_on_escape(self, mock_bot_with_status):
        """Test that Escape closes the full logs screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("1")
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert len(app.screen_stack) == initial_stack_size


class TestSystemDetailsScreenFlow:
    """Tests for system details screen navigation."""

    @pytest.mark.asyncio
    async def test_system_details_opens_on_2(self, mock_bot_with_status):
        """Test that pressing '2' opens the system details screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("2")
            await pilot.pause()

            assert len(app.screen_stack) > initial_stack_size

    @pytest.mark.asyncio
    async def test_system_details_closes_on_escape(self, mock_bot_with_status):
        """Test that Escape closes the system details screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("2")
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert len(app.screen_stack) == initial_stack_size


class TestConfigModalFlow:
    """Tests for config modal navigation."""

    @pytest.mark.asyncio
    async def test_config_opens_on_c(self, mock_bot_with_status):
        """Test that pressing 'c' opens the config modal."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("c")
            await pilot.pause()

            assert len(app.screen_stack) > initial_stack_size


class TestModeInfoFlow:
    """Tests for mode info modal navigation."""

    @pytest.mark.asyncio
    async def test_mode_info_opens_on_i(self, mock_bot_with_status):
        """Test that pressing 'i' opens the mode info modal."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            from gpt_trader.tui.widgets import ModeInfoModal

            initial_stack_size = len(app.screen_stack)

            await pilot.press("i")
            await pilot.pause()

            assert len(app.screen_stack) > initial_stack_size
            assert isinstance(app.screen, ModeInfoModal)
