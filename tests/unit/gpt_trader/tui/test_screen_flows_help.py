"""
Screen Flow Integration Tests: Help screen.

Tests navigation flows between TUI screens using Textual's Pilot API.
"""

from __future__ import annotations

import pytest

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.screens.help import HelpScreen


class TestHelpScreenFlow:
    """Tests for help screen navigation."""

    @pytest.mark.asyncio
    async def test_help_screen_opens_on_question_mark(self, mock_bot_with_status):
        """Test that pressing '?' opens the help screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("?")
            await pilot.pause()

            assert len(app.screen_stack) > initial_stack_size
            assert isinstance(app.screen, HelpScreen)

    @pytest.mark.asyncio
    async def test_help_screen_closes_on_escape(self, mock_bot_with_status):
        """Test that pressing Escape closes the help screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("?")
            await pilot.pause()
            assert len(app.screen_stack) > initial_stack_size

            await pilot.press("escape")
            await pilot.pause()

            assert len(app.screen_stack) == initial_stack_size

    @pytest.mark.asyncio
    async def test_help_screen_closes_on_question_mark(self, mock_bot_with_status):
        """Test that pressing '?' again closes the help screen."""
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            initial_stack_size = len(app.screen_stack)

            await pilot.press("?")
            await pilot.pause()
            assert len(app.screen_stack) > initial_stack_size

            await pilot.press("?")
            await pilot.pause()

            assert len(app.screen_stack) == initial_stack_size
