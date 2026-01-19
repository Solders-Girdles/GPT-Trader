"""
Screen Flow Integration Tests: Shortcut screens and modals.

Tests navigation flows between TUI screens using Textual's Pilot API.
"""

from __future__ import annotations

import pytest

from gpt_trader.tui.app import TraderApp


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
