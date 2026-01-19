"""
Widget interaction tests: keyboard shortcuts and bot controls.

Uses Textual's Pilot API to validate navigation and actions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gpt_trader.tui.app import TraderApp


class TestKeyboardNavigation:
    @pytest.mark.asyncio
    async def test_quit_shortcut(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)
        app.action_dispatcher.quit_app = AsyncMock()

        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
            app.action_dispatcher.quit_app.assert_called_once()

    @pytest.mark.asyncio
    async def test_help_screen_toggle(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.press("?")
            await pilot.pause()

            assert len(app.screen_stack) > 1, "Help screen should be pushed"

    @pytest.mark.asyncio
    async def test_theme_toggle(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)
        app.action_dispatcher.toggle_theme = AsyncMock()

        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()

            app.action_dispatcher.toggle_theme.assert_called_once()


class TestBotControlInteractions:
    @pytest.mark.asyncio
    async def test_start_bot_updates_ui(self, mock_bot_with_status) -> None:
        bot = mock_bot_with_status
        app = TraderApp(bot=bot)

        async with app.run_test() as pilot:
            assert app.tui_state.running is False

            await pilot.press("s")
            for _ in range(10):
                await pilot.pause()
                if bot.run.call_count:
                    break

            assert bot.run.call_count == 1

            bot.running = True

            await pilot.press("s")
            for _ in range(10):
                await pilot.pause()
                if bot.stop.call_count:
                    break

            assert bot.stop.call_count == 1
