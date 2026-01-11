from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp


@pytest.fixture
def mock_bot():
    """
    Creates a mock TradingBot with all necessary components for TUI testing.
    """
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
    bot.set_ui_adapter = MagicMock(side_effect=lambda adapter: adapter.attach(bot))

    return bot


@pytest.fixture
def mock_app(mock_bot):
    """
    Creates a TraderApp instance with a mock bot.
    """
    return TraderApp(bot=mock_bot)


@pytest.fixture
async def pilot_app(mock_bot):
    """
    Creates a TraderApp with Pilot for interactive testing.

    Use this fixture to test keyboard interactions, widget updates,
    and screen navigation flows.

    Usage:
        async def test_start_stop(pilot_app):
            pilot, app = pilot_app
            await pilot.press("s")  # Press 's' to start bot
            await pilot.pause()     # Wait for UI updates
            assert app.tui_state.running is True

    Yields:
        tuple[Pilot, TraderApp]: The pilot instance and app for testing
    """
    app = TraderApp(bot=mock_bot)
    async with app.run_test() as pilot:
        yield pilot, app
