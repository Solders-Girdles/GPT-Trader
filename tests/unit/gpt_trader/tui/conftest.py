from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter


@pytest.fixture
def mock_bot_with_status() -> MagicMock:
    """Creates a mock bot with a StatusReporter that has initial status."""
    bot = MagicMock()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    return bot


@pytest.fixture
def mock_demo_bot() -> MagicMock:
    """Mock bot that looks like a DemoBot for mode detection."""

    class DemoBot(MagicMock):
        pass

    bot = DemoBot()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    return bot
