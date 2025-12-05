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

    return bot


class TestTraderApp(TraderApp):
    """Subclass with inline CSS for testing environment."""

    # Override CSS_PATH to avoid file loading issues in tests
    CSS_PATH = None

    # Define essential variables inline
    CSS = """
    $bg-primary: #1A1815;
    $bg-secondary: #2A2520;
    $bg-elevated: #3D3833;
    $accent: #D4744F;
    $accent-hover: #E08A6A;
    $accent-muted: #9B5338;
    $text-primary: #F0EDE9;
    $text-secondary: #B8B5B2;
    $text-muted: #7A7672;
    $success: #85B77F;
    $warning: #E0B366;
    $error: #E08580;
    $info: #7FB8D4;
    $border-subtle: #3D3833;
    $border-emphasis: #544F49;
    $spacing-xs: 1;
    $spacing-sm: 2;
    $spacing-md: 3;
    $spacing-lg: 4;

    /* Essential Layout */
    Screen {
        layout: vertical;
        background: $bg-primary;
        color: $text-primary;
    }
    """


@pytest.fixture
def mock_app(mock_bot):
    """
    Creates a TraderApp instance with a mock bot.
    Uses a subclass with inline CSS to guarantee variable availability.
    """
    return TestTraderApp(bot=mock_bot)
