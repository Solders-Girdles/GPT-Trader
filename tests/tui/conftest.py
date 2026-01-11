from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.pilot import Pilot

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp

# Small pause to let Textual process events in tests without slowing runs too much.
_DEFAULT_PILOT_PAUSE_SECONDS = 0.01


@pytest.fixture(autouse=True)
def stabilize_pilot_pause(monkeypatch):
    """Ensure pilot.pause always yields at least a short delay."""
    original_pause = Pilot.pause

    async def _pause(self, delay: float | None = None) -> None:
        await original_pause(self, _DEFAULT_PILOT_PAUSE_SECONDS if delay is None else delay)

    monkeypatch.setattr(Pilot, "pause", _pause)


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


class TestTraderApp(TraderApp):
    """Subclass with inline CSS for testing environment."""

    # Override CSS_PATH to avoid file loading issues in tests
    CSS_PATH = None

    # Define essential variables inline
    CSS = """
    /* Nord Theme Colors */
    $nord0: #2e3440;
    $nord1: #3b4252;
    $nord2: #434c5e;
    $nord3: #4c566a;
    $nord4: #d8dee9;
    $nord5: #e5e9f0;
    $nord6: #eceff4;
    $nord7: #8fbcbb;
    $nord8: #88c0d0;
    $nord9: #81a1c1;
    $nord10: #5e81ac;
    $nord11: #bf616a;
    $nord12: #d08770;
    $nord13: #ebcb8b;
    $nord14: #a3be8c;
    $nord15: #b48ead;

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
        background: $nord0;
        color: $nord4;
    }
    """


@pytest.fixture
def mock_app(mock_bot):
    """
    Creates a TraderApp instance with a mock bot.
    Uses a subclass with inline CSS to guarantee variable availability.
    """
    return TestTraderApp(bot=mock_bot)
