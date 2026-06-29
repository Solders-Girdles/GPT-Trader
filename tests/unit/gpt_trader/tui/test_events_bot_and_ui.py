"""Tests for core bot/UI TUI events."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.message import Message

from gpt_trader.tui.events import (
    BotModeChanged,
    BotStartRequested,
    BotStateChanged,
    ConfigChanged,
    ConfigReloadRequested,
    ResponsiveStateChanged,
    ThemeChanged,
)
from gpt_trader.tui.responsive_state import ResponsiveState


class TestBotLifecycleEvents:
    """Test bot lifecycle events."""

    def test_bot_start_requested_creation(self):
        """Test BotStartRequested event creation."""
        event = BotStartRequested()
        assert isinstance(event, Message)

    def test_bot_state_changed_creation(self):
        """Test BotStateChanged event creation with attributes."""
        event = BotStateChanged(running=True, uptime=123.45)
        assert isinstance(event, Message)
        assert event.running is True
        assert event.uptime == 123.45

    def test_bot_state_changed_default_uptime(self):
        """Test BotStateChanged with default uptime."""
        event = BotStateChanged(running=False)
        assert event.running is False
        assert event.uptime == 0.0

    def test_bot_mode_changed_creation(self):
        """Test BotModeChanged event creation."""
        event = BotModeChanged(new_mode="live", old_mode="demo")
        assert isinstance(event, Message)
        assert event.new_mode == "live"
        assert event.old_mode == "demo"


class TestUICoordinationEvents:
    """Test UI coordination events."""

    def test_responsive_state_changed_creation(self):
        """Test ResponsiveStateChanged event creation."""
        event = ResponsiveStateChanged(state=ResponsiveState.COMFORTABLE, width=140)
        assert isinstance(event, Message)
        assert event.state == ResponsiveState.COMFORTABLE
        assert event.width == 140


class TestConfigurationEvents:
    """Test configuration events."""

    def test_config_reload_requested_creation(self):
        """Test ConfigReloadRequested event creation."""
        event = ConfigReloadRequested()
        assert isinstance(event, Message)

    def test_config_changed_creation(self):
        """Test ConfigChanged event creation."""
        mock_config = MagicMock()
        event = ConfigChanged(config=mock_config)
        assert isinstance(event, Message)
        assert event.config == mock_config


class TestThemeEvents:
    """Test theme events."""

    def test_theme_changed_creation(self):
        """Test ThemeChanged event creation."""
        event = ThemeChanged(theme_mode="light")
        assert isinstance(event, Message)
        assert event.theme_mode == "light"
