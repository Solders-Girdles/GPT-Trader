"""Tests for ModeService core behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.services.mode_service import ModeService


class TestModeService:
    """Test ModeService functionality."""

    def test_init_sets_app_and_scenario(self):
        """Test initialization sets app and scenario."""
        mock_app = MagicMock()
        service = ModeService(mock_app, demo_scenario="bullish")

        assert service.app == mock_app
        assert service.demo_scenario == "bullish"

    def test_detect_bot_mode_demo(self):
        """Test detect_bot_mode identifies DemoBot."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        mock_bot = MagicMock()
        mock_bot.__class__.__name__ = "DemoBot"

        result = service.detect_bot_mode(mock_bot)

        assert result == "demo"

    def test_detect_bot_mode_read_only(self):
        """Test detect_bot_mode identifies read_only mode."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        mock_bot = MagicMock()
        mock_bot.__class__.__name__ = "TradingBot"
        mock_bot.config.read_only = True

        result = service.detect_bot_mode(mock_bot)

        assert result == "read_only"

    def test_detect_bot_mode_live(self):
        """Test detect_bot_mode identifies live mode."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        mock_bot = MagicMock()
        mock_bot.__class__.__name__ = "TradingBot"
        mock_bot.config.read_only = False
        mock_bot.config.profile = "PROD"

        result = service.detect_bot_mode(mock_bot)

        assert result == "live"

    def test_detect_bot_mode_paper_default(self):
        """Test detect_bot_mode defaults to paper for safety."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        mock_bot = MagicMock()
        mock_bot.__class__.__name__ = "TradingBot"
        # No config attributes set
        del mock_bot.config

        result = service.detect_bot_mode(mock_bot)

        assert result == "paper"

    def test_create_bot_delegates_to_function(self):
        """Test create_bot uses the module function."""
        mock_app = MagicMock()
        service = ModeService(mock_app, demo_scenario="volatile")

        bot = service.create_bot("demo")

        assert bot.__class__.__name__ == "DemoBot"

    @pytest.mark.asyncio
    async def test_show_live_warning_pushes_modal(self):
        """Test show_live_warning pushes LiveWarningModal."""
        mock_app = MagicMock()

        async def mock_push_screen_wait(modal):
            return True

        mock_app.push_screen_wait = mock_push_screen_wait

        service = ModeService(mock_app)
        result = await service.show_live_warning()

        assert result is True

    def test_notify_mode_changed_posts_event(self):
        """Test notify_mode_changed posts BotModeChanged event."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        service.notify_mode_changed("live", "paper")

        mock_app.post_message.assert_called_once()
        event = mock_app.post_message.call_args[0][0]
        assert event.new_mode == "live"
        assert event.old_mode == "paper"
