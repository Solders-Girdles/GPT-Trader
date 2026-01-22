"""Tests for ModeService core behavior and create_bot_for_mode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.cli.services as cli_services_module
from gpt_trader.tui.services.mode_service import ModeService, create_bot_for_mode


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


class TestCreateBotForMode:
    """Test create_bot_for_mode function."""

    def test_demo_mode_creates_demo_bot(self):
        bot = create_bot_for_mode("demo")

        assert bot.__class__.__name__ == "DemoBot"

    def test_unknown_mode_raises_error(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            create_bot_for_mode("invalid_mode")

    def test_paper_mode_loads_paper_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_config = MagicMock()
        mock_load_config = MagicMock(return_value=mock_config)
        mock_bot = MagicMock()
        mock_instantiate = MagicMock(return_value=mock_bot)
        monkeypatch.setattr(cli_services_module, "load_config_from_yaml", mock_load_config)
        monkeypatch.setattr(cli_services_module, "instantiate_bot", mock_instantiate)

        result = create_bot_for_mode("paper")

        mock_load_config.assert_called_once_with("config/profiles/paper.yaml")
        assert result == mock_bot
