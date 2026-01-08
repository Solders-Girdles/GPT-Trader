"""Tests for ModeService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.config import path_registry
from gpt_trader.tui.services.mode_service import ModeService, create_bot_for_mode


class TestCreateBotForMode:
    """Test create_bot_for_mode function."""

    def test_demo_mode_creates_demo_bot(self):
        """Test demo mode creates a DemoBot instance."""
        bot = create_bot_for_mode("demo")

        assert bot.__class__.__name__ == "DemoBot"

    def test_unknown_mode_raises_error(self):
        """Test unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            create_bot_for_mode("invalid_mode")

    @patch("gpt_trader.cli.services.load_config_from_yaml")
    @patch("gpt_trader.cli.services.instantiate_bot")
    def test_paper_mode_loads_paper_config(self, mock_instantiate, mock_load_config):
        """Test paper mode attempts to load paper config."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        mock_bot = MagicMock()
        mock_instantiate.return_value = mock_bot

        result = create_bot_for_mode("paper")

        mock_load_config.assert_called_once_with("config/profiles/paper.yaml")
        assert result == mock_bot


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
        mock_app.push_screen_wait = MagicMock(return_value=True)

        # Make it an async mock
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


class TestModePreferencePersistence:
    """Test mode preference persistence functionality."""

    def test_load_mode_preference_returns_saved_mode(self, tmp_path: Path):
        """Test loading a previously saved mode."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"mode": "paper", "theme": "dark"}))

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result == "paper"

    def test_load_mode_preference_returns_none_when_missing(self, tmp_path: Path):
        """Test returns None when no mode saved."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"theme": "dark"}))  # No mode key

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result is None

    def test_load_mode_preference_returns_none_for_nonexistent_file(self, tmp_path: Path):
        """Test returns None when preferences file doesn't exist."""
        prefs_file = tmp_path / "nonexistent.json"

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result is None

    def test_load_mode_preference_ignores_invalid_mode(self, tmp_path: Path):
        """Test ignores saved mode if it's not a valid mode."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"mode": "invalid_mode"}))

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result is None

    def test_save_mode_preference_creates_file(self, tmp_path: Path):
        """Test saving mode creates preferences file."""
        prefs_file = tmp_path / "config" / "tui_preferences.json"
        # Parent directory doesn't exist yet

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.save_mode_preference("demo")

        assert result is True
        assert prefs_file.exists()
        saved = json.loads(prefs_file.read_text())
        assert saved["mode"] == "demo"

    def test_save_mode_preference_preserves_other_keys(self, tmp_path: Path):
        """Test saving mode preserves other preferences."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"theme": "light", "other": "value"}))

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.save_mode_preference("live")

        assert result is True
        saved = json.loads(prefs_file.read_text())
        assert saved["mode"] == "live"
        assert saved["theme"] == "light"
        assert saved["other"] == "value"

    def test_save_mode_preference_rejects_invalid_mode(self, tmp_path: Path):
        """Test saving invalid mode returns False."""
        prefs_file = tmp_path / "tui_preferences.json"

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.save_mode_preference("invalid_mode")

        assert result is False
        assert not prefs_file.exists()

    def test_save_mode_preference_accepts_all_valid_modes(self, tmp_path: Path):
        """Test all valid modes can be saved."""
        valid_modes = ["demo", "paper", "read_only", "live"]

        for mode in valid_modes:
            prefs_file = tmp_path / f"prefs_{mode}.json"

            mock_app = MagicMock()
            service = ModeService(mock_app, preferences_path=prefs_file)

            result = service.save_mode_preference(mode)

            assert result is True, f"Failed to save mode: {mode}"
            saved = json.loads(prefs_file.read_text())
            assert saved["mode"] == mode

    @pytest.mark.uses_real_preferences
    def test_init_sets_default_preferences_path(self):
        """Test init uses default preferences path when not specified."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        assert service.preferences_path == (path_registry.RUNTIME_DATA_DIR / "tui_preferences.json")
