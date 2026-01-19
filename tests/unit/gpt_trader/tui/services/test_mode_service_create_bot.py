"""Tests for create_bot_for_mode helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.tui.services.mode_service import create_bot_for_mode


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
