"""Tests for create_bot_for_mode helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.cli.services as cli_services_module
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

    def test_paper_mode_loads_paper_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test paper mode attempts to load paper config."""
        mock_config = MagicMock()
        mock_load_config = MagicMock(return_value=mock_config)
        mock_bot = MagicMock()
        mock_instantiate = MagicMock(return_value=mock_bot)
        monkeypatch.setattr(cli_services_module, "load_config_from_yaml", mock_load_config)
        monkeypatch.setattr(cli_services_module, "instantiate_bot", mock_instantiate)

        result = create_bot_for_mode("paper")

        mock_load_config.assert_called_once_with("config/profiles/paper.yaml")
        assert result == mock_bot
