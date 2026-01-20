"""Tests for ConfigService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.tui.widgets.config as config_widget_module
from gpt_trader.tui.services.config_service import ConfigService


class TestConfigService:
    """Test ConfigService functionality."""

    def test_init_sets_app(self):
        """Test initialization sets app reference."""
        mock_app = MagicMock()
        service = ConfigService(mock_app)

        assert service.app == mock_app

    def test_show_config_modal_pushes_screen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test show_config_modal pushes ConfigModal screen."""
        mock_app = MagicMock()
        mock_config = MagicMock()
        mock_modal = MagicMock()
        mock_modal_class = MagicMock(return_value=mock_modal)
        monkeypatch.setattr(config_widget_module, "ConfigModal", mock_modal_class)

        service = ConfigService(mock_app)
        service.show_config_modal(mock_config)

        mock_modal_class.assert_called_once_with(mock_config)
        mock_app.push_screen.assert_called_once_with(mock_modal)

    def test_show_config_modal_handles_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test show_config_modal handles errors gracefully."""
        mock_app = MagicMock()
        mock_app.push_screen.side_effect = Exception("Test error")
        mock_modal_class = MagicMock()
        monkeypatch.setattr(config_widget_module, "ConfigModal", mock_modal_class)

        service = ConfigService(mock_app)
        # Should not raise
        service.show_config_modal(MagicMock())

        mock_app.notify.assert_called_once()
        assert "error" in mock_app.notify.call_args[1]["severity"].lower()

    def test_request_reload_posts_event(self):
        """Test request_reload posts ConfigReloadRequested event."""
        mock_app = MagicMock()
        service = ConfigService(mock_app)

        service.request_reload()

        mock_app.post_message.assert_called_once()

    def test_notify_config_changed_posts_event(self):
        """Test notify_config_changed posts ConfigChanged event."""
        mock_app = MagicMock()
        mock_config = MagicMock()
        service = ConfigService(mock_app)

        service.notify_config_changed(mock_config)

        mock_app.post_message.assert_called_once()
        event = mock_app.post_message.call_args[0][0]
        assert event.config == mock_config
