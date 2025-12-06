"""Tests for AlertHistoryScreen."""

from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from gpt_trader.tui.screens.alert_history import AlertHistoryScreen
from gpt_trader.tui.services.alert_manager import Alert, AlertSeverity


class TestAlertHistoryScreen:
    """Test suite for AlertHistoryScreen."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock TraderApp."""
        app = MagicMock()
        app.notify = MagicMock()
        app.pop_screen = MagicMock()

        # Create mock AlertManager with history
        app.alert_manager = MagicMock()
        app.alert_manager.get_history.return_value = [
            Alert(
                rule_id="connection_lost",
                title="Connection Lost",
                message="Lost connection to exchange",
                severity=AlertSeverity.ERROR,
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
            ),
            Alert(
                rule_id="rate_limit_high",
                title="Rate Limit Warning",
                message="Rate limit at 85%",
                severity=AlertSeverity.WARNING,
                timestamp=datetime(2024, 1, 15, 10, 25, 0),
            ),
        ]
        return app

    def test_screen_has_bindings(self):
        """Test that screen has expected key bindings."""
        screen = AlertHistoryScreen()
        binding_keys = [b[0] for b in screen.BINDINGS]
        assert "escape" in binding_keys
        assert "c" in binding_keys
        assert "r" in binding_keys

    def test_action_clear_history_clears_alerts(self, mock_app):
        """Test that clear history action calls alert manager."""
        screen = AlertHistoryScreen()

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            # Mock _refresh_table since DataTable isn't mounted
            with patch.object(screen, "_refresh_table"):
                screen.action_clear_history()

        mock_app.alert_manager.clear_history.assert_called_once()
        mock_app.notify.assert_called()

    def test_action_reset_cooldowns_resets_alerts(self, mock_app):
        """Test that reset cooldowns action calls alert manager."""
        screen = AlertHistoryScreen()

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            screen.action_reset_cooldowns()

        mock_app.alert_manager.reset_cooldowns.assert_called_once()
        mock_app.notify.assert_called()

    def test_button_pressed_clear_btn(self, mock_app):
        """Test clear button calls action_clear_history."""
        screen = AlertHistoryScreen()

        mock_button = MagicMock()
        mock_button.id = "clear-btn"
        event = MagicMock()
        event.button = mock_button

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(screen, "action_clear_history") as mock_action:
                screen.on_button_pressed(event)
                mock_action.assert_called_once()

    def test_button_pressed_reset_btn(self, mock_app):
        """Test reset button calls action_reset_cooldowns."""
        screen = AlertHistoryScreen()

        mock_button = MagicMock()
        mock_button.id = "reset-btn"
        event = MagicMock()
        event.button = mock_button

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(screen, "action_reset_cooldowns") as mock_action:
                screen.on_button_pressed(event)
                mock_action.assert_called_once()

    def test_button_pressed_close_btn(self, mock_app):
        """Test close button pops screen."""
        screen = AlertHistoryScreen()

        mock_button = MagicMock()
        mock_button.id = "close-btn"
        event = MagicMock()
        event.button = mock_button

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            screen.on_button_pressed(event)
            mock_app.pop_screen.assert_called_once()

    def test_no_alert_manager_graceful_handling(self):
        """Test screen handles missing alert_manager gracefully."""
        screen = AlertHistoryScreen()
        mock_app = MagicMock(spec=[])  # Empty spec means no alert_manager attribute

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            # Should not raise
            screen.action_clear_history()
            screen.action_reset_cooldowns()
