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
        assert "x" in binding_keys  # Clear history
        assert "y" in binding_keys  # Copy row
        assert "r" in binding_keys  # Reset cooldowns

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


class TestAlertHistoryFilterActions:
    """Tests for filter cycling actions (aligned with Logs 'f'/'F' pattern)."""

    @pytest.fixture
    def screen_with_mocks(self):
        """Create AlertHistoryScreen with mocked _set_filter and notify."""
        screen = AlertHistoryScreen()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()
        return screen, mock_app

    def test_cycle_filter_cycles_through_all_filters(self, screen_with_mocks):
        """Test 'f' key cycles through all filter options."""
        screen, mock_app = screen_with_mocks

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(screen, "_set_filter") as mock_set_filter:
                # Start at 'all', should cycle to 'trade'
                screen._current_filter = "all"
                screen.action_cycle_filter()
                mock_set_filter.assert_called_with("trade")

    def test_cycle_filter_wraps_around(self, screen_with_mocks):
        """Test filter cycling wraps from last to first."""
        screen, mock_app = screen_with_mocks

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(screen, "_set_filter") as mock_set_filter:
                # At 'error' (last), should wrap to 'all'
                screen._current_filter = "error"
                screen.action_cycle_filter()
                mock_set_filter.assert_called_with("all")

    def test_clear_filter_resets_to_all(self, screen_with_mocks):
        """Test 'F' key resets filter to 'all'."""
        screen, mock_app = screen_with_mocks

        with patch.object(type(screen), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(screen, "_set_filter") as mock_set_filter:
                screen._current_filter = "risk"
                screen.action_clear_filter()
                mock_set_filter.assert_called_with("all")

    def test_filter_bindings_present(self):
        """Test 'f' and 'F' keybinds are present."""
        screen = AlertHistoryScreen()
        binding_keys = [b[0] for b in screen.BINDINGS]
        assert "f" in binding_keys  # Cycle filter
        assert "F" in binding_keys  # Clear filter

    def test_number_filter_bindings_present(self):
        """Test 1-5 keybinds for direct filter selection are present."""
        screen = AlertHistoryScreen()
        binding_keys = [b[0] for b in screen.BINDINGS]
        for key in ["1", "2", "3", "4", "5"]:
            assert key in binding_keys
