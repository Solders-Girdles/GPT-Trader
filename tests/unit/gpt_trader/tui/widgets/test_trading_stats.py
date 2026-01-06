"""Tests for TradingStatsWidget."""

from unittest.mock import MagicMock, PropertyMock, patch

from gpt_trader.tui.services.focus_manager import FocusManager
from gpt_trader.tui.widgets.trading_stats import TradingStatsWidget


class TestTradingStatsWidgetBindings:
    """Tests for TradingStatsWidget keybindings."""

    def test_window_bindings_present(self):
        """Test 'w' and 'W' keybinds for window toggle are present."""
        widget = TradingStatsWidget()
        binding_keys = [b.key for b in widget.BINDINGS]

        assert "w" in binding_keys  # Cycle window
        assert "W" in binding_keys  # Reset to all session

    def test_window_binding_actions_exist(self):
        """Test window action methods exist and are callable."""
        widget = TradingStatsWidget()

        assert hasattr(widget, "action_cycle_window")
        assert callable(widget.action_cycle_window)
        assert hasattr(widget, "action_reset_window")
        assert callable(widget.action_reset_window)


class TestTradingStatsWidgetModes:
    """Tests for TradingStatsWidget view modes."""

    def test_compact_mode_initialization(self):
        """Test widget can be initialized in compact mode."""
        widget = TradingStatsWidget(compact=True)
        assert widget._compact is True

    def test_default_mode_is_expanded(self):
        """Test widget defaults to expanded mode."""
        widget = TradingStatsWidget()
        assert widget._compact is False


class TestTradingStatsWidgetActions:
    """Tests for TradingStatsWidget action methods."""

    def test_action_cycle_window_calls_service(self):
        """Test action_cycle_window uses the service."""
        widget = TradingStatsWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()
        mock_state_registry = MagicMock()
        mock_state_registry.get_state.return_value = None
        mock_app.state_registry = mock_state_registry

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(
                widget._service, "cycle_window", return_value=(5, "Last 5 min")
            ) as mock_cycle:
                with patch.object(widget, "_update_window_chips"):
                    widget.action_cycle_window()
                    mock_cycle.assert_called_once()

    def test_action_reset_window_calls_service(self):
        """Test action_reset_window uses the service."""
        widget = TradingStatsWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()
        mock_state_registry = MagicMock()
        mock_state_registry.get_state.return_value = None
        mock_app.state_registry = mock_state_registry

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(
                widget._service, "reset_window", return_value=(0, "All Session")
            ) as mock_reset:
                with patch.object(widget, "_update_window_chips"):
                    widget.action_reset_window()
                    mock_reset.assert_called_once()


class TestTradingStatsWidgetStateUpdates:
    """Tests for TradingStatsWidget state handling."""

    def test_has_received_update_initially_false(self):
        """Test _has_received_update starts as False."""
        widget = TradingStatsWidget()
        assert widget._has_received_update is False

    def test_on_state_updated_sets_flag(self):
        """Test on_state_updated sets _has_received_update to True."""
        widget = TradingStatsWidget()
        mock_state = MagicMock()
        mock_state.trade_data = None

        widget.on_state_updated(mock_state)

        assert widget._has_received_update is True


class TestTradingStatsDashboardIntegration:
    """Tests for TradingStats integration in main dashboard."""

    def test_account_tile_actions_include_window_keybind(self):
        """Test tile-account shows w/Window action hint."""
        actions = FocusManager.TILE_ACTIONS.get("tile-account", [])
        action_keys = [key for key, _ in actions]

        assert "w" in action_keys, "Window keybind 'w' should be in account tile actions"

    def test_account_tile_has_window_description(self):
        """Test tile-account has Window description for w keybind."""
        actions = FocusManager.TILE_ACTIONS.get("tile-account", [])
        action_dict = dict(actions)

        assert action_dict.get("w") == "Window", "w keybind should have 'Window' description"

    def test_main_screen_has_stats_widget_bindings(self):
        """Test MainScreen has w/W bindings for stats window cycling."""
        from gpt_trader.tui.screens.main_screen import MainScreen

        binding_keys = [b.key for b in MainScreen.BINDINGS]

        assert "w" in binding_keys, "MainScreen should have 'w' binding"
        assert "W" in binding_keys, "MainScreen should have 'W' binding"

    def test_main_screen_has_stats_action_methods(self):
        """Test MainScreen has action methods for stats window cycling."""
        from gpt_trader.tui.screens.main_screen import MainScreen

        assert hasattr(MainScreen, "action_cycle_stats_window")
        assert hasattr(MainScreen, "action_reset_stats_window")
