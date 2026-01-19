"""Tests for AlertInbox clear action and key bindings."""

from unittest.mock import MagicMock

from textual.widgets import Label, ListView

from gpt_trader.tui.widgets.tile_states import TileEmptyState


class TestAlertInboxClearAction:
    """Test suite for AlertInbox clear action."""

    def _create_inbox_mock(self, mock_manager=None):
        """Create AlertInbox mock for testing clear action."""
        from gpt_trader.tui.widgets.alert_inbox import AlertInbox

        mock_label = MagicMock(spec=Label)
        mock_list = MagicMock(spec=ListView)
        mock_empty = MagicMock(spec=TileEmptyState)

        def query_side_effect(selector, widget_type=None):
            if selector == "#alert-count":
                return mock_label
            if selector == "#alert-list":
                return mock_list
            if selector == "#empty-inbox":
                return mock_empty
            return MagicMock()

        inbox = MagicMock()
        inbox._alert_manager = mock_manager
        inbox._max_alerts = 10
        inbox.category_filter = set()
        inbox.min_severity = None
        inbox.active_filter = "all"
        inbox.query_one = MagicMock(side_effect=query_side_effect)
        inbox.notify = MagicMock()

        # Bind real methods
        inbox._refresh_alerts = lambda: AlertInbox._refresh_alerts(inbox)
        inbox.action_clear_alerts = lambda: AlertInbox.action_clear_alerts(inbox)

        return inbox

    def test_action_clear_alerts_calls_clear_history(self):
        """Test that action_clear_alerts calls alert_manager.clear_history()."""
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        inbox = self._create_inbox_mock(mock_manager)

        inbox.action_clear_alerts()

        mock_manager.clear_history.assert_called_once()

    def test_action_clear_alerts_calls_notify(self):
        """Test that action_clear_alerts calls notify."""
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        inbox = self._create_inbox_mock(mock_manager)

        inbox.action_clear_alerts()

        inbox.notify.assert_called_once_with("Alerts cleared", timeout=2)

    def test_action_clear_alerts_with_no_manager_does_not_crash(self):
        """Test that action_clear_alerts handles missing manager gracefully."""
        inbox = self._create_inbox_mock(None)

        # Should not raise
        inbox.action_clear_alerts()

        inbox.notify.assert_not_called()


class TestAlertInboxBindings:
    """Test suite for AlertInbox key bindings."""

    def test_has_expected_bindings(self):
        """Test that AlertInbox class has expected key bindings."""
        from gpt_trader.tui.widgets.alert_inbox import AlertInbox

        # Access BINDINGS directly from the class, not an instance
        binding_keys = [b.key for b in AlertInbox.BINDINGS]

        assert "1" in binding_keys  # filter_all
        assert "2" in binding_keys  # filter_trade
        assert "3" in binding_keys  # filter_system
        assert "4" in binding_keys  # filter_error
        assert "c" in binding_keys  # clear_alerts
