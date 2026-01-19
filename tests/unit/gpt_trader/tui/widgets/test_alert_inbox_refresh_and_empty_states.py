"""Tests for AlertInbox refresh and empty state behavior."""

from unittest.mock import MagicMock

from textual.widgets import Label, ListView

from gpt_trader.tui.widgets.tile_states import TileEmptyState


class TestAlertInboxRefreshAndEmptyStates:
    """Test suite for AlertInbox refresh and empty state behavior."""

    def _create_inbox_mock(self, alert_manager=None):
        """Create a mock that simulates AlertInbox behavior."""
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

        # Create a mock that has the methods we want to test
        inbox = MagicMock(spec=AlertInbox)
        inbox._alert_manager = alert_manager
        inbox._max_alerts = 10
        inbox.category_filter = set()
        inbox.min_severity = None
        inbox.active_filter = "all"
        inbox.query_one = MagicMock(side_effect=query_side_effect)
        inbox.notify = MagicMock()

        # Bind the real methods to the mock
        inbox._refresh_alerts = lambda: AlertInbox._refresh_alerts(inbox)
        inbox._show_empty = lambda: AlertInbox._show_empty(inbox)

        return inbox, mock_label, mock_list, mock_empty

    def test_refresh_with_no_manager_calls_show_empty(self):
        """Test that _refresh_alerts with no manager shows empty state."""
        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock(alert_manager=None)

        inbox._refresh_alerts()

        # Empty state should be visible
        assert mock_empty.display is True
        # List should be hidden
        assert mock_list.display is False
        # List should be cleared
        mock_list.clear.assert_called()

    def test_show_empty_hides_list_shows_empty_state(self):
        """Test that _show_empty hides list and shows empty state."""
        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock()

        inbox._show_empty()

        mock_list.clear.assert_called_once()
        assert mock_list.display is False
        assert mock_empty.display is True
