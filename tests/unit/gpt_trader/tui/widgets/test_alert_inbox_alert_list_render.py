"""Tests for AlertInbox alert list rendering."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from textual.widgets import Label, ListView

from gpt_trader.tui.services.alert_manager import Alert, AlertCategory, AlertSeverity
from gpt_trader.tui.widgets.alert_inbox import AlertItem
from gpt_trader.tui.widgets.tile_states import TileEmptyState


class TestAlertInboxAlertListRender:
    """Test suite for AlertInbox alert list rendering."""

    @pytest.fixture
    def sample_alerts(self):
        """Create sample alerts for testing."""
        return [
            Alert(
                rule_id="connection_lost",
                title="Connection Lost",
                message="Lost connection to exchange",
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
            ),
            Alert(
                rule_id="rate_limit_high",
                title="Rate Limit Warning",
                message="Rate limit at 85%",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                timestamp=datetime(2024, 1, 15, 10, 25, 0),
            ),
        ]

    def _create_inbox_mock(self, mock_manager):
        """Create AlertInbox mock with alert manager."""
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

        # Bind the real method
        inbox._refresh_alerts = lambda: AlertInbox._refresh_alerts(inbox)

        return inbox, mock_label, mock_list, mock_empty

    def test_refresh_with_alerts_updates_count_label(self, sample_alerts):
        """Test that refreshing with alerts updates the count label."""
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = sample_alerts

        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock(mock_manager)

        inbox._refresh_alerts()

        mock_label.update.assert_called_with("(2)")

    def test_refresh_with_alerts_shows_list_hides_empty(self, sample_alerts):
        """Test that refreshing with alerts shows list and hides empty state."""
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = sample_alerts

        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock(mock_manager)

        inbox._refresh_alerts()

        assert mock_list.display is True
        assert mock_empty.display is False

    def test_refresh_with_alerts_appends_alert_items(self, sample_alerts):
        """Test that refreshing with alerts appends AlertItem entries."""
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = sample_alerts

        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock(mock_manager)

        inbox._refresh_alerts()

        # Should append one AlertItem per alert
        assert mock_list.append.call_count == 2
        # Check that AlertItem instances are appended
        for call in mock_list.append.call_args_list:
            item = call[0][0]
            assert isinstance(item, AlertItem)

    def test_refresh_with_empty_alerts_shows_empty_state(self):
        """Test that refreshing with empty alert list shows empty state."""
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock(mock_manager)

        inbox._refresh_alerts()

        assert mock_list.display is False
        assert mock_empty.display is True
