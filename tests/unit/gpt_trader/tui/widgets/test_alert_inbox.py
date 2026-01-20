"""Tests for AlertInbox widget core functionality."""

from datetime import datetime
from unittest.mock import MagicMock

from textual.widgets import Label, ListView

from gpt_trader.tui.services.alert_manager import Alert, AlertCategory, AlertSeverity
from gpt_trader.tui.widgets.alert_inbox import AlertItem, get_recovery_hint
from gpt_trader.tui.widgets.tile_states import TileEmptyState


class TestAlertItem:
    """Test suite for AlertItem widget."""

    def test_alert_item_stores_alert(self):
        """Test that AlertItem stores the alert reference."""
        alert = Alert(
            rule_id="test_rule",
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            timestamp=datetime.now(),
        )
        item = AlertItem(alert)
        assert item.alert is alert

    def test_alert_item_truncate_short_text(self):
        """Test _truncate with text shorter than max_length."""
        alert = Alert(
            rule_id="test_rule",
            title="Test",
            message="Short",
            severity=AlertSeverity.INFORMATION,
            category=AlertCategory.TRADE,
            timestamp=datetime.now(),
        )
        item = AlertItem(alert)
        result = item._truncate("Short text", 20)
        assert result == "Short text"

    def test_alert_item_truncate_long_text(self):
        """Test _truncate with text longer than max_length."""
        alert = Alert(
            rule_id="test_rule",
            title="Test",
            message="Long",
            severity=AlertSeverity.INFORMATION,
            category=AlertCategory.TRADE,
            timestamp=datetime.now(),
        )
        item = AlertItem(alert)
        result = item._truncate("This is a very long text that should be truncated", 20)
        assert result == "This is a very lo..."
        assert len(result) == 20


class TestGetRecoveryHint:
    """Test suite for get_recovery_hint function."""

    def test_returns_rule_specific_hint(self):
        assert get_recovery_hint("connection_lost", "system") == "[R] Reconnect"

    def test_returns_rule_specific_hint_rate_limit(self):
        assert get_recovery_hint("rate_limit_high", "system") == "Wait or reduce requests"

    def test_returns_rule_specific_hint_reduce_only(self):
        assert get_recovery_hint("reduce_only_active", "risk") == "[C] Check config"

    def test_returns_rule_specific_hint_daily_loss(self):
        assert get_recovery_hint("daily_loss_warning", "risk") == "[P] Pause trading"

    def test_returns_rule_specific_hint_bot_stopped(self):
        assert get_recovery_hint("bot_stopped", "system") == "[S] Start bot"

    def test_falls_back_to_category_hint_system(self):
        assert get_recovery_hint("unknown_rule", "system") == "[R] Reconnect"

    def test_falls_back_to_category_hint_risk(self):
        assert get_recovery_hint("unknown_rule", "risk") == "[C] Check config"

    def test_falls_back_to_category_hint_error(self):
        assert get_recovery_hint("unknown_rule", "error") == "Check logs"

    def test_returns_none_for_unknown_category(self):
        assert get_recovery_hint("unknown_rule", "trade") is None


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

        inbox = MagicMock(spec=AlertInbox)
        inbox._alert_manager = alert_manager
        inbox._max_alerts = 10
        inbox.category_filter = set()
        inbox.min_severity = None
        inbox.active_filter = "all"
        inbox.query_one = MagicMock(side_effect=query_side_effect)
        inbox.notify = MagicMock()

        inbox._refresh_alerts = lambda: AlertInbox._refresh_alerts(inbox)
        inbox._show_empty = lambda: AlertInbox._show_empty(inbox)

        return inbox, mock_label, mock_list, mock_empty

    def test_refresh_with_no_manager_calls_show_empty(self):
        """Test that _refresh_alerts with no manager shows empty state."""
        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock(alert_manager=None)
        inbox._refresh_alerts()
        assert mock_empty.display is True
        assert mock_list.display is False
        mock_list.clear.assert_called()

    def test_show_empty_hides_list_shows_empty_state(self):
        """Test that _show_empty hides list and shows empty state."""
        inbox, mock_label, mock_list, mock_empty = self._create_inbox_mock()
        inbox._show_empty()
        mock_list.clear.assert_called_once()
        assert mock_list.display is False
        assert mock_empty.display is True
