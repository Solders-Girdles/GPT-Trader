"""Tests for AlertInbox widget core functionality and rendering."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from textual.widgets import Label, ListView

from gpt_trader.tui.services.alert_manager import Alert, AlertCategory, AlertSeverity
from gpt_trader.tui.widgets.alert_inbox import AlertInbox, AlertItem, get_recovery_hint
from gpt_trader.tui.widgets.tile_states import TileEmptyState


def _make_alert(
    *,
    rule_id: str = "test_rule",
    title: str = "Test Alert",
    message: str = "Test message",
    severity: AlertSeverity = AlertSeverity.WARNING,
    category: AlertCategory = AlertCategory.SYSTEM,
    timestamp: datetime | None = None,
) -> Alert:
    return Alert(
        rule_id=rule_id,
        title=title,
        message=message,
        severity=severity,
        category=category,
        timestamp=timestamp or datetime(2024, 1, 15, 10, 30, 0),
    )


def _build_inbox(mock_manager=None):
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
    inbox._alert_manager = mock_manager
    inbox._max_alerts = 10
    inbox.category_filter = set()
    inbox.min_severity = None
    inbox.active_filter = "all"
    inbox.query_one = MagicMock(side_effect=query_side_effect)
    inbox.notify = MagicMock()

    inbox._refresh_alerts = lambda: AlertInbox._refresh_alerts(inbox)
    inbox._show_empty = lambda: AlertInbox._show_empty(inbox)

    return inbox, mock_label, mock_list, mock_empty


@pytest.fixture
def sample_alerts():
    return [
        _make_alert(
            rule_id="connection_lost",
            title="Connection Lost",
            message="Lost connection to exchange",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.SYSTEM,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        ),
        _make_alert(
            rule_id="rate_limit_high",
            title="Rate Limit Warning",
            message="Rate limit at 85%",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            timestamp=datetime(2024, 1, 15, 10, 25, 0),
        ),
    ]


class TestAlertItem:
    """Test suite for AlertItem widget."""

    def test_alert_item_stores_alert(self):
        alert = _make_alert()
        item = AlertItem(alert)
        assert item.alert is alert

    def test_alert_item_truncate_short_text(self):
        alert = _make_alert(title="Test", message="Short", severity=AlertSeverity.INFORMATION)
        item = AlertItem(alert)
        result = item._truncate("Short text", 20)
        assert result == "Short text"

    def test_alert_item_truncate_long_text(self):
        alert = _make_alert(title="Test", message="Long", severity=AlertSeverity.INFORMATION)
        item = AlertItem(alert)
        result = item._truncate("This is a very long text that should be truncated", 20)
        assert result == "This is a very lo..."
        assert len(result) == 20


class TestGetRecoveryHint:
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
    def test_refresh_with_no_manager_calls_show_empty(self):
        inbox, _mock_label, mock_list, mock_empty = _build_inbox(mock_manager=None)
        inbox._refresh_alerts()
        assert mock_empty.display is True
        assert mock_list.display is False
        mock_list.clear.assert_called()

    def test_show_empty_hides_list_shows_empty_state(self):
        inbox, _mock_label, mock_list, mock_empty = _build_inbox(mock_manager=MagicMock())
        inbox._show_empty()
        mock_list.clear.assert_called_once()
        assert mock_list.display is False
        assert mock_empty.display is True


class TestAlertInboxAlertListRender:
    def test_refresh_with_alerts_updates_count_label(self, sample_alerts):
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = sample_alerts

        inbox, mock_label, _mock_list, _mock_empty = _build_inbox(mock_manager)

        inbox._refresh_alerts()

        mock_label.update.assert_called_with("(2)")

    def test_refresh_with_alerts_shows_list_hides_empty(self, sample_alerts):
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = sample_alerts

        inbox, _mock_label, mock_list, mock_empty = _build_inbox(mock_manager)

        inbox._refresh_alerts()

        assert mock_list.display is True
        assert mock_empty.display is False

    def test_refresh_with_alerts_appends_alert_items(self, sample_alerts):
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = sample_alerts

        inbox, _mock_label, mock_list, _mock_empty = _build_inbox(mock_manager)

        inbox._refresh_alerts()

        assert mock_list.append.call_count == 2
        for call in mock_list.append.call_args_list:
            item = call[0][0]
            assert isinstance(item, AlertItem)

    def test_refresh_with_empty_alerts_shows_empty_state(self):
        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        inbox, _mock_label, mock_list, mock_empty = _build_inbox(mock_manager)

        inbox._refresh_alerts()

        assert mock_list.display is False
        assert mock_empty.display is True
