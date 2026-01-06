"""Tests for AlertInbox widget."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Label, ListView

from gpt_trader.tui.services.alert_manager import (
    Alert,
    AlertCategory,
    AlertSeverity,
)
from gpt_trader.tui.widgets.alert_inbox import (
    AlertItem,
    get_recovery_hint,
)
from gpt_trader.tui.widgets.tile_states import TileEmptyState


class TestGetRecoveryHint:
    """Test suite for get_recovery_hint function."""

    def test_returns_rule_specific_hint(self):
        """Test that specific rule_id returns its mapped hint."""
        hint = get_recovery_hint("connection_lost", "system")
        assert hint == "[R] Reconnect"

    def test_returns_rule_specific_hint_rate_limit(self):
        """Test rate_limit_high rule returns expected hint."""
        hint = get_recovery_hint("rate_limit_high", "system")
        assert hint == "Wait or reduce requests"

    def test_returns_rule_specific_hint_reduce_only(self):
        """Test reduce_only_active rule returns expected hint."""
        hint = get_recovery_hint("reduce_only_active", "risk")
        assert hint == "[C] Check config"

    def test_returns_rule_specific_hint_daily_loss(self):
        """Test daily_loss_warning rule returns expected hint."""
        hint = get_recovery_hint("daily_loss_warning", "risk")
        assert hint == "[P] Pause trading"

    def test_returns_rule_specific_hint_bot_stopped(self):
        """Test bot_stopped rule returns expected hint."""
        hint = get_recovery_hint("bot_stopped", "system")
        assert hint == "[S] Start bot"

    def test_falls_back_to_category_hint_system(self):
        """Test unknown rule falls back to category-based hint."""
        hint = get_recovery_hint("unknown_rule", "system")
        assert hint == "[R] Reconnect"

    def test_falls_back_to_category_hint_risk(self):
        """Test unknown rule falls back to risk category hint."""
        hint = get_recovery_hint("unknown_rule", "risk")
        assert hint == "[C] Check config"

    def test_falls_back_to_category_hint_error(self):
        """Test unknown rule falls back to error category hint."""
        hint = get_recovery_hint("unknown_rule", "error")
        assert hint == "Check logs"

    def test_returns_none_for_unknown_category(self):
        """Test returns None when both rule and category are unknown."""
        hint = get_recovery_hint("unknown_rule", "trade")
        assert hint is None


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


class TestAlertInboxFilterActions:
    """Test suite for AlertInbox filter actions."""

    def _create_inbox_mock(self):
        """Create AlertInbox mock with mocked components."""
        from gpt_trader.tui.widgets.alert_inbox import AlertInbox

        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        mock_label = MagicMock(spec=Label)
        mock_list = MagicMock(spec=ListView)
        mock_empty = MagicMock(spec=TileEmptyState)
        mock_buttons = {}
        for btn_id in ["filter-all", "filter-trade", "filter-system", "filter-error"]:
            mock_buttons[btn_id] = MagicMock(spec=Button)

        def query_side_effect(selector, widget_type=None):
            if selector == "#alert-count":
                return mock_label
            if selector == "#alert-list":
                return mock_list
            if selector == "#empty-inbox":
                return mock_empty
            # Handle button queries
            for btn_id, btn in mock_buttons.items():
                if selector == f"#{btn_id}":
                    return btn
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
        inbox._get_alert_category = lambda name: AlertInbox._get_alert_category(inbox, name)
        inbox._get_alert_severity = lambda name: AlertInbox._get_alert_severity(inbox, name)
        inbox._set_active_filter = lambda f: AlertInbox._set_active_filter(inbox, f)
        inbox._refresh_alerts = lambda: AlertInbox._refresh_alerts(inbox)
        inbox.action_filter_trade = lambda: AlertInbox.action_filter_trade(inbox)
        inbox.action_filter_system = lambda: AlertInbox.action_filter_system(inbox)
        inbox.action_filter_error = lambda: AlertInbox.action_filter_error(inbox)
        inbox.action_filter_all = lambda: AlertInbox.action_filter_all(inbox)

        return inbox, mock_manager, mock_buttons

    def test_action_filter_trade_sets_category_filter(self):
        """Test that action_filter_trade sets correct category_filter."""
        inbox, mock_manager, mock_buttons = self._create_inbox_mock()

        inbox.action_filter_trade()

        # Should filter for TRADE and POSITION categories
        assert AlertCategory.TRADE in inbox.category_filter
        assert AlertCategory.POSITION in inbox.category_filter
        assert inbox.min_severity is None
        assert inbox.active_filter == "trade"

    def test_action_filter_system_sets_category_filter(self):
        """Test that action_filter_system sets correct category_filter."""
        inbox, mock_manager, mock_buttons = self._create_inbox_mock()

        inbox.action_filter_system()

        # Should filter for SYSTEM and STRATEGY categories
        assert AlertCategory.SYSTEM in inbox.category_filter
        assert AlertCategory.STRATEGY in inbox.category_filter
        assert inbox.min_severity is None
        assert inbox.active_filter == "system"

    def test_action_filter_error_sets_category_and_severity(self):
        """Test that action_filter_error sets category_filter and min_severity."""
        inbox, mock_manager, mock_buttons = self._create_inbox_mock()

        inbox.action_filter_error()

        # Should filter for ERROR and RISK categories
        assert AlertCategory.ERROR in inbox.category_filter
        assert AlertCategory.RISK in inbox.category_filter
        # Should also set min_severity to WARNING
        assert inbox.min_severity == AlertSeverity.WARNING
        assert inbox.active_filter == "error"

    def test_action_filter_all_resets_filters(self):
        """Test that action_filter_all resets category_filter and min_severity."""
        inbox, mock_manager, mock_buttons = self._create_inbox_mock()

        # First set some filters
        inbox.action_filter_error()
        assert len(inbox.category_filter) > 0
        assert inbox.min_severity is not None

        # Then reset with filter_all
        inbox.action_filter_all()

        assert inbox.category_filter == set()
        assert inbox.min_severity is None
        assert inbox.active_filter == "all"

    def test_set_active_filter_toggles_button_classes(self):
        """Test that _set_active_filter toggles button active classes."""
        inbox, mock_manager, mock_buttons = self._create_inbox_mock()

        inbox._set_active_filter("trade")

        # Trade button should have active class added
        mock_buttons["filter-trade"].add_class.assert_called_with("active")
        # Other buttons should have active class removed
        mock_buttons["filter-all"].remove_class.assert_called_with("active")
        mock_buttons["filter-system"].remove_class.assert_called_with("active")
        mock_buttons["filter-error"].remove_class.assert_called_with("active")


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
