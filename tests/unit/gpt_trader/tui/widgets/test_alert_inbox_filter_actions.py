"""Tests for AlertInbox filter actions."""

from unittest.mock import MagicMock

from textual.widgets import Button, Label, ListView

from gpt_trader.tui.services.alert_manager import AlertCategory, AlertSeverity
from gpt_trader.tui.widgets.tile_states import TileEmptyState


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
