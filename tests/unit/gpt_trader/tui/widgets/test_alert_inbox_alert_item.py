"""Tests for AlertItem widget used by AlertInbox."""

from datetime import datetime

from gpt_trader.tui.services.alert_manager import Alert, AlertCategory, AlertSeverity
from gpt_trader.tui.widgets.alert_inbox import AlertItem


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
