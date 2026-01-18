"""Tests for get_recovery_hint in AlertInbox widget."""

from gpt_trader.tui.widgets.alert_inbox import get_recovery_hint


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
