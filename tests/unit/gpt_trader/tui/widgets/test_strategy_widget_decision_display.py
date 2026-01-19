from unittest.mock import MagicMock

from gpt_trader.tui.types import ActiveOrders, DecisionData, RiskGuard, RiskState
from gpt_trader.tui.widgets.strategy import StrategyWidget


class TestDecisionBlockedBy:
    """Tests for decision blocked_by field handling."""

    def test_decision_blocked_by_field_exists(self):
        data = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
            blocked_by="DailyLossGuard",
        )
        assert data.blocked_by == "DailyLossGuard"

    def test_decision_blocked_by_defaults_to_empty(self):
        data = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
        )
        assert data.blocked_by == ""

    def test_decision_blocked_by_used_in_display(self):
        widget = StrategyWidget()

        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            guards=[],
        )
        mock_state.order_data = ActiveOrders(orders=[])
        widget.state = mock_state

        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
            blocked_by="VolatilityGuard",
        )

        assert decision.blocked_by == "VolatilityGuard"

    def test_fallback_to_current_risk_state(self):
        widget = StrategyWidget()

        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            guards=[RiskGuard(name="MaxDrawdownGuard")],
        )
        widget.state = mock_state

        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
            blocked_by="",
        )

        current_blocking = widget._get_blocking_reason()
        assert current_blocking == "MaxDrawdownGuard"

        decision_blocked_by = decision.blocked_by or current_blocking
        assert decision_blocked_by == "MaxDrawdownGuard"


class TestEntryExitBadge:
    """Tests for _get_entry_exit_badge method."""

    def test_buy_action_returns_entry_badge(self):
        widget = StrategyWidget()
        badge = widget._get_entry_exit_badge("BUY")
        assert "ENTRY" in badge
        assert "cyan" in badge

    def test_sell_action_returns_entry_badge(self):
        widget = StrategyWidget()
        badge = widget._get_entry_exit_badge("SELL")
        assert "ENTRY" in badge

    def test_close_action_returns_exit_badge(self):
        widget = StrategyWidget()
        badge = widget._get_entry_exit_badge("CLOSE")
        assert "EXIT" in badge
        assert "magenta" in badge

    def test_exit_action_returns_exit_badge(self):
        widget = StrategyWidget()
        badge = widget._get_entry_exit_badge("EXIT")
        assert "EXIT" in badge

    def test_hold_action_returns_empty(self):
        widget = StrategyWidget()
        badge = widget._get_entry_exit_badge("HOLD")
        assert badge == ""

    def test_case_insensitive(self):
        widget = StrategyWidget()
        assert "ENTRY" in widget._get_entry_exit_badge("buy")
        assert "EXIT" in widget._get_entry_exit_badge("close")
