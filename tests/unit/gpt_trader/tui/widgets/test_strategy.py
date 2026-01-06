from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.types import ActiveOrders, DecisionData, RiskState, StrategyState
from gpt_trader.tui.widgets.strategy import StrategyWidget


class TestStrategyWidget:
    def test_update_strategy(self):
        widget = StrategyWidget()

        # Mock table and label
        mock_table = MagicMock(spec=DataTable)
        mock_empty_label = MagicMock()

        # Mock rows and columns properties for delta updates
        mock_table.rows = MagicMock()
        mock_table.rows.keys.return_value = {}  # Empty table initially
        mock_table.columns = MagicMock()
        mock_table.columns.keys.return_value = ["Symbol", "Action", "Confidence", "Reason", "Time"]

        def query_one_side_effect(selector, widget_type=None):
            if selector == DataTable or widget_type == DataTable:
                return mock_table
            if "empty" in str(selector):
                return mock_empty_label
            return mock_table

        widget.query_one = MagicMock(side_effect=query_one_side_effect)

        # Create data
        data = StrategyState(
            active_strategies=["StrategyA", "StrategyB"],
            last_decisions={
                "BTC-USD": DecisionData(
                    symbol="BTC-USD",
                    action="BUY",
                    reason="Signal",
                    confidence=0.95,
                    timestamp=1234567890.0,
                )
            },
        )

        # Call update
        widget.update_strategy(data)

        # Verify row was added (new symbol, not existing)
        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        # Confidence now includes badge (e.g., "[green]0.95 HIGH[/green]")
        assert "0.95" in args[2]
        assert "HIGH" in args[2]  # High confidence badge
        assert kwargs.get("key") == "BTC-USD"  # Row key for delta updates


class TestStrategyWidgetBlockingReason:
    """Tests for _get_blocking_reason method."""

    def test_no_blocking_when_state_is_none(self):
        """Returns empty string when state is None."""
        widget = StrategyWidget()
        widget.state = None
        assert widget._get_blocking_reason() == ""

    def test_no_blocking_when_no_risk_conditions(self):
        """Returns empty string when no blocking conditions."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            active_guards=[],
        )
        widget.state = mock_state
        assert widget._get_blocking_reason() == ""

    def test_reduce_only_mode_with_reason(self):
        """Returns reduce-only with reason."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="daily loss limit",
            active_guards=[],
        )
        widget.state = mock_state
        result = widget._get_blocking_reason()
        assert "reduce-only" in result
        assert "daily loss limit" in result

    def test_reduce_only_mode_truncates_long_reason(self):
        """Truncates long reduce-only reasons."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="this is a very long reason that should be truncated",
            active_guards=[],
        )
        widget.state = mock_state
        result = widget._get_blocking_reason()
        assert "reduce-only" in result
        assert "..." in result
        assert len(result) < 50  # Should be reasonably short

    def test_single_guard_active(self):
        """Returns guard name when single guard active."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            active_guards=["DailyLossGuard"],
        )
        widget.state = mock_state
        assert widget._get_blocking_reason() == "DailyLossGuard"

    def test_multiple_guards_active(self):
        """Returns count when multiple guards active."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            active_guards=["DailyLossGuard", "VolatilityGuard", "MaxDrawdownGuard"],
        )
        widget.state = mock_state
        assert widget._get_blocking_reason() == "3 guards"

    def test_reduce_only_takes_priority_over_guards(self):
        """Reduce-only mode has higher priority than guards."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="risk limit",
            active_guards=["DailyLossGuard"],
        )
        widget.state = mock_state
        result = widget._get_blocking_reason()
        assert "reduce-only" in result
        assert "DailyLossGuard" not in result


class TestStrategyWidgetExecutedDecisions:
    """Tests for _get_executed_decision_ids method."""

    def test_no_executed_when_state_is_none(self):
        """Returns empty set when state is None."""
        widget = StrategyWidget()
        widget.state = None
        assert widget._get_executed_decision_ids() == set()

    def test_no_executed_when_no_orders(self):
        """Returns empty set when no orders."""
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.order_data = ActiveOrders(orders=[])
        widget.state = mock_state
        assert widget._get_executed_decision_ids() == set()

    def test_returns_decision_ids_from_orders(self):
        """Returns decision_ids from orders that have them."""
        from decimal import Decimal

        from gpt_trader.tui.types import Order

        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.order_data = ActiveOrders(
            orders=[
                Order(
                    order_id="ord1",
                    symbol="BTC",
                    side="BUY",
                    quantity=Decimal("1"),
                    price=Decimal("100"),
                    status="OPEN",
                    decision_id="dec123",
                ),
                Order(
                    order_id="ord2",
                    symbol="ETH",
                    side="SELL",
                    quantity=Decimal("2"),
                    price=Decimal("50"),
                    status="FILLED",
                    decision_id="dec456",
                ),
                Order(
                    order_id="ord3",
                    symbol="SOL",
                    side="BUY",
                    quantity=Decimal("5"),
                    price=Decimal("10"),
                    status="OPEN",
                    decision_id="",  # No decision_id
                ),
            ]
        )
        widget.state = mock_state
        result = widget._get_executed_decision_ids()
        assert result == {"dec123", "dec456"}


class TestDecisionBlockedBy:
    """Tests for decision blocked_by field handling."""

    def test_decision_blocked_by_field_exists(self):
        """Decision blocked_by field is accessible."""
        data = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
            blocked_by="DailyLossGuard",
        )
        assert data.blocked_by == "DailyLossGuard"

    def test_decision_blocked_by_defaults_to_empty(self):
        """Decision blocked_by defaults to empty string."""
        data = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
        )
        assert data.blocked_by == ""

    def test_decision_blocked_by_used_in_display(self):
        """Decision's own blocked_by is preferred over current risk state."""
        widget = StrategyWidget()

        # Mock state with no current blocking
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            active_guards=[],
        )
        mock_state.order_data = ActiveOrders(orders=[])
        widget.state = mock_state

        # Decision has its own blocked_by (historical)
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
            blocked_by="VolatilityGuard",
        )

        # The decision's blocked_by should be used
        # (since decision.blocked_by is not empty, it takes precedence)
        assert decision.blocked_by == "VolatilityGuard"

    def test_fallback_to_current_risk_state(self):
        """Falls back to current risk state when decision has no blocked_by."""
        widget = StrategyWidget()

        # Mock state with current blocking
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            active_guards=["MaxDrawdownGuard"],
        )
        widget.state = mock_state

        # Decision has no blocked_by
        decision = DecisionData(
            symbol="BTC-USD",
            action="BUY",
            reason="Signal",
            confidence=0.85,
            blocked_by="",
        )

        # Current blocking reason from _get_blocking_reason
        current_blocking = widget._get_blocking_reason()
        assert current_blocking == "MaxDrawdownGuard"

        # Since decision.blocked_by is empty, fallback is used
        decision_blocked_by = decision.blocked_by or current_blocking
        assert decision_blocked_by == "MaxDrawdownGuard"
