from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.types import ActiveOrders, DecisionData, RiskGuard, RiskState, StrategyState
from gpt_trader.tui.widgets.strategy import StrategyWidget


class TestStrategyWidget:
    def test_update_strategy(self):
        widget = StrategyWidget()

        mock_table = MagicMock(spec=DataTable)
        mock_empty_label = MagicMock()

        mock_table.rows = MagicMock()
        mock_table.rows.keys.return_value = {}
        mock_table.columns = MagicMock()
        mock_table.columns.keys.return_value = ["Symbol", "Action", "Confidence", "Reason", "Time"]

        def query_one_side_effect(selector, widget_type=None):
            if selector == DataTable or widget_type == DataTable:
                return mock_table
            if "empty" in str(selector):
                return mock_empty_label
            return mock_table

        widget.query_one = MagicMock(side_effect=query_one_side_effect)

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

        widget.update_strategy(data)

        mock_table.add_row.assert_called_once()
        args, kwargs = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert "0.95" in args[2]
        assert "HIGH" in args[2]
        assert kwargs.get("key") == "BTC-USD"


class TestStrategyWidgetBlockingReason:
    """Tests for _get_blocking_reason method."""

    def test_no_blocking_when_state_is_none(self):
        widget = StrategyWidget()
        widget.state = None
        assert widget._get_blocking_reason() == ""

    def test_no_blocking_when_no_risk_conditions(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            guards=[],
        )
        widget.state = mock_state
        assert widget._get_blocking_reason() == ""

    def test_reduce_only_mode_with_reason(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="daily loss limit",
            guards=[],
        )
        widget.state = mock_state
        result = widget._get_blocking_reason()
        assert "reduce-only" in result
        assert "daily loss limit" in result

    def test_reduce_only_mode_truncates_long_reason(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="this is a very long reason that should be truncated",
            guards=[],
        )
        widget.state = mock_state
        result = widget._get_blocking_reason()
        assert "reduce-only" in result
        assert "..." in result
        assert len(result) < 50

    def test_single_guard_active(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            guards=[RiskGuard(name="DailyLossGuard")],
        )
        widget.state = mock_state
        assert widget._get_blocking_reason() == "DailyLossGuard"

    def test_multiple_guards_active(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=False,
            guards=[
                RiskGuard(name="DailyLossGuard"),
                RiskGuard(name="VolatilityGuard"),
                RiskGuard(name="MaxDrawdownGuard"),
            ],
        )
        widget.state = mock_state
        assert widget._get_blocking_reason() == "3 guards"

    def test_reduce_only_takes_priority_over_guards(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.risk_data = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="risk limit",
            guards=[RiskGuard(name="DailyLossGuard")],
        )
        widget.state = mock_state
        result = widget._get_blocking_reason()
        assert "reduce-only" in result
        assert "DailyLossGuard" not in result


class TestStrategyWidgetExecutedDecisions:
    """Tests for _get_executed_decision_ids method."""

    def test_no_executed_when_state_is_none(self):
        widget = StrategyWidget()
        widget.state = None
        assert widget._get_executed_decision_ids() == set()

    def test_no_executed_when_no_orders(self):
        widget = StrategyWidget()
        mock_state = MagicMock()
        mock_state.order_data = ActiveOrders(orders=[])
        widget.state = mock_state
        assert widget._get_executed_decision_ids() == set()

    def test_returns_decision_ids_from_orders(self):
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
                    decision_id="",
                ),
            ]
        )
        widget.state = mock_state
        result = widget._get_executed_decision_ids()
        assert result == {"dec123", "dec456"}
