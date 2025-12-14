from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.types import DecisionData, StrategyState
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
        assert args[2] == "0.95"
        assert kwargs.get("key") == "BTC-USD"  # Row key for delta updates
