from unittest.mock import MagicMock

from textual.widgets import DataTable

from gpt_trader.tui.types import DecisionData, StrategyState
from gpt_trader.tui.widgets.strategy import StrategyWidget


class TestStrategyWidget:
    def test_update_strategy(self):
        widget = StrategyWidget()

        # Mock query_one
        mock_table = MagicMock(spec=DataTable)
        widget.query_one = MagicMock(return_value=mock_table)

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

        # Verify Table update
        mock_table.clear.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert args[2] == "0.95"
