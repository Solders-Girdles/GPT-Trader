from unittest.mock import MagicMock

from textual.widgets import DataTable, Label

from gpt_trader.tui.types import DecisionData, StrategyState
from gpt_trader.tui.widgets.strategy import StrategyWidget


class TestStrategyWidget:
    def test_update_strategy(self):
        widget = StrategyWidget()

        # Mock query_one
        mock_label = MagicMock(spec=Label)
        mock_table = MagicMock(spec=DataTable)

        def query_side_effect(arg, *args, **kwargs):
            if arg == "#active-strategies":
                return mock_label
            if arg == "#decisions-table":
                return mock_table
            return MagicMock()

        widget.query_one = MagicMock(side_effect=query_side_effect)

        # Create data
        data = StrategyState(
            active_strategies=["StrategyA", "StrategyB"],
            last_decisions={
                "BTC-USD": DecisionData(
                    symbol="BTC-USD", action="BUY", reason="Signal", confidence=0.95
                )
            },
        )

        # Call update
        widget.update_strategy(data)

        # Verify Label update
        mock_label.update.assert_called_with("StrategyA, StrategyB")

        # Verify Table update
        mock_table.clear.assert_called_once()
        args, _ = mock_table.add_row.call_args
        assert args[0] == "BTC-USD"
        assert "BUY" in args[1]
        assert args[2] == "0.95"
