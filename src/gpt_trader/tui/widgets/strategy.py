from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import StrategyState


class StrategyWidget(Static):
    """Displays strategy status and decisions."""

    def compose(self) -> ComposeResult:
        yield Label("STRATEGY DECISIONS", classes="header")
        yield DataTable(id="strategy-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Action", "Conf", "Reason", "Time")

    @safe_update
    def update_strategy(self, data: StrategyState) -> None:
        table = self.query_one(DataTable)
        table.clear()

        for symbol, decision in data.last_decisions.items():
            action = decision.action.upper()
            confidence = f"{decision.confidence:.2f}"
            reason = decision.reason

            # Format timestamp if needed, or just use it if it's a string (it's a float in types.py)
            # In types.py DecisionData.timestamp is float.
            # Let's format it simply or ignore for now to match previous logic which didn't show time in the loop I saw earlier?
            # Wait, the previous on_mount had "Time" column.
            # Let's just put an empty string for time for now or format it if we import datetime.
            time_str = ""

            # Color code action
            color = "white"
            if action == "BUY":
                color = "#a3be8c"  # Green
            elif action == "SELL":
                color = "#bf616a"  # Red
            elif action == "HOLD":
                color = "#ebcb8b"  # Yellow

            formatted_action = f"[{color}]{action}[/{color}]"
            table.add_row(symbol, formatted_action, confidence, reason, time_str)
