from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.models import StrategyData


class StrategyWidget(Static):
    """Widget to display strategy status and decisions."""

    def compose(self) -> ComposeResult:
        yield Label("STRATEGY ENGINE", classes="header")
        with Vertical(classes="strategy-container"):
            yield Label("Active Strategies:", classes="subheader")
            yield Label("-", id="active-strategies")
            yield Label("Latest Decisions:", classes="subheader")
            yield DataTable(id="decisions-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#decisions-table", DataTable)
        table.add_columns("Symbol", "Action", "Conf", "Reason")

    def update_strategy(self, data: StrategyData) -> None:
        # Update Active Strategies
        strategies_label = self.query_one("#active-strategies", Label)
        if data.active_strategies:
            strategies_label.update(", ".join(data.active_strategies))
        else:
            strategies_label.update("None")

        # Update Decisions Table
        table = self.query_one("#decisions-table", DataTable)
        table.clear()

        for symbol, decision in data.last_decisions.items():
            action = decision.action.upper()
            confidence = f"{decision.confidence:.2f}"
            reason = decision.reason

            # Color code action
            color = "white"
            if action == "BUY":
                color = "#a3be8c"  # Green
            elif action == "SELL":
                color = "#bf616a"  # Red
            elif action == "HOLD":
                color = "#ebcb8b"  # Yellow

            formatted_action = f"[{color}]{action}[/{color}]"
            table.add_row(symbol, formatted_action, confidence, reason)
