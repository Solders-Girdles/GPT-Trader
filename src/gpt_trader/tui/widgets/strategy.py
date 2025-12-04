from datetime import datetime

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.types import StrategyState


class StrategyWidget(Static):
    """Displays strategy status and decisions."""

    DEFAULT_CSS = """
    StrategyWidget {
        layout: vertical;
        height: 1fr;
    }

    StrategyWidget DataTable {
        height: 1fr;
    }
    """

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state) -> None:  # type: ignore[no-untyped-def]
        """React to state changes - update strategy automatically."""
        if state is None:
            return
        self.update_strategy(state.strategy_data)

    def compose(self) -> ComposeResult:
        yield Label("🎯 STRATEGY DECISIONS", classes="header")
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

            # Format timestamp
            time_str = ""
            if decision.timestamp > 0:
                time_str = datetime.fromtimestamp(decision.timestamp).strftime("%H:%M:%S")

            # Color code action
            color = THEME.colors.text_primary
            if action == "BUY":
                color = THEME.colors.success
            elif action == "SELL":
                color = THEME.colors.error
            elif action == "HOLD":
                color = THEME.colors.warning

            formatted_action = f"[{color}]{action}[/{color}]"
            table.add_row(symbol, formatted_action, confidence, reason, time_str)
