from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.types import StrategyState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


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

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update strategy automatically."""
        if state is None:
            return

        logger.debug(
            f"[StrategyWidget] State update: "
            f"strategies={len(state.strategy_data.active_strategies)}, "
            f"decisions={len(state.strategy_data.last_decisions)}"
        )

        self.update_strategy(state.strategy_data)

    def compose(self) -> ComposeResult:
        yield Label("🎯 STRATEGY DECISIONS", classes="header")
        table = DataTable(id="strategy-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield Label("", id="strategy-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Action", "Confidence", "Reason", "Time")

    @safe_update
    def update_strategy(self, data: StrategyState) -> None:
        table = self.query_one(DataTable)
        empty_label = self.query_one("#strategy-empty", Label)
        table.clear()

        # Handle empty state
        if not data.last_decisions:
            table.display = False
            empty_label.display = True
            empty_label.update("🎯 No decisions • Waiting for strategy analysis")
            return
        else:
            table.display = True
            empty_label.display = False

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
