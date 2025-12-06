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

        # Register with state registry for state updates
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from state registry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Called by StateRegistry when state changes."""
        self.state = state

    @safe_update
    def update_strategy(self, data: StrategyState) -> None:
        table = self.query_one(DataTable)
        empty_label = self.query_one("#strategy-empty", Label)

        # Handle empty state with actionable hint
        if not data.last_decisions:
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_label.display = True
            empty_label.update("No decisions yet • Press [S] to start bot")
            return
        else:
            table.display = True
            empty_label.display = False

        # Get current row keys for delta updates
        existing_keys = set(table.rows.keys())
        new_keys = set(data.last_decisions.keys())

        # Remove symbols no longer in decisions
        for key in existing_keys - new_keys:
            try:
                table.remove_row(key)
            except Exception:
                pass

        # Get column keys for cell updates
        columns = list(table.columns.keys()) if table.columns else []

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
            row_data = (symbol, formatted_action, confidence, reason, time_str)

            if symbol in existing_keys:
                # Update existing row in-place
                try:
                    for col_idx, (col_key, value) in enumerate(zip(columns, row_data)):
                        table.update_cell(symbol, col_key, value)
                except Exception:
                    # Fallback: remove and re-add if update fails
                    try:
                        table.remove_row(symbol)
                    except Exception:
                        pass
                    table.add_row(*row_data, key=symbol)
            else:
                # Add new decision
                table.add_row(*row_data, key=symbol)
