"""
Positions widget for displaying active trading positions.

This widget displays a table of active positions with P&L information,
entry/current prices, and leverage indicators.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import (
    format_currency,
    format_percentage,
    format_price,
    format_quantity,
)
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import Position
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class PositionsWidget(Static):
    """Displays active positions with P&L and leverage information.

    This widget shows a data table with all active trading positions,
    including entry price, current price, unrealized P&L, and leverage.

    Attributes:
        state: Reactive TuiState for automatic updates when state changes.
    """

    DEFAULT_CSS = """
    PositionsWidget {
        layout: vertical;
        height: 1fr;
    }

    PositionsWidget DataTable {
        height: 1fr;
    }
    """

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update positions automatically."""
        if state is None:
            return

        logger.debug(
            f"[PositionsWidget] State update: "
            f"positions={len(state.position_data.positions)}, "
            f"total_pnl={state.position_data.total_unrealized_pnl}, "
            f"timestamp={state.last_update_timestamp:.2f}"
        )

        self.update_positions(
            state.position_data.positions,
            state.position_data.total_unrealized_pnl,
        )

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Label("ACTIVE POSITIONS", classes="header")
        table = DataTable(id="positions-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield Label("", id="positions-empty", classes="empty-state")

    def on_mount(self) -> None:
        """Initialize the positions table with columns."""
        table = self.query_one(DataTable)
        # Add columns - alignment handled in add_row with Text objects
        table.add_columns("Symbol", "Quantity", "Entry", "Current", "PnL", "%", "Leverage")

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

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def update_positions(
        self,
        positions: dict[str, Position],
        total_pnl: Decimal,
        risk_data: dict[str, float] | None = None,
    ) -> None:
        """Update the positions table with current data.

        Uses row keys for efficient diffing - only adds/removes changed rows.

        Args:
            positions: Dictionary mapping symbols to Position objects.
            total_pnl: Total unrealized P&L across all positions.
            risk_data: Optional dictionary of leverage values per symbol.
        """
        table = self.query_one(DataTable)
        empty_label = self.query_one("#positions-empty", Label)

        # Show empty state or data
        if not positions:
            # Clear all rows when empty
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_label.display = True

            # Mode-aware empty state message (single-line format)
            if self.app and hasattr(self.app, "data_source_mode"):
                mode = self.app.data_source_mode  # type: ignore[attr-defined]
                if mode == "read_only":
                    empty_label.update("No positions - Read-only mode - observing market data")
                else:
                    empty_label.update("No positions - Press [S] to start bot")
            else:
                empty_label.update("No positions - Press [S] to start bot")
        else:
            table.display = True
            empty_label.display = False

            # Get current row keys
            existing_keys = set(table.rows.keys())
            new_keys = set(positions.keys())

            # Remove closed positions (no longer in positions dict)
            for key in existing_keys - new_keys:
                try:
                    table.remove_row(key)
                    logger.debug(f"Removed closed position row: {key}")
                except Exception:
                    pass  # Row may not exist

            # Add/update positions
            for symbol, pos in positions.items():
                row_data = self._format_position_row(pos, risk_data)

                if symbol in existing_keys:
                    # Update existing row in-place using update_cell
                    try:
                        self._update_row_cells(table, symbol, row_data)
                    except Exception:
                        # Fallback: remove and re-add if update fails
                        try:
                            table.remove_row(symbol)
                        except Exception:
                            pass
                        table.add_row(*row_data, key=symbol)
                else:
                    # Add new position
                    table.add_row(*row_data, key=symbol)
                    logger.debug(f"Added new position row: {symbol}")

    def _format_position_row(
        self,
        pos: Position,
        risk_data: dict[str, float] | None,
    ) -> tuple:
        """Format a position into row data tuple.

        Args:
            pos: Position object to format.
            risk_data: Optional leverage data.

        Returns:
            Tuple of formatted cell values.
        """
        # Calculate P&L percentage
        try:
            if pos.entry_price > 0:
                pnl_pct = ((pos.mark_price - pos.entry_price) / pos.entry_price) * 100
                pnl_pct_str = format_percentage(Decimal(str(pnl_pct)))
            else:
                pnl_pct_str = "N/A"
        except (ValueError, ZeroDivisionError):
            pnl_pct_str = "N/A"

        # Get leverage from risk data
        leverage_val = 1.0
        if risk_data and pos.symbol in risk_data:
            leverage_val = risk_data[pos.symbol]

        # Format and color-code leverage
        leverage_str = f"{leverage_val:.1f}x"
        if leverage_val < 2.0:
            leverage_display = f"[green]{leverage_str}[/green]"
        elif leverage_val < 5.0:
            leverage_display = f"[yellow]{leverage_str}[/yellow]"
        else:
            leverage_display = f"[red]{leverage_str}[/red]"

        # Use mark_price as current price, fallback to entry if not available
        current_price = (
            pos.mark_price if pos.mark_price and pos.mark_price != Decimal("0") else pos.entry_price
        )

        return (
            pos.symbol,
            Text(format_quantity(pos.quantity), justify="right"),
            Text(format_price(pos.entry_price), justify="right"),
            Text(format_price(current_price), justify="right"),
            Text(format_currency(pos.unrealized_pnl), justify="right"),
            Text(pnl_pct_str, justify="right"),
            Text.from_markup(leverage_display, justify="right"),
        )

    def _update_row_cells(
        self,
        table: DataTable,
        row_key: str,
        row_data: tuple,
    ) -> None:
        """Update cells in an existing row.

        Args:
            table: DataTable to update.
            row_key: Key of the row to update.
            row_data: New data for the row cells.
        """
        # Get column keys
        columns = list(table.columns.keys())

        # Update each cell
        for col_idx, (col_key, value) in enumerate(zip(columns, row_data)):
            table.update_cell(row_key, col_key, value)
