"""
Trades widget for displaying recent trade history.

This widget displays a table of executed trades with P&L calculation
using the TradeMatcher for matching buy/sell pairs.

Uses row keys for efficient DataTable updates - only adds/removes
changed rows instead of rebuilding the entire table.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.events import TradeMatcherResetRequested
from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.trade_matcher import TradeMatcher
from gpt_trader.tui.types import Trade
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class TradesWidget(Static):
    """Displays recent trades with P&L calculation.

    Shows executed trades with their details including symbol, side,
    quantity, price, order ID, calculated P&L, and timestamp.

    The widget uses a TradeMatcher to calculate P&L by matching
    buy and sell trades for the same symbol.
    """

    # Styles moved to styles/widgets/portfolio.tcss

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        table = DataTable(id="trades-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield Label("", id="trades-empty", classes="empty-state")

    def on_mount(self) -> None:
        """Initialize the trades table with columns and trade matcher."""
        table = self.query_one("#trades-table", DataTable)
        # Add columns - alignment handled in add_row with Text objects
        table.add_columns("Symbol", "Side", "Quantity", "Price", "Order ID", "P&L", "Time")
        # Initialize trade matcher
        self._trade_matcher = TradeMatcher()

    def on_trade_matcher_reset_requested(
        self, event: TradeMatcherResetRequested  # noqa: ARG002
    ) -> None:
        """Handle trade matcher reset request from event system.

        This event handler replaces direct access to _trade_matcher.reset()
        from external components like BotLifecycleManager.

        Args:
            event: The reset request event (unused but required by handler signature).
        """
        if hasattr(self, "_trade_matcher"):
            logger.debug("[TradesWidget] Resetting trade matcher via event")
            self._trade_matcher.reset()

    @safe_update
    def update_trades(self, trades: list[Trade]) -> None:
        """Update the trades table with current data.

        Uses row keys for efficient diffing - only adds/removes changed rows.

        Args:
            trades: List of Trade objects to display.
        """
        table = self.query_one("#trades-table", DataTable)
        empty_label = self.query_one("#trades-empty", Label)

        # Show empty state or data
        if not trades:
            # Clear all rows when empty
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_label.display = True
            empty_label.update("No trades yet. Trade history appears after execution.")
        else:
            table.display = True
            empty_label.display = False

            # Process trades to calculate P&L matches
            pnl_map = self._trade_matcher.process_trades(trades)

            # Get current row keys
            existing_keys = set(table.rows.keys())
            new_keys = {trade.trade_id for trade in trades}

            # Remove trades no longer present
            for key in existing_keys - new_keys:
                try:
                    table.remove_row(key)
                    logger.debug(f"Removed trade row: {key}")
                except Exception:
                    pass  # Row may not exist

            # Add/update trades
            for trade in trades:
                row_data = self._format_trade_row(trade, pnl_map)

                if trade.trade_id in existing_keys:
                    # Update existing row in-place
                    try:
                        self._update_row_cells(table, trade.trade_id, row_data)
                    except Exception:
                        # Fallback: remove and re-add if update fails
                        try:
                            table.remove_row(trade.trade_id)
                        except Exception:
                            pass
                        table.add_row(*row_data, key=trade.trade_id)
                else:
                    # Add new trade
                    table.add_row(*row_data, key=trade.trade_id)
                    logger.debug(f"Added new trade row: {trade.trade_id}")

    def _format_trade_row(
        self,
        trade: Trade,
        pnl_map: dict[str, str],
    ) -> tuple:
        """Format a trade into row data tuple.

        Args:
            trade: Trade object to format.
            pnl_map: Dictionary of trade_id to P&L string.

        Returns:
            Tuple of formatted cell values.
        """
        # Colorize Side
        side_color = THEME.colors.success if trade.side == "BUY" else THEME.colors.error
        formatted_side = f"[{side_color}]{trade.side}[/{side_color}]"

        # Simplify time string if needed
        time_str = trade.time
        if "T" in time_str:
            try:
                time_str = time_str.split("T")[1].split(".")[0]
            except IndexError:
                pass

        # Get P&L display value and apply color coding
        pnl_str: str = pnl_map.get(trade.trade_id, "N/A") or "N/A"

        # Parse P&L value
        try:
            pnl_value = float(pnl_str) if pnl_str != "N/A" else None
        except (ValueError, TypeError):
            pnl_value = None

        # Apply color based on value
        if pnl_value is None:
            pnl_display = Text(pnl_str or "N/A", style="dim", justify="right")
        elif pnl_value > 0:
            pnl_display = Text.from_markup(
                f"[{THEME.colors.success}]{pnl_str}[/{THEME.colors.success}]",
                justify="right",
            )
        elif pnl_value < 0:
            pnl_display = Text.from_markup(
                f"[{THEME.colors.error}]{pnl_str}[/{THEME.colors.error}]",
                justify="right",
            )
        else:
            # Zero P&L - neutral
            pnl_display = Text(pnl_str, justify="right")

        return (
            trade.symbol,
            formatted_side,  # Preserves color markup
            Text(format_quantity(trade.quantity), justify="right"),
            Text(format_price(trade.price), justify="right"),
            trade.order_id[-8:] if trade.order_id else "",
            pnl_display,  # P&L column
            Text(time_str, justify="right"),
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
        for col_key, value in zip(columns, row_data):
            table.update_cell(row_key, col_key, value)
