"""
Trades widget for displaying recent trade history.

This widget displays a table of executed trades with P&L calculation
using the TradeMatcher for matching buy/sell pairs.

Uses row keys for efficient DataTable updates - only adds/removes
changed rows instead of rebuilding the entire table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.events import TradeMatcherResetRequested
from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.trade_matcher import TradeMatcher
from gpt_trader.tui.types import Trade
from gpt_trader.tui.widgets.table_copy_mixin import TableCopyMixin
from gpt_trader.tui.widgets.tile_states import TileEmptyState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class TradesWidget(TableCopyMixin, Static):
    """Displays recent trades with P&L calculation.

    Shows executed trades with their details including symbol, side,
    quantity, price, order ID, calculated P&L, and timestamp.

    The widget uses a TradeMatcher to calculate P&L by matching
    buy and sell trades for the same symbol.

    Keyboard shortcuts:
        c: Copy selected row to clipboard
        C: Copy all rows to clipboard
        f: Cycle symbol filter
        F: Clear all filters
    """

    BINDINGS = [
        *TableCopyMixin.COPY_BINDINGS,
        Binding("f", "cycle_symbol_filter", "Filter Symbol", show=True),
        Binding("F", "clear_filters", "Clear Filters", show=True),
    ]

    # Filter state
    symbol_filter: reactive[str] = reactive("")
    _all_symbols: list[str] = []
    _state: TuiState | None = None

    # Styles moved to styles/widgets/portfolio.tcss

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Label("", id="trades-filter-status", classes="filter-status hidden")
        table = DataTable(id="trades-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield TileEmptyState(
            title="No Trade History",
            subtitle="Trades appear after execution",
            icon="○",
            actions=["[S] Start Bot", "[R] Refresh"],
            id="trades-empty",
        )

    def on_mount(self) -> None:
        """Initialize the trades table with columns and trade matcher."""
        table = self.query_one("#trades-table", DataTable)
        # Add columns - alignment handled in add_row with Text objects
        # Links column shows badges for order (⬚) and decision (◇) linkage
        table.add_columns("Symbol", "Links", "Side", "Quantity", "Price", "Order ID", "P&L", "Time")
        # Initialize trade matcher
        self._trade_matcher = TradeMatcher()

    def on_trade_matcher_reset_requested(
        self,
        event: TradeMatcherResetRequested,  # noqa: ARG002
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
    def update_trades(self, trades: list[Trade], state: TuiState | None = None) -> None:
        """Update the trades table with current data.

        Uses row keys for efficient diffing - only adds/removes changed rows.

        Args:
            trades: List of Trade objects to display.
            state: Optional TuiState for linkage lookups.
        """
        # Store state for linkage lookups
        self._state = state

        # Collect all symbols for filter cycling
        self._all_symbols = sorted({t.symbol for t in trades})

        # Apply symbol filter
        filtered_trades = trades
        if self.symbol_filter:
            filtered_trades = [t for t in trades if t.symbol == self.symbol_filter]

        # Update filter status label
        self._update_filter_status(len(trades), len(filtered_trades))

        table = self.query_one("#trades-table", DataTable)
        empty_state = self.query_one("#trades-empty", TileEmptyState)

        # Show empty state or data
        if not filtered_trades:
            # Clear all rows when empty
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_state.display = True
            if self.symbol_filter and trades:
                # Filtered to empty - show filter message
                empty_state.update_state(
                    title=f"No {self.symbol_filter} Trades",
                    subtitle="Press F to clear filters",
                    icon="⚙",
                    actions=["[F] Clear Filter"],
                )
            else:
                # Keep default message - trades only appear after execution
                empty_state.update_state(
                    title="No Trade History",
                    subtitle="Trades appear after execution",
                    icon="○",
                    actions=["[S] Start Bot", "[R] Refresh"],
                )
            return
        else:
            table.display = True
            empty_state.display = False

            # Process trades to calculate P&L matches
            pnl_map = self._trade_matcher.process_trades(trades)

            # Get current row keys
            existing_keys = set(table.rows.keys())
            new_keys = {trade.trade_id for trade in filtered_trades}

            # Remove trades no longer present (includes filtered out)
            for key in existing_keys - new_keys:
                try:
                    table.remove_row(key)
                    logger.debug(f"Removed trade row: {key}")
                except Exception:
                    pass  # Row may not exist

            # Add/update trades
            for trade in filtered_trades:
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
        # Build linkage badges
        links_text = self._get_linkage_badges(trade)

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
            links_text,  # Linkage badges column
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

    def _get_linkage_badges(self, trade: Trade) -> Text:
        """Build linkage badges showing order and decision connections.

        Badge legend:
            ⬚ = Has linked order (cyan if found)
            ◇ = Has linked decision (yellow if found)

        Args:
            trade: Trade to check for linkages.

        Returns:
            Text with colored badges.
        """
        badges = []

        # Check for order linkage
        has_order = bool(trade.order_id)
        order_linked = False

        if has_order and self._state and self._state.order_data:
            # Check if order still exists in current orders
            order_linked = any(o.order_id == trade.order_id for o in self._state.order_data.orders)

        if has_order:
            if order_linked:
                badges.append("[cyan]⬚[/cyan]")  # Order found
            else:
                badges.append("[dim]⬚[/dim]")  # Order existed but not current

        # Check for decision linkage via order_id -> decision_id mapping
        has_decision = False
        decision_linked = False

        if has_order and self._state and self._state.strategy_data:
            # Look for decision with matching decision_id derived from order
            for decision in self._state.strategy_data.last_decisions:
                if decision.decision_id and trade.order_id:
                    # Decision ID format is typically "timestamp_symbol"
                    # Order might reference it via decision_id field if present
                    if decision.symbol == trade.symbol:
                        has_decision = True
                        # Check if decision action matches trade side
                        if (decision.action == "BUY" and trade.side == "BUY") or (
                            decision.action == "SELL" and trade.side == "SELL"
                        ):
                            decision_linked = True
                            break

        if has_decision:
            if decision_linked:
                badges.append("[yellow]◇[/yellow]")  # Decision matched
            else:
                badges.append("[dim]◇[/dim]")  # Decision exists but not matched

        if badges:
            return Text.from_markup(" ".join(badges))
        return Text("[dim]--[/dim]")

    def _update_filter_status(self, total: int, filtered: int) -> None:
        """Update the filter status label.

        Args:
            total: Total number of trades before filtering.
            filtered: Number of trades after filtering.
        """
        try:
            status_label = self.query_one("#trades-filter-status", Label)
            if self.symbol_filter:
                status_label.update(f"[dim]Filter:[/dim] {self.symbol_filter} ({filtered}/{total})")
                status_label.remove_class("hidden")
            else:
                status_label.update("")
                status_label.add_class("hidden")
        except Exception as e:
            logger.debug(f"Failed to update filter status: {e}")

    def action_cycle_symbol_filter(self) -> None:
        """Cycle through symbol filters."""
        if not self._all_symbols:
            return

        if not self.symbol_filter:
            # Start with first symbol
            self.symbol_filter = self._all_symbols[0]
        else:
            # Find current index and move to next
            try:
                current_idx = self._all_symbols.index(self.symbol_filter)
                next_idx = (current_idx + 1) % (len(self._all_symbols) + 1)
                if next_idx == len(self._all_symbols):
                    # Wrap around to no filter
                    self.symbol_filter = ""
                else:
                    self.symbol_filter = self._all_symbols[next_idx]
            except ValueError:
                # Current filter not in list, reset
                self.symbol_filter = ""

        # Trigger refresh
        self._refresh_with_current_state()

    def action_clear_filters(self) -> None:
        """Clear all active filters."""
        self.symbol_filter = ""
        self._refresh_with_current_state()

    def _refresh_with_current_state(self) -> None:
        """Refresh display using cached state data."""
        if self._state and self._state.trade_data:
            self.update_trades(self._state.trade_data.trades, self._state)
