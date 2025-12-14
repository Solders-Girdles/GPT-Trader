"""
Market Detail Screen with enhanced watchlist and depth visualization.

Extends the basic market screen with:
- Full watchlist table with sorting (symbol, price, change, spread)
- Order book depth visualization (top 5 bids/asks)
- Price chart sparkline history with color trend
- Data state indicators (loading, empty, error)
- Watchlist editing via 'W' key
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import format_price
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.utilities import (
    copy_to_clipboard,
    format_direction_colored,
    get_sort_indicator,
    sort_table_data,
)
from gpt_trader.tui.widgets import ContextualFooter
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState
    from gpt_trader.tui.types import MarketState

logger = get_logger(__name__, component="tui")


class SparklineChart(Static):
    """Compact sparkline chart widget showing price history."""

    def __init__(self, symbol: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self._history: list[Decimal] = []

    def update_history(self, history: list[Decimal]) -> None:
        """Update price history and redraw sparkline."""
        self._history = history
        self._redraw()

    def _redraw(self) -> None:
        """Redraw the sparkline."""
        if not self._history or len(self._history) < 2:
            self.update(Text("─" * 20, style="dim"))
            return

        # Generate sparkline
        blocks = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        recent = self._history[-20:]  # Last 20 points

        min_val = float(min(recent))
        max_val = float(max(recent))
        rng = max_val - min_val

        if rng == 0:
            sparkline = "▄" * len(recent)
        else:
            chars = []
            for price in recent:
                normalized = (float(price) - min_val) / rng
                index = int(normalized * (len(blocks) - 1))
                chars.append(blocks[index])
            sparkline = "".join(chars)

        # Color based on trend
        trend_up = recent[-1] >= recent[0]
        color = THEME.colors.success if trend_up else THEME.colors.error
        self.update(Text.from_markup(f"[{color}]{sparkline}[/{color}]"))


class DepthWidget(Static):
    """Order book depth visualization showing top bids/asks."""

    def compose(self) -> ComposeResult:
        """Compose the depth widget layout."""
        yield Label("ORDER BOOK", classes="widget-header")
        with Horizontal(classes="depth-container"):
            with Vertical(classes="depth-side bids"):
                yield Label("BIDS", classes="depth-header")
                yield Static(id="bids-display", classes="depth-levels")
            with Vertical(classes="depth-side asks"):
                yield Label("ASKS", classes="depth-header")
                yield Static(id="asks-display", classes="depth-levels")

    def update_depth(
        self,
        bids: list[tuple[Decimal, Decimal]] | None = None,
        asks: list[tuple[Decimal, Decimal]] | None = None,
    ) -> None:
        """
        Update order book depth display.

        Args:
            bids: List of (price, quantity) tuples for bids (highest first).
            asks: List of (price, quantity) tuples for asks (lowest first).
        """
        try:
            bids_display = self.query_one("#bids-display", Static)
            asks_display = self.query_one("#asks-display", Static)

            if not bids and not asks:
                bids_display.update(Text("No data", style="dim"))
                asks_display.update(Text("No data", style="dim"))
                return

            # Format bids (green, highest price first)
            bid_lines = []
            for price, qty in (bids or [])[:5]:
                bid_lines.append(
                    f"[{THEME.colors.success}]{format_price(price)}[/{THEME.colors.success}] "
                    f"[dim]{float(qty):.4f}[/dim]"
                )
            bids_display.update(Text.from_markup("\n".join(bid_lines) or "No bids"))

            # Format asks (red, lowest price first)
            ask_lines = []
            for price, qty in (asks or [])[:5]:
                ask_lines.append(
                    f"[{THEME.colors.error}]{format_price(price)}[/{THEME.colors.error}] "
                    f"[dim]{float(qty):.4f}[/dim]"
                )
            asks_display.update(Text.from_markup("\n".join(ask_lines) or "No asks"))

        except Exception as e:
            logger.debug(f"Failed to update depth: {e}")


class MarketDetailScreen(Screen):
    """Enhanced market detail screen with watchlist, depth, and sparklines.

    Features:
    - Sortable watchlist table (symbol, price, change, spread)
    - Order book depth visualization
    - Price sparkline chart for selected symbol
    - Keyboard shortcuts for sorting, copying, watchlist editing

    Keyboard:
    - ESC/Q: Close and return to main screen
    - W: Edit watchlist
    - S: Cycle sort column
    - C: Copy selected row
    - ↑↓: Navigate table rows
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("w", "edit_watchlist", "Edit Watchlist", show=True),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("c", "copy_row", "Copy", show=True),
        Binding("r", "refresh", "Refresh", show=False),
    ]

    CSS = """
    #market-detail-container {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 2fr 1fr;
        grid-rows: 1fr auto;
        height: 1fr;
        padding: 1;
    }

    #watchlist-panel {
        column-span: 1;
        row-span: 2;
        border: round $border-primary;
        padding: 1;
    }

    #depth-panel {
        border: round $border-primary;
        padding: 1;
    }

    #sparkline-panel {
        border: round $border-primary;
        padding: 1;
    }

    .depth-container {
        height: auto;
    }

    .depth-side {
        width: 1fr;
        padding: 0 1;
    }

    .depth-header {
        text-style: bold;
        color: $text-secondary;
    }

    .depth-levels {
        height: auto;
        min-height: 5;
    }
    """

    # State
    state: reactive[TuiState | None] = reactive(None)

    # Sort state
    sort_column: reactive[str] = reactive("symbol")
    sort_ascending: reactive[bool] = reactive(True)

    # Selected symbol for detail view
    selected_symbol: reactive[str] = reactive("")

    def __init__(self, **kwargs) -> None:
        """Initialize MarketDetailScreen."""
        super().__init__(**kwargs)
        self._watchlist_data: list[dict[str, Any]] = []
        self._previous_prices: dict[str, float] = {}
        self._column_keys = ["symbol", "price", "change", "spread"]

    def compose(self) -> ComposeResult:
        """Compose the market detail screen layout."""
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )

        with Container(id="market-detail-container"):
            # Left: Watchlist table
            with Vertical(id="watchlist-panel"):
                yield Label("WATCHLIST", classes="widget-header")
                yield Label("", id="sort-indicator", classes="sort-hint")
                table = DataTable(
                    id="watchlist-table",
                    zebra_stripes=True,
                    cursor_type="row",
                )
                table.can_focus = True
                yield table
                yield Label("", id="watchlist-empty", classes="empty-state")

            # Right top: Order book depth
            with Container(id="depth-panel"):
                yield DepthWidget(id="depth-widget")

            # Right bottom: Sparkline chart
            with Vertical(id="sparkline-panel"):
                yield Label("PRICE TREND", classes="widget-header")
                yield Label("", id="sparkline-symbol", classes="symbol-label")
                yield SparklineChart(id="sparkline-chart")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        logger.debug("MarketDetailScreen mounted")

        # Set up table columns
        table = self.query_one("#watchlist-table", DataTable)
        table.add_column("Symbol", key="symbol")
        table.add_column("Price", key="price")
        table.add_column("Change", key="change")
        table.add_column("Spread", key="spread")

        # Register for state updates
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

        # Load initial state
        if hasattr(self.app, "tui_state"):
            self.state = self.app.tui_state  # type: ignore[attr-defined]

        # Update sort indicator
        self._update_sort_indicator()

    def on_unmount(self) -> None:
        """Clean up on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry."""
        self.state = state

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update market display."""
        if state is None:
            return

        self._update_watchlist(state.market_data)

    def watch_sort_column(self, column: str) -> None:
        """Handle sort column change."""
        self._update_sort_indicator()
        self._refresh_table()

    def watch_sort_ascending(self, ascending: bool) -> None:
        """Handle sort direction change."""
        self._update_sort_indicator()
        self._refresh_table()

    def watch_selected_symbol(self, symbol: str) -> None:
        """Update detail views when selected symbol changes."""
        if not symbol or not self.state:
            return

        # Update sparkline
        try:
            chart = self.query_one("#sparkline-chart", SparklineChart)
            symbol_label = self.query_one("#sparkline-symbol", Label)
            symbol_label.update(symbol)

            history = self.state.market_data.price_history.get(symbol, [])
            chart.update_history(history)
        except Exception as e:
            logger.debug(f"Failed to update sparkline: {e}")

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def _update_watchlist(self, market_data: MarketState) -> None:
        """Update watchlist table with market data."""
        table = self.query_one("#watchlist-table", DataTable)
        empty_label = self.query_one("#watchlist-empty", Label)

        prices = market_data.prices
        price_history = market_data.price_history
        spreads = market_data.spreads

        if not prices:
            table.display = False
            empty_label.display = True
            empty_label.update("No market data. Press [W] to edit watchlist.")
            return

        table.display = True
        empty_label.display = False

        # Build data rows
        data_rows: list[dict[str, Any]] = []
        for symbol, price in prices.items():
            current = float(price)
            prev = self._previous_prices.get(symbol, current)
            self._previous_prices[symbol] = current

            # Calculate change percentage
            change_pct = ((current - prev) / prev * 100) if prev != 0 else 0

            # Get spread if available
            spread = spreads.get(symbol, Decimal("0"))

            data_rows.append({
                "symbol": symbol,
                "price": current,
                "change": change_pct,
                "spread": float(spread),
                "_raw_price": price,
            })

        self._watchlist_data = data_rows

        # Sort data
        sorted_data = sort_table_data(
            data_rows,
            self.sort_column,
            self.sort_ascending,
            numeric_columns={"price", "change", "spread"},
        )

        # Update table
        self._populate_table(table, sorted_data)

        # Select first symbol if none selected
        if not self.selected_symbol and sorted_data:
            self.selected_symbol = sorted_data[0]["symbol"]

    def _populate_table(self, table: DataTable, data: list[dict[str, Any]]) -> None:
        """Populate table with sorted data."""
        existing_keys = set(table.rows.keys())
        new_keys = {row["symbol"] for row in data}

        # Remove old rows
        for key in existing_keys - new_keys:
            try:
                table.remove_row(key)
            except Exception:
                pass

        columns = list(table.columns.keys())

        for row in data:
            symbol = row["symbol"]
            price = row["price"]
            change = row["change"]
            spread = row["spread"]

            # Format cells
            price_color = THEME.colors.text_primary
            if symbol in self._previous_prices:
                prev = self._previous_prices.get(symbol, price)
                if price > prev:
                    price_color = THEME.colors.success
                elif price < prev:
                    price_color = THEME.colors.error

            formatted_price = f"[{price_color}]{format_price(Decimal(str(price)))}[/{price_color}]"
            formatted_change = format_direction_colored(change, f"{change:+.2f}%")
            formatted_spread = f"{spread:.4f}%" if spread else "--"

            row_data = (symbol, formatted_price, formatted_change, formatted_spread)

            if symbol in existing_keys:
                # Update existing row
                try:
                    for col_key, value in zip(columns, row_data):
                        table.update_cell(symbol, col_key, value)
                except Exception:
                    try:
                        table.remove_row(symbol)
                    except Exception:
                        pass
                    table.add_row(*row_data, key=symbol)
            else:
                table.add_row(*row_data, key=symbol)

    def _update_sort_indicator(self) -> None:
        """Update sort indicator label."""
        try:
            indicator = self.query_one("#sort-indicator", Label)
            arrow = get_sort_indicator(self.sort_column, self.sort_column, self.sort_ascending)
            col_display = self.sort_column.capitalize()
            indicator.update(f"Sorted by: {col_display}{arrow}")
        except Exception:
            pass

    def _refresh_table(self) -> None:
        """Re-sort and refresh the table."""
        if self._watchlist_data:
            sorted_data = sort_table_data(
                self._watchlist_data,
                self.sort_column,
                self.sort_ascending,
                numeric_columns={"price", "change", "spread"},
            )
            try:
                table = self.query_one("#watchlist-table", DataTable)
                self._populate_table(table, sorted_data)
            except Exception:
                pass

    # === Actions ===

    def action_dismiss(self) -> None:
        """Close screen and return to main."""
        self.app.pop_screen()

    def action_cycle_sort(self) -> None:
        """Cycle through sort columns."""
        current_idx = self._column_keys.index(self.sort_column)
        next_idx = (current_idx + 1) % len(self._column_keys)

        if next_idx == 0 and self.sort_column == self._column_keys[-1]:
            # Completed a cycle, toggle direction
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_ascending = True

        self.sort_column = self._column_keys[next_idx]
        self.notify(f"Sorted by {self.sort_column} {'↑' if self.sort_ascending else '↓'}", timeout=2)

    def action_copy_row(self) -> None:
        """Copy selected row to clipboard."""
        try:
            table = self.query_one("#watchlist-table", DataTable)
            cursor = table.cursor_coordinate
            if cursor is None:
                return

            row_key = table.get_row_at(cursor.row)
            if row_key is None:
                return

            # Find row data
            for row in self._watchlist_data:
                if row["symbol"] == row_key:
                    text = f"{row['symbol']}\t{row['price']}\t{row['change']:.2f}%\t{row['spread']:.4f}%"
                    if copy_to_clipboard(text):
                        self.notify("Row copied to clipboard", timeout=2)
                    else:
                        self.notify("Copy failed", severity="warning", timeout=2)
                    break
        except Exception as e:
            logger.debug(f"Copy row failed: {e}")

    def action_edit_watchlist(self) -> None:
        """Open watchlist editor modal."""
        # Import here to avoid circular imports
        try:
            from gpt_trader.tui.screens.watchlist_screen import WatchlistScreen
            self.app.push_screen(WatchlistScreen())
        except ImportError:
            self.notify("Watchlist editor not available", severity="warning", timeout=2)
            logger.debug("WatchlistScreen not yet implemented")

    def action_refresh(self) -> None:
        """Manually refresh market data."""
        if hasattr(self.app, "tui_state"):
            self.state = self.app.tui_state  # type: ignore[attr-defined]
            self.notify("Market data refreshed", timeout=2)

    # === Table Events ===

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection for detail views."""
        if event.row_key and event.row_key.value:
            self.selected_symbol = str(event.row_key.value)
