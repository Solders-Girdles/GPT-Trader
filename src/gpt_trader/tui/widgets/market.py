from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import format_price
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class MarketWatchWidget(Static):
    """Displays market data."""

    # Styles moved to styles/widgets/market.tcss

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update market data automatically."""
        if state is None:
            return

        # Precompute timestamp for logging (avoid invalid format spec)
        last_update_val = state.market_data.last_update or 0.0
        logger.debug(
            f"[MarketWatchWidget] State update: "
            f"symbols={len(state.market_data.prices)}, "
            f"last_update={last_update_val:.2f}"
        )

        self.update_prices(
            state.market_data.prices,
            state.market_data.last_update,
            state.market_data.price_history,
        )

    def _generate_sparkline(self, prices: list[Decimal], max_points: int = 12) -> str:
        """
        Generate Unicode sparkline from price history.

        Args:
            prices: List of historical prices (most recent last)
            max_points: Maximum number of points to display (default 12)

        Returns:
            String of Unicode block characters representing trend
        """
        if not prices or len(prices) < 2:
            return "─" * 8  # Flat line for insufficient data

        # Take last N points
        recent = prices[-max_points:]
        blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        # Normalize to 0-1 range
        min_val = float(min(recent))
        max_val = float(max(recent))
        rng = max_val - min_val

        if rng == 0:
            # Flat line - all same price
            return "▄" * len(recent)

        # Map each price to block character
        chars = []
        for price in recent:
            normalized = (float(price) - min_val) / rng
            index = int(normalized * (len(blocks) - 1))
            chars.append(blocks[index])

        return "".join(chars)

    def compose(self) -> ComposeResult:
        yield Label("MARKET WATCH", classes="widget-header")
        table = DataTable(id="market-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield Label("", id="market-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Price", "Trend", "Updated")
        self.previous_prices: dict[str, float] = {}

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
    def update_prices(
        self,
        prices: dict[str, Decimal],
        last_update: float | None,
        price_history: dict[str, list[Decimal]] | None = None,
    ) -> None:
        table = self.query_one(DataTable)
        empty_label = self.query_one("#market-empty", Label)

        # Handle empty state with actionable hint
        if not prices:
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_label.display = True
            mode = getattr(self.state, "data_source_mode", None) if self.state is not None else None
            hint = "Press [S] to start bot."
            if mode == "read_only":
                hint = "Press [S] to start data feed."
            empty_label.update(f"No market data yet. {hint}")
            return
        else:
            table.display = True
            empty_label.display = False

        updated_str = "N/A"
        if last_update:
            updated_str = datetime.fromtimestamp(last_update).strftime("%H:%M:%S")

        # Get current row keys for delta updates
        existing_keys = set(table.rows.keys())
        new_keys = set(prices.keys())

        # Remove symbols no longer in the data
        for key in existing_keys - new_keys:
            try:
                table.remove_row(key)
            except Exception:
                pass

        # Get column keys for cell updates
        columns = list(table.columns.keys()) if table.columns else []

        for symbol, price_decimal in prices.items():
            current_price = float(price_decimal)
            prev_price = self.previous_prices.get(symbol, current_price)

            color = THEME.colors.text_primary
            if current_price > prev_price:
                color = THEME.colors.success
            elif current_price < prev_price:
                color = THEME.colors.error

            self.previous_prices[symbol] = current_price

            # Format price with color
            formatted_price = f"[{color}]{format_price(price_decimal)}[/{color}]"

            # Generate sparkline
            sparkline = "─" * 8  # Default fallback
            if price_history and symbol in price_history:
                history = price_history[symbol]
                if history:
                    sparkline = self._generate_sparkline(history)
                    # Color sparkline based on trend direction
                    trend_color = (
                        THEME.colors.success if history[-1] >= history[0] else THEME.colors.error
                    )
                    sparkline = f"[{trend_color}]{sparkline}[/{trend_color}]"

            row_data = (symbol, formatted_price, sparkline, updated_str)

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
                # Add new symbol
                table.add_row(*row_data, key=symbol)
