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

    DEFAULT_CSS = """
    MarketWatchWidget {
        layout: vertical;
        height: 1fr;
    }

    MarketWatchWidget DataTable {
        height: 1fr;
        padding: 1 2;
    }
    """

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
        yield Label("📊 MARKET WATCH", classes="header")
        yield DataTable(id="market-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Price", "Trend", "Updated")
        self.previous_prices: dict[str, float] = {}

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def update_prices(
        self,
        prices: dict[str, Decimal],
        last_update: float | None,
        price_history: dict[str, list[Decimal]] | None = None,
    ) -> None:
        table = self.query_one(DataTable)
        table.clear()

        updated_str = "N/A"
        if last_update:
            updated_str = datetime.fromtimestamp(last_update).strftime("%H:%M:%S")

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

            table.add_row(symbol, formatted_price, sparkline, updated_str)
