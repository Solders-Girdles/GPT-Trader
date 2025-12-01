from datetime import datetime
from decimal import Decimal

from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update


class MarketWatchWidget(Static):
    """Displays market data."""

    def compose(self) -> ComposeResult:
        yield Label("MARKET WATCH", classes="header")
        yield DataTable(id="market-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Price", "Updated")
        self.previous_prices: dict[str, float] = {}

    @safe_update
    def update_prices(self, prices: dict[str, str], last_update: float | None) -> None:
        table = self.query_one(DataTable)
        table.clear()

        updated_str = "N/A"
        if last_update:
            updated_str = datetime.fromtimestamp(last_update).strftime("%H:%M:%S")

        for symbol, price_str in prices.items():
            try:
                current_price = float(price_str)
            except ValueError:
                current_price = 0.0

            prev_price = self.previous_prices.get(symbol, current_price)

            color = "white"
            if current_price > prev_price:
                color = "#a3be8c"  # Nord Green
            elif current_price < prev_price:
                color = "#bf616a"  # Nord Red

            self.previous_prices[symbol] = current_price

            # Format price with color
            formatted_price = f"[{color}]{price_str}[/{color}]"
            table.add_row(symbol, formatted_price, updated_str)


class BlockChartWidget(Static):
    """
    Renders a simple block-based chart for price history.
    Uses Unicode block characters:  ▂ ▃ ▄ ▅ ▆ ▇ █
    """

    BLOCKS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    def compose(self) -> ComposeResult:
        yield Label("PRICE CHART (Last 20 Ticks)", classes="header")
        yield Static(id="chart-display", classes="chart-content")

    @safe_update
    def update_chart(self, prices: list[Decimal]) -> None:
        if not prices:
            self.query_one("#chart-display", Static).update("No Data")
            return

        # Convert to floats for calculation
        values = [float(p) for p in prices]
        min_val = min(values)
        max_val = max(values)
        rng = max_val - min_val

        if rng == 0:
            # Flat line
            chart_str = "█" * len(values)
        else:
            # Map values to blocks
            chart_chars = []
            for v in values:
                normalized = (v - min_val) / rng
                index = int(normalized * (len(self.BLOCKS) - 1))
                chart_chars.append(self.BLOCKS[index])
            chart_str = "".join(chart_chars)

        # Add min/max labels and color
        color = "#a3be8c" if values[-1] >= values[0] else "#bf616a"  # Nord Green / Red
        display = f"[{color}]{max_val:.2f}\n{chart_str}\n{min_val:.2f}[/{color}]"
        self.query_one("#chart-display", Static).update(display)
