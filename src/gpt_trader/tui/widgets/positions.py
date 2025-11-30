from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static


class PositionsWidget(Static):
    """Displays active positions."""

    def compose(self) -> ComposeResult:
        yield Label("ACTIVE POSITIONS", classes="header")
        yield DataTable(id="positions-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Sym", "Qty", "Entry", "PnL")

    def update_positions(self, positions: dict[str, Any], total_pnl: str) -> None:
        table = self.query_one(DataTable)
        table.clear()

        # Determine structure of positions dict
        # StatusReporter uses: {symbol: {quantity, mark, ...}} or similar
        # But let's handle the generic dict passed from StatusReporter

        # If positions is empty, show nothing
        if not positions:
            return

        # Iterate through positions
        # Note: The structure depends on what TradingEngine passes to StatusReporter
        # In TradingEngine._positions_to_risk_format: {symbol: {quantity, mark}}
        # But StatusReporter might have different structure if it aggregates PnL

        # For now, we iterate assuming keys are symbols
        for symbol, data in positions.items():
            # Handle both object and dict (just in case)
            if hasattr(data, "quantity"):
                quantity = str(data.quantity)
                entry = str(getattr(data, "entry_price", "N/A"))
                pnl = str(getattr(data, "unrealized_pnl", "0"))
            elif isinstance(data, dict):
                quantity = str(data.get("quantity", "0"))
                entry = str(data.get("entry_price", "N/A"))
                pnl = str(data.get("unrealized_pnl", "0"))
            else:
                continue

            table.add_row(symbol, quantity, entry, pnl)


class OrdersWidget(Static):
    """Widget to display active orders."""

    def compose(self) -> ComposeResult:
        yield Label("Active Orders", classes="header")
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Side", "Type", "Quantity", "Price", "TIF", "Status", "Time")

    def update_orders(self, orders: list[Any]) -> None:
        table = self.query_one(DataTable)
        table.clear()

        for order in orders:
            # Symbol
            symbol = order.get("symbol", "")

            # Side
            side = order.get("side", "")

            # Quantity
            quantity = order.get("quantity", "")
            if not quantity:
                # Try configuration
                config = order.get("order_configuration", {})
                if "market_market_ioc" in config:
                    quantity = config["market_market_ioc"].get("base_size", "")
                elif "limit_limit_gtc" in config:
                    quantity = config["limit_limit_gtc"].get("base_size", "")

            # Fallback
            if not quantity:
                quantity = getattr(order, "quantity", "")

            # Price
            price = order.get("avg_execution_price", "")
            if not price:
                # Try limit price
                config = order.get("order_configuration", {})
                if "limit_limit_gtc" in config:
                    price = config["limit_limit_gtc"].get("limit_price", "")

            # Status
            status = order.get("status", "UNKNOWN")

            # Time
            time_str = order.get("creation_time", "")
            if time_str:
                try:
                    # Simplify time string
                    time_str = time_str.split("T")[1].split(".")[0]
                except IndexError:
                    pass

            # Type & TIF
            order_type = "MARKET"
            tif = "IOC"
            config = order.get("order_configuration", {})
            if "limit_limit_gtc" in config:
                order_type = "LIMIT"
                tif = "GTC"
            elif "stop_limit_stop_limit_gtc" in config:
                order_type = "STOP_LIMIT"
                tif = "GTC"
            elif "stop_limit_stop_limit_gtd" in config:
                order_type = "STOP_LIMIT"
                tif = "GTD"

            # Colorize Side
            side_style = "green" if side == "BUY" else "red"
            formatted_side = f"[{side_style}]{side}[/{side_style}]"

            table.add_row(
                symbol, formatted_side, order_type, quantity, str(price), tif, status, time_str
            )


class TradesWidget(Static):
    """Widget to display recent trades."""

    def compose(self) -> ComposeResult:
        yield Label("Recent Trades", classes="header")
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Side", "Qty", "Price", "Order ID", "Time")

    def update_trades(self, trades: list[Any]) -> None:
        table = self.query_one(DataTable)
        table.clear()

        for trade in trades:
            symbol = trade.get("product_id", "UNKNOWN")
            side = trade.get("side", "UNKNOWN")
            quantity = trade.get("quantity", "")
            price = trade.get("price", "")
            time_str = trade.get("time", "")
            order_id = trade.get("order_id", "")

            # Fallback
            if not quantity:
                quantity = getattr(trade, "quantity", "")

            if time_str:
                try:
                    time_str = time_str.split("T")[1].split(".")[0]
                except IndexError:
                    pass

            # Colorize Side
            side_style = "green" if side == "BUY" else "red"
            formatted_side = f"[{side_style}]{side}[/{side_style}]"

            table.add_row(symbol, formatted_side, quantity, price, order_id, time_str)
