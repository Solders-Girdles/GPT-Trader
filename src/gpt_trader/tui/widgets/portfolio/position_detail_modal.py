"""Position detail modal for displaying position breakdown and linked trades."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static

from gpt_trader.tui.formatting import (
    format_currency,
    format_percentage,
    format_price,
    format_quantity,
)
from gpt_trader.tui.responsive import calculate_modal_width
from gpt_trader.tui.theme import THEME

if TYPE_CHECKING:
    from gpt_trader.tui.types import Position, Trade


class PositionDetailModal(ModalScreen):
    """Modal displaying detailed position information with linked trades.

    Shows:
    - Position summary (symbol, side, qty, entry, mark, P&L, leverage)
    - Linked trades that contributed to this position
    - Realized P&L from closed portions
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(
        self,
        position: Position,
        trades: list[Trade] | None = None,
    ) -> None:
        """Initialize position detail modal.

        Args:
            position: Position to display details for.
            trades: Optional list of trades to filter for linked trades.
        """
        super().__init__()
        self.position = position
        self.trades = trades or []
        self._linked_trades: list[Trade] = []

    def compose(self) -> ComposeResult:
        """Compose modal layout."""
        pos = self.position

        # Filter trades linked to this position by symbol
        self._linked_trades = [t for t in self.trades if t.symbol == pos.symbol]

        # Sort by time (most recent first)
        self._linked_trades.sort(key=lambda t: t.time, reverse=True)

        # Calculate realized P&L from trades
        realized_pnl = self._calculate_realized_pnl()
        total_fees = self._calculate_total_fees()

        # Side color
        side = pos.side.upper() if pos.side else "LONG"
        side_color = THEME.colors.success if side == "LONG" else THEME.colors.error

        # Calculate P&L percentage
        pnl_pct = Decimal("0")
        if pos.entry_price > 0:
            pnl_pct = ((pos.mark_price - pos.entry_price) / pos.entry_price) * 100

        with Container(id="position-detail-modal"):
            with Vertical():
                # Header
                yield Label(f"Position: {pos.symbol}", id="position-detail-title")

                # Position summary section
                yield Static("─── Position Summary ───", classes="section-header")
                yield Static(f"Symbol: {pos.symbol}")
                yield Static(
                    Text.assemble(
                        "Side: ",
                        Text(side, style=side_color),
                    )
                )
                yield Static(f"Type: {pos.product_type}")
                yield Static(f"Quantity: {format_quantity(pos.quantity)}")

                # Price section
                yield Static("─── Prices ───", classes="section-header")
                yield Static(f"Entry Price: {format_price(pos.entry_price)}")
                yield Static(f"Mark Price: {format_price(pos.mark_price)}")
                if pos.liquidation_price:
                    yield Static(f"Liquidation: {format_price(pos.liquidation_price)}")
                if pos.liquidation_buffer_pct is not None:
                    liq_buffer = pos.liquidation_buffer_pct
                    if liq_buffer < 25:
                        liq_style = "red"
                    elif liq_buffer < 50:
                        liq_style = "yellow"
                    else:
                        liq_style = "green"
                    yield Static(
                        Text.assemble(
                            "Liq Buffer: ",
                            Text(f"{liq_buffer:.1f}%", style=liq_style),
                        )
                    )

                # P&L section
                yield Static("─── Profit & Loss ───", classes="section-header")

                # Unrealized P&L with color
                upnl = pos.unrealized_pnl
                upnl_color = THEME.colors.success if upnl >= 0 else THEME.colors.error
                yield Static(
                    Text.assemble(
                        "Unrealized P&L: ",
                        Text(format_currency(upnl), style=upnl_color),
                        Text(f" ({format_percentage(pnl_pct)})", style="dim"),
                    )
                )

                # Realized P&L (from closed portions)
                if realized_pnl != Decimal("0"):
                    rpnl_color = THEME.colors.success if realized_pnl >= 0 else THEME.colors.error
                    yield Static(
                        Text.assemble(
                            "Realized P&L: ",
                            Text(format_currency(realized_pnl), style=rpnl_color),
                        )
                    )
                else:
                    yield Static("Realized P&L: --")

                # Fees
                yield Static(f"Total Fees: {format_currency(total_fees)}")

                # Recent fills section (quick glance at most recent executions)
                yield Static("─── Recent Fills ───", classes="section-header")
                recent_fills = self._get_recent_fills(limit=3)
                if recent_fills:
                    for fill in recent_fills:
                        yield Static(self._format_recent_fill_row(fill), classes="recent-fills-row")
                    if len(self._linked_trades) > 3:
                        yield Static(
                            f"  … and {len(self._linked_trades) - 3} more (see Linked Trades below)",
                            classes="recent-fills-note",
                        )
                else:
                    yield Static("No recent fills", classes="muted")

                # Fill quality section
                yield Static("─── Fill Quality ───", classes="section-header")
                opens, closes = self._categorize_trades()
                open_quality = self._build_fill_quality_line(opens, "Open fills")
                close_quality = self._build_fill_quality_line(closes, "Close fills")

                if opens or closes:
                    yield Static(open_quality, classes="fill-quality-row")
                    yield Static(close_quality, classes="fill-quality-row")
                else:
                    yield Static("No fill data available", classes="muted")

                # Leverage section (for futures)
                if pos.is_futures:
                    yield Static("─── Leverage ───", classes="section-header")
                    lev_val = pos.leverage or 1
                    if lev_val < 2:
                        lev_style = "green"
                    elif lev_val < 5:
                        lev_style = "yellow"
                    else:
                        lev_style = "red"
                    yield Static(
                        Text.assemble(
                            "Leverage: ",
                            Text(f"{lev_val}x", style=lev_style),
                        )
                    )

                # Linked trades section
                yield Static("─── Linked Trades ───", classes="section-header")
                if self._linked_trades:
                    yield Static(f"{len(self._linked_trades)} trade(s) for this position")
                    table: DataTable[str] = DataTable(
                        id="linked-trades-table",
                        zebra_stripes=True,
                    )
                    table.cursor_type = "row"
                    yield table
                else:
                    yield Static("No trades found for this position", classes="muted")

                # Trade attribution hint
                if self._linked_trades:
                    opens, closes = self._categorize_trades()
                    if opens or closes:
                        yield Static(
                            f"  {len(opens)} opening, {len(closes)} closing",
                            classes="muted",
                        )

                # Close button
                yield Button("Close", variant="primary", id="close-btn")

    def on_mount(self) -> None:
        """Set dynamic width and populate trades table."""
        width = calculate_modal_width(self.app.size.width, "large")
        self.query_one("#position-detail-modal").styles.width = width

        # Populate trades table if we have linked trades
        if self._linked_trades:
            table = self.query_one("#linked-trades-table", DataTable)
            table.add_columns("Time", "Type", "Side", "Qty", "Price", "Fee")

            for trade in self._linked_trades:
                # Determine if this trade opened or closed the position
                trade_type = self._get_trade_type(trade)
                type_text = Text(trade_type, style="cyan" if trade_type == "OPEN" else "magenta")

                # Side color
                side = trade.side.upper()
                side_text = Text(
                    side,
                    style="green" if side == "BUY" else "red",
                )

                # Parse time for display
                time_display = self._format_trade_time(trade.time)

                table.add_row(
                    time_display,
                    type_text,
                    side_text,
                    Text(format_quantity(trade.quantity), justify="right"),
                    Text(format_price(trade.price), justify="right"),
                    Text(format_currency(trade.fee), justify="right"),
                    key=trade.trade_id,
                )

    def _calculate_realized_pnl(self) -> Decimal:
        """Calculate realized P&L from closing trades.

        For LONG positions: realized = (sell_price - entry_price) * sell_qty
        For SHORT positions: realized = (entry_price - buy_price) * buy_qty

        Returns:
            Realized P&L from closing trades.
        """
        # This is a simplified calculation - in production you'd track
        # cost basis per share for accurate FIFO/LIFO accounting
        pos = self.position
        side = pos.side.upper() if pos.side else "LONG"

        realized = Decimal("0")
        entry = pos.entry_price

        for trade in self._linked_trades:
            trade_side = trade.side.upper()

            # Closing trade for LONG = SELL, for SHORT = BUY
            is_closing = (side == "LONG" and trade_side == "SELL") or (
                side == "SHORT" and trade_side == "BUY"
            )

            if is_closing:
                if side == "LONG":
                    # Profit = (sell_price - entry) * qty
                    realized += (trade.price - entry) * trade.quantity
                else:
                    # Profit for short = (entry - buy_price) * qty
                    realized += (entry - trade.price) * trade.quantity

        return realized

    def _calculate_total_fees(self) -> Decimal:
        """Calculate total fees from linked trades."""
        return sum((t.fee for t in self._linked_trades), Decimal("0"))

    def _categorize_trades(self) -> tuple[list[Trade], list[Trade]]:
        """Categorize trades into opening and closing.

        Returns:
            Tuple of (opening_trades, closing_trades).
        """
        pos = self.position
        side = pos.side.upper() if pos.side else "LONG"

        opens = []
        closes = []

        for trade in self._linked_trades:
            trade_side = trade.side.upper()
            if side == "LONG":
                if trade_side == "BUY":
                    opens.append(trade)
                else:
                    closes.append(trade)
            else:  # SHORT
                if trade_side == "SELL":
                    opens.append(trade)
                else:
                    closes.append(trade)

        return opens, closes

    def _get_trade_type(self, trade: Trade) -> str:
        """Determine if a trade opened or closed the position.

        Args:
            trade: Trade to classify.

        Returns:
            "OPEN" or "CLOSE" string.
        """
        pos = self.position
        side = pos.side.upper() if pos.side else "LONG"
        trade_side = trade.side.upper()

        # For LONG: BUY opens, SELL closes
        # For SHORT: SELL opens, BUY closes
        if side == "LONG":
            return "OPEN" if trade_side == "BUY" else "CLOSE"
        else:
            return "OPEN" if trade_side == "SELL" else "CLOSE"

    def _format_trade_time(self, time_str: str) -> str:
        """Format trade time for compact display.

        Args:
            time_str: ISO timestamp string.

        Returns:
            Compact time display (e.g., "12:00:00").
        """
        try:
            if "T" in time_str:
                time_part = time_str.split("T")[1]
                return time_part.split(".")[0].split("Z")[0]
        except (IndexError, ValueError):
            pass
        return time_str[:10]

    def _get_recent_fills(self, limit: int = 3) -> list[Trade]:
        """Get the most recent fills for this position.

        Args:
            limit: Maximum number of fills to return.

        Returns:
            List of most recent trades, sorted by time descending.
        """
        # _linked_trades is already sorted by time (most recent first) in compose()
        return self._linked_trades[:limit]

    def _format_recent_fill_row(self, trade: Trade) -> Text:
        """Format a single recent fill as a compact row.

        Args:
            trade: Trade to format.

        Returns:
            Rich Text object with time, side, type, qty, price, fee.
        """
        time_display = self._format_trade_time(trade.time)
        trade_type = self._get_trade_type(trade)

        # Side color
        side = trade.side.upper()
        side_color = "green" if side == "BUY" else "red"

        # Type color (OPEN=cyan, CLOSE=magenta)
        type_color = "cyan" if trade_type == "OPEN" else "magenta"

        return Text.assemble(
            Text(time_display, style="dim"),
            " ",
            Text(side, style=side_color),
            " ",
            Text(trade_type, style=type_color),
            " ",
            Text(format_quantity(trade.quantity), style="white"),
            " @ ",
            Text(format_price(trade.price), style="white"),
            Text(f" (fee: {format_currency(trade.fee)})", style="dim"),
        )

    def _build_fill_quality_line(self, trades: list[Trade], label: str) -> str:
        """Build a fill quality summary line for a set of trades.

        Computes VWAP, price range percentage, and fee basis points.

        Args:
            trades: List of trades to analyze.
            label: Label for the line (e.g., "Open fills", "Close fills").

        Returns:
            Formatted string like "Open fills (n=2): avg 105.0000 | range 9.52% | fee 9.5 bps"
            or "{label}: --" if no trades or zero notional.
        """
        if not trades:
            return f"{label}: --"

        # Calculate VWAP (Volume-Weighted Average Price)
        total_notional = sum(t.price * t.quantity for t in trades)
        total_quantity = sum(t.quantity for t in trades)

        if total_quantity == 0 or total_notional == 0:
            return f"{label}: --"

        avg_price = total_notional / total_quantity

        # Calculate price range percentage
        prices = [t.price for t in trades]
        min_price = min(prices)
        max_price = max(prices)
        range_pct = ((max_price - min_price) / avg_price) * 100 if avg_price > 0 else Decimal("0")

        # Calculate fee basis points (fee / notional * 10,000)
        total_fees = sum(t.fee for t in trades)
        fee_bps = (total_fees / total_notional) * 10000 if total_notional > 0 else Decimal("0")

        return (
            f"{label} (n={len(trades)}): "
            f"avg {format_price(avg_price)} | "
            f"range {range_pct:.2f}% | "
            f"fee {fee_bps:.1f} bps"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close-btn":
            self.dismiss()

    async def action_dismiss(self) -> None:
        """Dismiss the modal."""
        self.dismiss()
