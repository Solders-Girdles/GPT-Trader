"""Order detail modal for displaying order fill history and linked trades."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static

from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.responsive import calculate_modal_width
from gpt_trader.tui.staleness_helpers import format_freshness_label
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.thresholds import get_confidence_status, get_status_color
from gpt_trader.tui.utilities import get_age_seconds

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState
    from gpt_trader.tui.types import DecisionData, Order, Trade


class OrderDetailModal(ModalScreen):
    """Modal displaying detailed order information with linked trades.

    Shows:
    - Originating decision (action, confidence, reason) when linked
    - Order summary (symbol, side, qty, price, status, type)
    - Fill progress (filled_qty / total_qty, avg fill price)
    - Linked trades table (matched by order_id)
    - Total fees from linked trades
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(
        self,
        order: Order,
        trades: list[Trade] | None = None,
        tui_state: TuiState | None = None,
    ) -> None:
        """Initialize order detail modal.

        Args:
            order: Order to display details for.
            trades: Optional list of trades to filter for linked trades.
            tui_state: Optional TuiState for decision lookup by decision_id.
        """
        super().__init__()
        self.order = order
        self.trades = trades or []
        self.tui_state = tui_state
        self._linked_trades: list[Trade] = []
        self._linked_decision: DecisionData | None = None

    def compose(self) -> ComposeResult:
        """Compose modal layout."""
        order = self.order

        # Filter trades linked to this order
        self._linked_trades = [t for t in self.trades if t.order_id == order.order_id]

        # Look up linked decision by decision_id
        if self.tui_state and order.decision_id:
            self._linked_decision = self.tui_state.get_decision_by_id(order.decision_id)

        # Calculate fill stats
        fill_pct = self._calculate_fill_pct()
        total_fees = self._calculate_total_fees()

        # Side color
        side_color = THEME.colors.success if order.side == "BUY" else THEME.colors.error

        # Age display
        age = get_age_seconds(order.creation_time)
        age_display = format_freshness_label(age) if age is not None else "--"

        with Container(id="order-detail-modal"):
            with Vertical():
                # Header
                yield Label(f"Order: {order.symbol}", id="order-detail-title")

                # Decision section (when linked)
                if self._linked_decision:
                    yield from self._compose_decision_section(self._linked_decision)

                # Order summary section
                yield Static("─── Order Summary ───", classes="section-header")
                yield Static(f"Symbol: {order.symbol}")
                yield Static(
                    Text.assemble(
                        "Side: ",
                        Text(order.side, style=side_color),
                    )
                )
                yield Static(f"Type: {order.type}")
                yield Static(f"Time in Force: {order.time_in_force}")
                yield Static(f"Status: {order.status}")
                yield Static(f"Age: {age_display}")

                # Quantity and price section
                yield Static("─── Quantity & Price ───", classes="section-header")
                yield Static(f"Order Qty: {format_quantity(order.quantity)}")
                yield Static(f"Limit Price: {format_price(order.price)}")
                yield Static(f"Filled: {format_quantity(order.filled_quantity)} ({fill_pct})")
                if order.avg_fill_price:
                    yield Static(f"Avg Fill Price: {format_price(order.avg_fill_price)}")
                else:
                    yield Static("Avg Fill Price: --")

                # Fees section
                yield Static(f"Total Fees: {format_price(total_fees)}")

                # Linked trades section
                yield Static("─── Linked Trades ───", classes="section-header")
                if self._linked_trades:
                    yield Static(f"{len(self._linked_trades)} trade(s) linked to this order")
                    table: DataTable[str] = DataTable(
                        id="linked-trades-table",
                        zebra_stripes=True,
                    )
                    table.cursor_type = "row"
                    yield table
                else:
                    yield Static("No trades linked to this order", classes="muted")

                # Close button
                yield Button("Close", variant="primary", id="close-btn")

    def on_mount(self) -> None:
        """Set dynamic width and populate trades table."""
        width = calculate_modal_width(self.app.size.width, "large")
        self.query_one("#order-detail-modal").styles.width = width

        # Populate trades table if we have linked trades
        if self._linked_trades:
            table = self.query_one("#linked-trades-table", DataTable)
            table.add_columns("Time", "Qty", "Price", "Fee")

            for trade in self._linked_trades:
                # Parse time for display (extract time portion)
                time_display = self._format_trade_time(trade.time)
                table.add_row(
                    time_display,
                    Text(format_quantity(trade.quantity), justify="right"),
                    Text(format_price(trade.price), justify="right"),
                    Text(format_price(trade.fee), justify="right"),
                    key=trade.trade_id,
                )

    def _calculate_fill_pct(self) -> str:
        """Calculate fill percentage string."""
        order = self.order
        if order.quantity <= 0:
            return "0%"
        pct = (float(order.filled_quantity) / float(order.quantity)) * 100
        return f"{pct:.0f}%"

    def _calculate_total_fees(self) -> Decimal:
        """Calculate total fees from linked trades."""
        return sum((t.fee for t in self._linked_trades), Decimal("0"))

    def _format_trade_time(self, time_str: str) -> str:
        """Format trade time for compact display.

        Args:
            time_str: ISO timestamp string (e.g., "2023-01-01T12:00:00.000Z").

        Returns:
            Compact time display (e.g., "12:00:00").
        """
        try:
            # Extract time portion from ISO format
            if "T" in time_str:
                time_part = time_str.split("T")[1]
                # Remove timezone and milliseconds for compact display
                return time_part.split(".")[0].split("Z")[0]
        except (IndexError, ValueError):
            pass
        return time_str[:10]  # Fallback to first 10 chars

    def _compose_decision_section(self, decision: DecisionData) -> ComposeResult:
        """Compose the decision section showing strategy decision that triggered the order.

        Args:
            decision: The linked DecisionData.

        Yields:
            Widgets for the decision section.
        """
        yield Static("─── Strategy Decision ───", classes="section-header")

        # Action with color
        action = decision.action.upper()
        if action == "BUY":
            action_text = Text(action, style="green bold")
        elif action == "SELL":
            action_text = Text(action, style="red bold")
        else:
            action_text = Text(action, style="yellow bold")

        yield Static(Text.assemble("Action: ", action_text))

        # Confidence with status color
        conf_status = get_confidence_status(decision.confidence)
        conf_color = get_status_color(conf_status)
        conf_text = Text(f"{decision.confidence:.0%}", style=conf_color)
        yield Static(Text.assemble("Confidence: ", conf_text))

        # Reason (truncate if too long)
        reason = decision.reason
        if len(reason) > 60:
            reason = reason[:57] + "..."
        yield Static(f"Reason: {reason}")

        # Blocked by guard (if decision was blocked)
        if decision.blocked_by:
            yield Static(
                Text.assemble(
                    "Blocked By: ",
                    Text(decision.blocked_by, style="red bold"),
                )
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close-btn":
            self.dismiss()

    async def action_dismiss(self) -> None:
        """Dismiss the modal."""
        self.dismiss()
