"""
CFM Balance Widget for displaying Coinbase Financial Markets futures balance.

Displays margin, buying power, and liquidation metrics for CFM futures trading.

Implements StateObserver to receive updates via StateRegistry broadcast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.core.account import CFMBalance
from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.helpers import safe_update
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class CFMBalanceWidget(Static):
    """Widget to display CFM futures balance and margin metrics.

    Implements StateObserver to receive updates via StateRegistry broadcast.

    Shows key metrics for futures trading:
    - Total balance and buying power
    - Margin utilization
    - Liquidation buffer with color-coded warnings
    - Unrealized P&L
    """

    # Styles moved to styles/widgets/cfm_balance.tcss

    # Reactive property for CFM availability
    has_cfm = reactive(False)

    def on_mount(self) -> None:
        """Register with StateRegistry on mount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts CFM balance from TuiState.cfm_balance and calls update_balance().
        """
        # Update based on CFM access and balance data
        if state.has_cfm_access and state.cfm_balance:
            self.update_balance(state.cfm_balance)
        else:
            self.update_balance(None)

    def compose(self) -> ComposeResult:
        """Compose the CFM balance display."""
        yield Label("CFM FUTURES", classes="cfm-header")

        # Row 1: Balance and Buying Power
        with Horizontal(classes="cfm-row"):
            with Static(classes="cfm-metric"):
                yield Label("Balance", classes="cfm-label")
                yield Label("$0.00", id="cfm-balance", classes="cfm-value")
            with Static(classes="cfm-metric"):
                yield Label("Buying Power", classes="cfm-label")
                yield Label("$0.00", id="cfm-buying-power", classes="cfm-value")

        # Row 2: Margin Info
        with Horizontal(classes="cfm-row"):
            with Static(classes="cfm-metric"):
                yield Label("Avail. Margin", classes="cfm-label")
                yield Label("$0.00", id="cfm-avail-margin", classes="cfm-value")
            with Static(classes="cfm-metric"):
                yield Label("Margin Used", classes="cfm-label")
                yield Label("0%", id="cfm-margin-used", classes="cfm-value")

        # Row 3: P&L and Liquidation
        with Horizontal(classes="cfm-row"):
            with Static(classes="cfm-metric"):
                yield Label("Unrealized P&L", classes="cfm-label")
                yield Label("$0.00", id="cfm-pnl", classes="cfm-value")
            with Static(classes="cfm-metric"):
                yield Label("Liq. Buffer", classes="cfm-label")
                yield Label("—", id="cfm-liq-buffer", classes="cfm-value")

        # No CFM access message (hidden by default)
        yield Label(
            "[dim]CFM futures not available[/dim]",
            id="cfm-unavailable",
            classes="cfm-hidden",
        )

    def watch_has_cfm(self, has_cfm: bool) -> None:
        """React to CFM availability changes."""
        try:
            unavailable_label = self.query_one("#cfm-unavailable", Label)
            rows = self.query(".cfm-row")

            if has_cfm:
                unavailable_label.add_class("cfm-hidden")
                for row in rows:
                    row.remove_class("cfm-hidden")
            else:
                unavailable_label.remove_class("cfm-hidden")
                for row in rows:
                    row.add_class("cfm-hidden")
        except Exception:
            pass  # Widget not mounted yet

    @safe_update(notify_user=False, error_tracker=True, severity="warning")
    def update_balance(self, cfm_balance: CFMBalance | None) -> None:
        """Update the CFM balance display.

        Args:
            cfm_balance: CFMBalance object or None if not available.
        """
        if cfm_balance is None:
            self.has_cfm = False
            return

        self.has_cfm = True

        # Update balance
        balance_label = self.query_one("#cfm-balance", Label)
        balance_label.update(format_currency(cfm_balance.total_usd_balance, decimals=2))

        # Update buying power
        bp_label = self.query_one("#cfm-buying-power", Label)
        bp_label.update(format_currency(cfm_balance.futures_buying_power, decimals=2))

        # Update available margin
        margin_label = self.query_one("#cfm-avail-margin", Label)
        margin_label.update(format_currency(cfm_balance.available_margin, decimals=2))

        # Update margin utilization
        margin_used_label = self.query_one("#cfm-margin-used", Label)
        margin_pct = cfm_balance.margin_utilization_pct
        if margin_pct > 80:
            margin_used_label.update(f"[red]{margin_pct:.1f}%[/red]")
        elif margin_pct > 50:
            margin_used_label.update(f"[yellow]{margin_pct:.1f}%[/yellow]")
        else:
            margin_used_label.update(f"[green]{margin_pct:.1f}%[/green]")

        # Update unrealized P&L with color coding
        pnl_label = self.query_one("#cfm-pnl", Label)
        pnl = cfm_balance.unrealized_pnl
        pnl_str = format_currency(pnl, decimals=2)
        if pnl > 0:
            pnl_label.update(f"[green]+{pnl_str}[/green]")
        elif pnl < 0:
            pnl_label.update(f"[red]{pnl_str}[/red]")
        else:
            pnl_label.update(pnl_str)

        # Update liquidation buffer with color coding
        liq_label = self.query_one("#cfm-liq-buffer", Label)
        liq_pct = cfm_balance.liquidation_buffer_percentage
        if liq_pct < 25:
            liq_label.update(f"[red bold]{liq_pct:.0f}%[/red bold]")
        elif liq_pct < 50:
            liq_label.update(f"[yellow]{liq_pct:.0f}%[/yellow]")
        elif liq_pct < 100:
            liq_label.update(f"[green]{liq_pct:.0f}%[/green]")
        else:
            liq_label.update(f"[green bold]{liq_pct:.0f}%[/green bold]")

    def show_unavailable(self, message: str = "CFM futures not available") -> None:
        """Show the unavailable message."""
        self.has_cfm = False
        try:
            unavailable_label = self.query_one("#cfm-unavailable", Label)
            unavailable_label.update(f"[dim]{message}[/dim]")
        except Exception:
            pass
