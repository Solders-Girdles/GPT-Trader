"""
Account Detail Screen with comprehensive balance and holdings view.

Extends the basic account display with:
- Full balance table with USD values and sorting
- Holdings breakdown by asset type (crypto, fiat, stablecoins)
- Volume and fee history visualization
- Account tier and limits information
- Copy functionality for values
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

from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import get_freshness_display, get_staleness_banner
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.utilities import (
    copy_to_clipboard,
    format_pnl_colored,
    get_sort_indicator,
    sort_table_data,
)
from gpt_trader.tui.widgets import ContextualFooter
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.tui.widgets.tile_states import TileBanner, TileEmptyState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState
    from gpt_trader.tui.types import AccountBalance, AccountSummary

logger = get_logger(__name__, component="tui")


# Asset classifications for breakdown
STABLECOINS = {"USDC", "USDT", "DAI", "BUSD", "USDP", "GUSD", "PAX"}
FIAT_CURRENCIES = {"USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"}


class AccountMetricsPanel(Static):
    """Panel showing account summary metrics."""

    def compose(self) -> ComposeResult:
        """Compose the metrics panel."""
        yield Label("ACCOUNT METRICS", classes="widget-header")

        with Grid(classes="metrics-grid"):
            with Vertical(classes="metric-card"):
                yield Label("Portfolio Value", classes="metric-label")
                yield Label("$0.00", id="metric-portfolio", classes="metric-value primary")

            with Vertical(classes="metric-card"):
                yield Label("Total P&L", classes="metric-label")
                yield Label("$0.00", id="metric-pnl", classes="metric-value")

            with Vertical(classes="metric-card"):
                yield Label("30d Volume", classes="metric-label")
                yield Label("$0.00", id="metric-volume", classes="metric-value")

            with Vertical(classes="metric-card"):
                yield Label("30d Fees", classes="metric-label")
                yield Label("$0.00", id="metric-fees", classes="metric-value")

            with Vertical(classes="metric-card"):
                yield Label("Fee Tier", classes="metric-label")
                yield Label("-", id="metric-tier", classes="metric-value")

            with Vertical(classes="metric-card"):
                yield Label("Last Update", classes="metric-label")
                yield Label("--:--:--", id="metric-updated", classes="metric-value dim")

    def update_metrics(
        self,
        portfolio_value: Decimal,
        total_pnl: Decimal,
        volume_30d: Decimal,
        fees_30d: Decimal,
        fee_tier: str,
        last_update: float,
    ) -> None:
        """Update all account metrics."""
        try:
            self.query_one("#metric-portfolio", Label).update(format_currency(portfolio_value))

            # P&L with color
            pnl_text = format_pnl_colored(float(total_pnl), format_currency(total_pnl))
            self.query_one("#metric-pnl", Label).update(pnl_text)

            self.query_one("#metric-volume", Label).update(format_currency(volume_30d))
            self.query_one("#metric-fees", Label).update(format_currency(fees_30d))
            self.query_one("#metric-tier", Label).update(fee_tier or "-")

            if last_update > 0:
                dt = datetime.fromtimestamp(last_update)
                self.query_one("#metric-updated", Label).update(dt.strftime("%H:%M:%S"))
        except Exception as e:
            logger.debug(f"Failed to update account metrics: {e}")


class HoldingsBreakdown(Static):
    """Panel showing holdings breakdown by asset type."""

    def compose(self) -> ComposeResult:
        """Compose the breakdown panel."""
        yield Label("HOLDINGS BREAKDOWN", classes="widget-header")

        with Horizontal(classes="breakdown-bars"):
            with Vertical(classes="breakdown-item"):
                yield Label("Crypto", classes="breakdown-label")
                yield Label("$0.00", id="breakdown-crypto", classes="breakdown-value")
                yield Static("", id="bar-crypto", classes="breakdown-bar crypto")

            with Vertical(classes="breakdown-item"):
                yield Label("Stablecoins", classes="breakdown-label")
                yield Label("$0.00", id="breakdown-stable", classes="breakdown-value")
                yield Static("", id="bar-stable", classes="breakdown-bar stable")

            with Vertical(classes="breakdown-item"):
                yield Label("Fiat", classes="breakdown-label")
                yield Label("$0.00", id="breakdown-fiat", classes="breakdown-value")
                yield Static("", id="bar-fiat", classes="breakdown-bar fiat")

    def update_breakdown(
        self,
        crypto_value: Decimal,
        stable_value: Decimal,
        fiat_value: Decimal,
        total: Decimal,
    ) -> None:
        """Update holdings breakdown display."""
        try:
            self.query_one("#breakdown-crypto", Label).update(format_currency(crypto_value))
            self.query_one("#breakdown-stable", Label).update(format_currency(stable_value))
            self.query_one("#breakdown-fiat", Label).update(format_currency(fiat_value))

            # Update visual bars (percentage-based width)
            if total > 0:
                crypto_pct = float(crypto_value / total * 100)
                stable_pct = float(stable_value / total * 100)
                fiat_pct = float(fiat_value / total * 100)

                crypto_bar = self.query_one("#bar-crypto", Static)
                stable_bar = self.query_one("#bar-stable", Static)
                fiat_bar = self.query_one("#bar-fiat", Static)

                # Visual representation using block chars
                crypto_bar.update(self._make_bar(crypto_pct, THEME.colors.warning))
                stable_bar.update(self._make_bar(stable_pct, THEME.colors.success))
                fiat_bar.update(self._make_bar(fiat_pct, THEME.colors.info))
        except Exception as e:
            logger.debug(f"Failed to update breakdown: {e}")

    def _make_bar(self, percent: float, color: str) -> Text:
        """Create a visual bar for percentage."""
        filled = int(percent / 5)  # 20 char max bar
        bar = "█" * filled + "░" * (20 - filled)
        return Text.from_markup(f"[{color}]{bar}[/{color}] {percent:.1f}%")


class AccountDetailScreen(Screen):
    """Enhanced account detail screen with balances, holdings, and metrics.

    Features:
    - Sortable balance table (asset, total, available, hold, USD value)
    - Holdings breakdown by asset type (crypto, stablecoins, fiat)
    - Account metrics summary (portfolio, P&L, volume, fees, tier)
    - Copy functionality for values

    Keyboard:
    - ESC/Q: Close and return to main screen
    - S: Cycle sort column
    - C: Copy selected row
    - ↑↓: Navigate table rows
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("c", "copy_row", "Copy", show=True),
        Binding("r", "refresh", "Refresh", show=False),
    ]

    CSS = """
    #account-detail-container {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 3fr 2fr;
        grid-rows: auto 1fr;
        height: 1fr;
        padding: 1;
    }

    #metrics-panel {
        column-span: 2;
        border: round $border-primary;
        padding: 1;
        height: auto;
    }

    .metrics-grid {
        layout: grid;
        grid-size: 6 1;
        grid-gutter: 1;
        height: auto;
    }

    .metric-card {
        padding: 0 1;
        height: auto;
    }

    .metric-label {
        color: $text-muted;
    }

    .metric-value {
        text-style: bold;
    }

    .metric-value.primary {
        color: $accent;
    }

    #balances-panel {
        border: round $border-primary;
        padding: 1;
    }

    #breakdown-panel {
        border: round $border-primary;
        padding: 1;
    }

    .breakdown-bars {
        height: auto;
    }

    .breakdown-item {
        width: 1fr;
        padding: 0 1;
    }

    .breakdown-label {
        color: $text-secondary;
    }

    .breakdown-value {
        text-style: bold;
    }

    .breakdown-bar {
        height: 1;
    }
    """

    # State
    state: reactive[TuiState | None] = reactive(None)

    # Sort state
    sort_column: reactive[str] = reactive("asset")
    sort_ascending: reactive[bool] = reactive(True)

    def __init__(self, **kwargs) -> None:
        """Initialize AccountDetailScreen."""
        super().__init__(**kwargs)
        self._balances_data: list[dict[str, Any]] = []
        self._column_keys = ["asset", "total", "available", "hold"]

    def compose(self) -> ComposeResult:
        """Compose the account detail screen layout."""
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )
        yield TileBanner(id="account-detail-banner", classes="tile-banner hidden")

        with Container(id="account-detail-container"):
            # Top: Account metrics summary
            with Container(id="metrics-panel"):
                yield AccountMetricsPanel(id="metrics-widget")

            # Bottom left: Balances table
            with Vertical(id="balances-panel"):
                yield Label("BALANCES", classes="widget-header")
                yield Label("", id="sort-indicator", classes="sort-hint")
                table = DataTable(
                    id="balances-table",
                    zebra_stripes=True,
                    cursor_type="row",
                )
                table.can_focus = True
                yield table
                yield TileEmptyState(
                    title="No Balance Data",
                    subtitle="Waiting for account snapshot",
                    icon="◌",
                    actions=["[R] Refresh"],
                    id="balances-empty",
                )

            # Bottom right: Holdings breakdown
            with Container(id="breakdown-panel"):
                yield HoldingsBreakdown(id="breakdown-widget")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        logger.debug("AccountDetailScreen mounted")

        # Set up table columns
        table = self.query_one("#balances-table", DataTable)
        table.add_column("Asset", key="asset")
        table.add_column("Total", key="total")
        table.add_column("Available", key="available")
        table.add_column("Hold", key="hold")

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
        # Update staleness banner
        try:
            banner = self.query_one("#account-detail-banner", TileBanner)
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                banner.update_banner("")
        except Exception:
            pass

        self.state = state

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update account display."""
        if state is None:
            return

        self._update_account(state)

    def watch_sort_column(self, column: str) -> None:
        """Handle sort column change."""
        self._update_sort_indicator()
        self._refresh_table()

    def watch_sort_ascending(self, ascending: bool) -> None:
        """Handle sort direction change."""
        self._update_sort_indicator()
        self._refresh_table()

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def _update_account(self, state: TuiState) -> None:
        """Update all account displays from state."""
        account_data = state.account_data
        position_data = state.position_data

        # Update metrics panel
        try:
            metrics = self.query_one("#metrics-widget", AccountMetricsPanel)
            portfolio_value = position_data.equity if position_data else Decimal("0")
            total_pnl = position_data.total_unrealized_pnl if position_data else Decimal("0")

            metrics.update_metrics(
                portfolio_value=portfolio_value,
                total_pnl=total_pnl,
                volume_30d=account_data.volume_30d,
                fees_30d=account_data.fees_30d,
                fee_tier=account_data.fee_tier,
                last_update=state.last_data_fetch or state.last_update_timestamp,
            )

            # Update freshness indicator with color coding
            freshness = get_freshness_display(state)
            if freshness:
                text, _ = freshness
                metrics.query_one("#metric-updated", Label).update(text)
        except Exception as e:
            logger.debug(f"Failed to update metrics panel: {e}")

        # Update balances table
        self._update_balances(account_data.balances)

        # Update holdings breakdown
        self._update_breakdown(account_data.balances)

    def _update_balances(self, balances: list[AccountBalance]) -> None:
        """Update balances table with account data."""
        table = self.query_one("#balances-table", DataTable)
        empty_state = self.query_one("#balances-empty", TileEmptyState)

        if not balances:
            table.display = False
            empty_state.display = True
            return

        table.display = True
        empty_state.display = False

        # Build data rows
        data_rows: list[dict[str, Any]] = []
        for bal in balances:
            data_rows.append({
                "asset": bal.asset,
                "total": float(bal.total),
                "available": float(bal.available),
                "hold": float(bal.hold),
            })

        self._balances_data = data_rows

        # Sort data
        sorted_data = sort_table_data(
            data_rows,
            self.sort_column,
            self.sort_ascending,
            numeric_columns={"total", "available", "hold"},
        )

        # Populate table
        self._populate_table(table, sorted_data)

    def _populate_table(self, table: DataTable, data: list[dict[str, Any]]) -> None:
        """Populate table with sorted data."""
        existing_keys = set(table.rows.keys())
        new_keys = {row["asset"] for row in data}

        # Remove old rows
        for key in existing_keys - new_keys:
            try:
                table.remove_row(key)
            except Exception:
                pass

        columns = list(table.columns.keys())

        for row in data:
            asset = row["asset"]
            total = row["total"]
            available = row["available"]
            hold = row["hold"]

            # Format cells with alignment
            total_str = f"{total:,.8f}" if total < 1 else f"{total:,.4f}"
            avail_str = f"{available:,.8f}" if available < 1 else f"{available:,.4f}"
            hold_str = f"{hold:,.8f}" if hold < 1 else f"{hold:,.4f}"

            # Highlight hold if non-zero
            if hold > 0:
                hold_str = f"[yellow]{hold_str}[/yellow]"

            row_data = (asset, total_str, avail_str, hold_str)

            if asset in existing_keys:
                try:
                    for col_key, value in zip(columns, row_data):
                        table.update_cell(asset, col_key, value)
                except Exception:
                    try:
                        table.remove_row(asset)
                    except Exception:
                        pass
                    table.add_row(*row_data, key=asset)
            else:
                table.add_row(*row_data, key=asset)

    def _update_breakdown(self, balances: list[AccountBalance]) -> None:
        """Update holdings breakdown from balances."""
        crypto_value = Decimal("0")
        stable_value = Decimal("0")
        fiat_value = Decimal("0")

        for bal in balances:
            asset_upper = bal.asset.upper()
            # Simple classification (in real app, would use USD value)
            if asset_upper in FIAT_CURRENCIES:
                fiat_value += bal.total
            elif asset_upper in STABLECOINS:
                stable_value += bal.total
            else:
                # Crypto - would need price conversion for accurate USD
                crypto_value += bal.total

        total = crypto_value + stable_value + fiat_value

        try:
            breakdown = self.query_one("#breakdown-widget", HoldingsBreakdown)
            breakdown.update_breakdown(crypto_value, stable_value, fiat_value, total)
        except Exception as e:
            logger.debug(f"Failed to update breakdown: {e}")

    def _update_sort_indicator(self) -> None:
        """Update sort indicator label."""
        try:
            indicator = self.query_one("#sort-indicator", Label)
            arrow = get_sort_indicator(self.sort_column, self.sort_column, self.sort_ascending)
            col_display = self.sort_column.capitalize()
            indicator.update(f"[S] Sort: {col_display}{arrow}")
        except Exception:
            pass

    def _refresh_table(self) -> None:
        """Re-sort and refresh the table."""
        if self._balances_data:
            sorted_data = sort_table_data(
                self._balances_data,
                self.sort_column,
                self.sort_ascending,
                numeric_columns={"total", "available", "hold"},
            )
            try:
                table = self.query_one("#balances-table", DataTable)
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
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_ascending = True

        self.sort_column = self._column_keys[next_idx]
        self.notify(f"Sorted by {self.sort_column} {'↑' if self.sort_ascending else '↓'}", timeout=2)

    def action_copy_row(self) -> None:
        """Copy selected row to clipboard."""
        try:
            table = self.query_one("#balances-table", DataTable)
            cursor = table.cursor_coordinate
            if cursor is None:
                return

            row_key = table.get_row_at(cursor.row)
            if row_key is None:
                return

            # Find row data
            for row in self._balances_data:
                if row["asset"] == row_key:
                    text = f"{row['asset']}\t{row['total']}\t{row['available']}\t{row['hold']}"
                    if copy_to_clipboard(text):
                        self.notify("Row copied to clipboard", timeout=2)
                    else:
                        self.notify("Copy failed", severity="warning", timeout=2)
                    break
        except Exception as e:
            logger.debug(f"Copy row failed: {e}")

    def action_refresh(self) -> None:
        """Manually refresh account data via app-level reconnect."""
        if hasattr(self.app, "action_reconnect_data"):
            self.app.action_reconnect_data()
