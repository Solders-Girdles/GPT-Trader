from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import get_freshness_display, get_staleness_banner
from gpt_trader.tui.types import AccountSummary
from gpt_trader.tui.widgets.tile_states import TileBanner
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class AccountWidget(Static):
    """Widget to display account metrics with optional compact mode.

    Implements StateObserver to receive updates via StateRegistry broadcast.
    """

    # Styles moved to styles/widgets/account.tcss

    def __init__(self, compact_mode: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode
        self._has_received_update = False
        self._bot_running = False
        self._data_source_mode = "demo"

    def compose(self) -> ComposeResult:
        # Header with timestamp
        with Container(classes="account-header"):
            yield Label("ACCOUNT", classes="widget-header")
            yield Label("", id="account-timestamp", classes="timestamp-label")

        # Banner for warnings/errors
        yield TileBanner(id="account-banner", classes="tile-banner hidden")

        if self.compact_mode:
            # Compact horizontal layout - just portfolio value and P&L
            from textual.containers import Horizontal

            with Horizontal(classes="portfolio-summary"):
                yield Label("Portfolio: $0.00", id="portfolio-value", classes="value")
                yield Label("|", classes="metric-separator")
                yield Label("P&L: $0.00", id="total-pnl", classes="value pnl-value")
        else:
            # Full vertical layout (existing implementation)
            # Portfolio Summary Row (Primary Metrics)
            with Container(classes="portfolio-summary-row"):
                with Static(classes="portfolio-metric"):
                    yield Label("Portfolio Value", classes="portfolio-metric-label")
                    yield Label("$0.00", id="portfolio-value", classes="portfolio-metric-value")

                with Static(classes="portfolio-metric"):
                    yield Label("Total P&L", classes="portfolio-metric-label")
                    yield Label("$0.00", id="total-pnl", classes="portfolio-metric-value pnl-value")

            # Account Metrics Row
            with Container(classes="account-metrics-row"):
                with Static(classes="account-metric"):
                    yield Label("Volume 30d", classes="account-metric-label")
                    yield Label("$0.00", id="acc-volume", classes="account-metric-value")

                with Static(classes="account-metric"):
                    yield Label("Fees 30d", classes="account-metric-label")
                    yield Label("$0.00", id="acc-fees", classes="account-metric-value")

                with Static(classes="account-metric"):
                    yield Label("Fee Tier", classes="account-metric-label")
                    yield Label("-", id="acc-tier", classes="account-metric-value")

            # Balance details (shown only in expanded mode)
            with Container(classes="balance-details"):
                yield Label("BALANCES", classes="widget-header")
                yield DataTable()

    def on_mount(self) -> None:
        """Register with StateRegistry and set up expanded mode layouts."""
        # Register with StateRegistry for state updates
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

        # Only set up table and layouts in expanded mode
        if not self.compact_mode:
            try:
                table = self.query_one(DataTable)
                table.add_columns("Asset", "Total", "Available")

                # Add some CSS for the row layouts if not present globally
                self.styles.layout = "vertical"

                # Portfolio summary row (main metrics)
                portfolio_row = self.query_one(".portfolio-summary-row")
                portfolio_row.styles.layout = "horizontal"
                portfolio_row.styles.height = "auto"

                # Account metrics row
                account_row = self.query_one(".account-metrics-row")
                account_row.styles.layout = "horizontal"
                account_row.styles.height = "auto"
            except Exception:
                pass  # Elements don't exist in compact mode

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts account data from TuiState and calls update_account().
        """
        self._has_received_update = True
        self._bot_running = bool(getattr(state, "running", False))
        self._data_source_mode = str(getattr(state, "data_source_mode", "demo") or "demo")

        # Update freshness indicator (using shared helper)
        self._update_freshness_indicator(state)

        # Handle staleness/degraded mode with shared helper
        try:
            banner = self.query_one("#account-banner", TileBanner)
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                banner.update_banner("")
        except Exception as e:
            logger.debug("Failed to update account banner: %s", e)

        if not state.account_data:
            return

        # Toggle metrics row visibility based on data availability
        self._toggle_metrics_row_visibility(state.account_data)

        # Get portfolio value from position_data.equity
        portfolio_value = state.position_data.equity if state.position_data else Decimal("0")

        # Get total P&L from position_data
        total_pnl = state.position_data.total_unrealized_pnl if state.position_data else Decimal("0")

        self.update_account(
            state.account_data,
            portfolio_value=portfolio_value,
            total_pnl=total_pnl,
        )

    def _update_freshness_indicator(self, state: TuiState) -> None:
        """Update freshness indicator with relative time and color coding."""
        try:
            indicator = self.query_one("#account-timestamp", Label)
            freshness = get_freshness_display(state)

            if freshness:
                text, css_class = freshness
                indicator.update(text)
                indicator.remove_class("fresh", "stale", "critical")
                indicator.add_class(css_class)
            else:
                indicator.update("")
                indicator.remove_class("fresh", "stale", "critical")
        except Exception as e:
            logger.debug("Failed to update account freshness: %s", e)

    def _toggle_metrics_row_visibility(self, data: AccountSummary) -> None:
        """Show data or contextual messages for Volume/Fee/Tier.

        Instead of hiding the row when data is unavailable, shows "--" and "N/A"
        placeholders to indicate the row exists but data is not yet available.
        """
        if self.compact_mode:
            return  # Compact mode doesn't show this row

        try:
            metrics_row = self.query_one(".account-metrics-row", Container)
            has_data = (
                data.volume_30d > 0
                or data.fees_30d > 0
                or (data.fee_tier and data.fee_tier not in ("-", "", None))
            )

            if has_data:
                # Show actual values
                self.query_one("#acc-volume", Label).update(format_currency(data.volume_30d))
                self.query_one("#acc-fees", Label).update(format_currency(data.fees_30d))
                tier_str = str(data.fee_tier) if data.fee_tier else "-"
                self.query_one("#acc-tier", Label).update(tier_str)
            else:
                # Show contextual placeholders instead of hiding
                self.query_one("#acc-volume", Label).update("[dim]--[/dim]")
                self.query_one("#acc-fees", Label).update("[dim]--[/dim]")
                self.query_one("#acc-tier", Label).update("[dim]N/A[/dim]")

            # Always show the row now (remove hidden if previously added)
            metrics_row.remove_class("hidden")
        except Exception as e:
            logger.debug("Failed to update metrics row: %s", e)

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def update_account(
        self,
        data: AccountSummary,
        portfolio_value: Decimal = Decimal("0"),
        total_pnl: Decimal = Decimal("0"),
    ) -> None:
        # Update Portfolio Summary (Primary Metrics)
        try:
            portfolio_label = self.query_one("#portfolio-value", Label)
            portfolio_str = format_currency(portfolio_value, decimals=2)

            if self.compact_mode:
                # Compact mode shows labels inline
                portfolio_label.update(f"Portfolio: {portfolio_str}")
            else:
                # Expanded mode shows just values
                portfolio_label.update(portfolio_str)

            # Color-code P&L
            pnl_label = self.query_one("#total-pnl", Label)
            try:
                pnl_str = format_currency(total_pnl, decimals=2)
                if self.compact_mode:
                    # Compact mode with inline label
                    if total_pnl > 0:
                        pnl_label.update(f"P&L: [green]+{pnl_str}[/green]")
                    elif total_pnl < 0:
                        pnl_label.update(f"P&L: [red]{pnl_str}[/red]")
                    else:
                        pnl_label.update(f"P&L: {pnl_str}")
                else:
                    # Expanded mode without inline label
                    if total_pnl > 0:
                        pnl_label.update(f"[green]+{pnl_str}[/green]")
                    elif total_pnl < 0:
                        pnl_label.update(f"[red]{pnl_str}[/red]")
                    else:
                        pnl_label.update(pnl_str)
            except (ValueError, TypeError):
                pnl_str = format_currency(total_pnl, decimals=2)
                if self.compact_mode:
                    pnl_label.update(f"P&L: {pnl_str}")
                else:
                    pnl_label.update(pnl_str)
        except Exception as e:
            logger.error(f"Failed to update portfolio summary: {e}", exc_info=True)

        # Update Account Summary (only in expanded mode)
        if not self.compact_mode:
            try:
                self.query_one("#acc-volume", Label).update(format_currency(data.volume_30d))
                self.query_one("#acc-fees", Label).update(format_currency(data.fees_30d))
                self.query_one("#acc-tier", Label).update(f"{data.fee_tier}")
            except Exception as e:
                logger.error(f"Failed to update account metrics: {e}", exc_info=True)

            # Update Balances (only in expanded mode)
            try:
                table = self.query_one(DataTable)
                table.clear()

                for bal in data.balances:
                    table.add_row(
                        bal.asset, format_currency(bal.total), format_currency(bal.available)
                    )
            except Exception as e:
                logger.error(f"Failed to update balances: {e}", exc_info=True)
