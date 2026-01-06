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
        # Display signature cache for early-exit optimization
        self._last_display_signature: tuple | None = None

    def compose(self) -> ComposeResult:
        # Header with timestamp
        with Container(classes="account-header"):
            yield Label("ACCOUNT", classes="widget-header")
            yield Label("", id="account-timestamp", classes="timestamp-label")

        # Banner for warnings/errors
        yield TileBanner(id="account-banner", classes="tile-banner hidden")

        if self.compact_mode:
            # Compact horizontal layout - portfolio value, net P&L, and breakdown hint
            from textual.containers import Horizontal

            with Horizontal(classes="portfolio-summary"):
                yield Label("Portfolio: $0.00", id="portfolio-value", classes="value")
                yield Label("|", classes="metric-separator")
                yield Label("Net: $0.00", id="net-pnl", classes="value pnl-value")
                yield Label("", id="pnl-breakdown", classes="value pnl-breakdown")
        else:
            # Full vertical layout (existing implementation)
            # Portfolio Summary Row (Primary Metrics)
            with Container(classes="portfolio-summary-row"):
                with Static(classes="portfolio-metric"):
                    yield Label("Portfolio Value", classes="portfolio-metric-label")
                    yield Label("$0.00", id="portfolio-value", classes="portfolio-metric-value")

                with Static(classes="portfolio-metric"):
                    yield Label("Net P&L", classes="portfolio-metric-label")
                    yield Label("$0.00", id="net-pnl", classes="portfolio-metric-value pnl-value")

            # P&L Breakdown Row
            with Container(classes="pnl-breakdown-row"):
                with Static(classes="pnl-metric"):
                    yield Label("Unrealized", classes="pnl-metric-label")
                    yield Label("$0.00", id="unrealized-pnl", classes="pnl-metric-value")

                with Static(classes="pnl-metric"):
                    yield Label("Realized", classes="pnl-metric-label")
                    yield Label("$0.00", id="realized-pnl", classes="pnl-metric-value")

                with Static(classes="pnl-metric"):
                    yield Label("Fees", classes="pnl-metric-label")
                    yield Label("$0.00", id="total-fees", classes="pnl-metric-value")

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

    def _compute_display_signature(self, state: TuiState) -> tuple:
        """Compute a signature from all fields displayed by this widget.

        Returns a tuple that can be compared for equality to detect changes.
        """
        # Account data signature (balances hash)
        acct_sig = ()
        if state.account_data:
            acct = state.account_data
            # Use tuple of balance tuples for hashable signature
            bal_sig = tuple((b.asset, str(b.total), str(b.available)) for b in acct.balances)
            acct_sig = (acct.volume_30d, acct.fees_30d, acct.fee_tier, bal_sig)

        # Position data signature (portfolio metrics)
        pos_sig = ()
        if state.position_data:
            pos = state.position_data
            pos_sig = (
                str(pos.equity),
                str(pos.total_unrealized_pnl),
                str(pos.total_realized_pnl),
                str(pos.total_fees),
                str(pos.net_pnl),
            )

        return (
            state.running,
            state.data_source_mode,
            state.degraded_mode,
            state.last_update,
            acct_sig,
            pos_sig,
        )

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts account data from TuiState and calls update_account().
        """
        # Early exit if display signature unchanged
        sig = self._compute_display_signature(state)
        if sig == self._last_display_signature:
            return
        self._last_display_signature = sig

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

        # Get portfolio metrics from position_data
        portfolio_value = state.position_data.equity if state.position_data else Decimal("0")
        unrealized_pnl = (
            state.position_data.total_unrealized_pnl if state.position_data else Decimal("0")
        )
        realized_pnl = (
            state.position_data.total_realized_pnl if state.position_data else Decimal("0")
        )
        total_fees = state.position_data.total_fees if state.position_data else Decimal("0")
        net_pnl = state.position_data.net_pnl if state.position_data else Decimal("0")

        self.update_account(
            state.account_data,
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
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
        unrealized_pnl: Decimal = Decimal("0"),
        realized_pnl: Decimal = Decimal("0"),
        total_fees: Decimal = Decimal("0"),
        net_pnl: Decimal = Decimal("0"),
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

            # Color-code Net P&L
            pnl_label = self.query_one("#net-pnl", Label)
            pnl_str = format_currency(net_pnl, decimals=2)
            if self.compact_mode:
                # Compact mode with inline label
                if net_pnl > 0:
                    pnl_label.update(f"Net: [green]+{pnl_str}[/green]")
                elif net_pnl < 0:
                    pnl_label.update(f"Net: [red]{pnl_str}[/red]")
                else:
                    pnl_label.update(f"Net: {pnl_str}")

                # Show P&L breakdown hint in compact mode
                breakdown_label = self.query_one("#pnl-breakdown", Label)
                breakdown_parts = []
                if unrealized_pnl != 0:
                    u_color = "green" if unrealized_pnl > 0 else "red"
                    breakdown_parts.append(
                        f"[{u_color}]U:{format_currency(unrealized_pnl, decimals=0)}[/{u_color}]"
                    )
                if realized_pnl != 0:
                    r_color = "green" if realized_pnl > 0 else "red"
                    breakdown_parts.append(
                        f"[{r_color}]R:{format_currency(realized_pnl, decimals=0)}[/{r_color}]"
                    )
                if total_fees > 0:
                    breakdown_parts.append(
                        f"[dim]F:-{format_currency(total_fees, decimals=0)}[/dim]"
                    )
                if breakdown_parts:
                    breakdown_label.update(" " + " ".join(breakdown_parts))
                else:
                    breakdown_label.update("")
            else:
                # Expanded mode without inline label
                if net_pnl > 0:
                    pnl_label.update(f"[green]+{pnl_str}[/green]")
                elif net_pnl < 0:
                    pnl_label.update(f"[red]{pnl_str}[/red]")
                else:
                    pnl_label.update(pnl_str)
        except Exception as e:
            logger.error(f"Failed to update portfolio summary: {e}", exc_info=True)

        # Update P&L Breakdown (only in expanded mode)
        if not self.compact_mode:
            try:
                # Unrealized P&L
                unrealized_label = self.query_one("#unrealized-pnl", Label)
                u_str = format_currency(unrealized_pnl, decimals=2)
                if unrealized_pnl > 0:
                    unrealized_label.update(f"[green]+{u_str}[/green]")
                elif unrealized_pnl < 0:
                    unrealized_label.update(f"[red]{u_str}[/red]")
                else:
                    unrealized_label.update(u_str)

                # Realized P&L
                realized_label = self.query_one("#realized-pnl", Label)
                r_str = format_currency(realized_pnl, decimals=2)
                if realized_pnl > 0:
                    realized_label.update(f"[green]+{r_str}[/green]")
                elif realized_pnl < 0:
                    realized_label.update(f"[red]{r_str}[/red]")
                else:
                    realized_label.update(r_str)

                # Total Fees (always negative impact)
                fees_label = self.query_one("#total-fees", Label)
                f_str = format_currency(total_fees, decimals=2)
                if total_fees > 0:
                    fees_label.update(f"[red]-{f_str}[/red]")
                else:
                    fees_label.update(f_str)
            except Exception as e:
                logger.error(f"Failed to update P&L breakdown: {e}", exc_info=True)

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
