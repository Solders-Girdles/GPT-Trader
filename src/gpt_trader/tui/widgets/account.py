from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import AccountSummary
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class AccountWidget(Static):
    """Widget to display account metrics with optional compact mode."""

    DEFAULT_CSS = """
    AccountWidget {
        layout: vertical;
        height: 1fr;
    }

    AccountWidget DataTable {
        height: auto;
    }

    AccountWidget.compact .balance-details {
        display: none;  /* Hide balance table in compact mode */
    }

    AccountWidget.compact .account-metrics-row {
        display: none;  /* Hide detailed metrics in compact mode */
    }

    .portfolio-summary {
        layout: horizontal;
        align-vertical: middle;
    }

    .metric-separator {
        color: #7A7672;  /* text-muted - UPDATED */
        margin: 0 1;
    }
    """

    def __init__(self, compact_mode: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode

    def compose(self) -> ComposeResult:
        yield Label("💰 ACCOUNT", classes="header")

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
                yield Label("Balances", classes="header")
                yield DataTable()

    def on_mount(self) -> None:
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

    @safe_update
    def update_account(
        self, data: AccountSummary, portfolio_value: str = "0.00", total_pnl: str = "0.00"
    ) -> None:
        # Update Portfolio Summary (Primary Metrics)
        try:
            portfolio_label = self.query_one("#portfolio-value", Label)

            if self.compact_mode:
                # Compact mode shows labels inline
                portfolio_label.update(f"Portfolio: ${portfolio_value}")
            else:
                # Expanded mode shows just values
                portfolio_label.update(f"${portfolio_value}")

            # Color-code P&L
            pnl_label = self.query_one("#total-pnl", Label)
            try:
                pnl_float = float(total_pnl)
                if self.compact_mode:
                    # Compact mode with inline label
                    if pnl_float > 0:
                        pnl_label.update(f"P&L: [green]+${pnl_float:.2f}[/green]")
                    elif pnl_float < 0:
                        pnl_label.update(f"P&L: [red]-${abs(pnl_float):.2f}[/red]")
                    else:
                        pnl_label.update(f"P&L: ${pnl_float:.2f}")
                else:
                    # Expanded mode without inline label
                    if pnl_float > 0:
                        pnl_label.update(f"[green]+${pnl_float:.2f}[/green]")
                    elif pnl_float < 0:
                        pnl_label.update(f"[red]-${abs(pnl_float):.2f}[/red]")
                    else:
                        pnl_label.update(f"${pnl_float:.2f}")
            except (ValueError, TypeError):
                if self.compact_mode:
                    pnl_label.update(f"P&L: ${total_pnl}")
                else:
                    pnl_label.update(f"${total_pnl}")
        except Exception as e:
            logger.error(f"Failed to update portfolio summary: {e}", exc_info=True)

        # Update Account Summary (only in expanded mode)
        if not self.compact_mode:
            try:
                self.query_one("#acc-volume", Label).update(f"${data.volume_30d}")
                self.query_one("#acc-fees", Label).update(f"${data.fees_30d}")
                self.query_one("#acc-tier", Label).update(f"{data.fee_tier}")
            except Exception as e:
                logger.error(f"Failed to update account metrics: {e}", exc_info=True)

            # Update Balances (only in expanded mode)
            try:
                table = self.query_one(DataTable)
                table.clear()

                for bal in data.balances:
                    table.add_row(bal.asset, bal.total, bal.available)
            except Exception as e:
                logger.error(f"Failed to update balances: {e}", exc_info=True)
