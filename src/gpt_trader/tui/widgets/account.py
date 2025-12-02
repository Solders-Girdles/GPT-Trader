from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import AccountSummary
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class AccountWidget(Static):
    """Widget to display account metrics."""

    def compose(self) -> ComposeResult:
        yield Label("Account Summary", classes="header")

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

        yield Label("Balances", classes="header")
        yield DataTable()

    def on_mount(self) -> None:
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

    @safe_update
    def update_account(
        self, data: AccountSummary, portfolio_value: str = "0.00", total_pnl: str = "0.00"
    ) -> None:
        # Update Portfolio Summary (Primary Metrics)
        try:
            self.query_one("#portfolio-value", Label).update(f"${portfolio_value}")

            # Color-code P&L
            pnl_label = self.query_one("#total-pnl", Label)
            try:
                pnl_float = float(total_pnl)
                if pnl_float > 0:
                    pnl_label.update(f"[green]+${pnl_float:.2f}[/green]")
                elif pnl_float < 0:
                    pnl_label.update(f"[red]-${abs(pnl_float):.2f}[/red]")
                else:
                    pnl_label.update(f"${pnl_float:.2f}")
            except (ValueError, TypeError):
                pnl_label.update(f"${total_pnl}")
        except Exception as e:
            logger.error(f"Failed to update portfolio summary: {e}", exc_info=True)

        # Update Account Summary
        try:
            self.query_one("#acc-volume", Label).update(f"${data.volume_30d}")
            self.query_one("#acc-fees", Label).update(f"${data.fees_30d}")
            self.query_one("#acc-tier", Label).update(f"{data.fee_tier}")
        except Exception as e:
            logger.error(f"Failed to update account metrics: {e}", exc_info=True)

        # Update Balances
        table = self.query_one(DataTable)
        table.clear()

        for bal in data.balances:
            table.add_row(bal.asset, bal.total, bal.available)
