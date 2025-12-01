from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import AccountSummary


class AccountWidget(Static):
    """Widget to display account metrics."""

    def compose(self) -> ComposeResult:
        yield Label("Account Summary", classes="header")

        # Metrics Row
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

        # Add some CSS for the row layout if not present globally
        self.styles.layout = "vertical"

        # We can inject styles dynamically or rely on tcss.
        # Let's assume tcss handles .account-metric.
        # But we need a container for the row.
        row = self.query_one(".account-metrics-row")
        row.styles.layout = "horizontal"
        row.styles.height = "auto"

    @safe_update
    def update_account(self, data: AccountSummary) -> None:
        # Update Summary
        try:
            self.query_one("#acc-volume", Label).update(f"${data.volume_30d}")
            self.query_one("#acc-fees", Label).update(f"${data.fees_30d}")
            self.query_one("#acc-tier", Label).update(f"{data.fee_tier}")
        except Exception:
            pass

        # Update Balances
        table = self.query_one(DataTable)
        table.clear()

        for bal in data.balances:
            table.add_row(bal.asset, bal.total, bal.available)
