from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.types import AccountSummary


class AccountWidget(Static):
    """Widget to display account metrics."""

    def compose(self) -> ComposeResult:
        yield Label("Account Summary", classes="header")
        with Container(classes="account-summary"):
            yield Label("Volume 30d: $0.00", id="acc-volume")
            yield Label("Fees 30d: $0.00", id="acc-fees")
            yield Label("Fee Tier: -", id="acc-tier")

        yield Label("Balances", classes="header")
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Asset", "Total", "Available")

    def update_account(self, data: AccountSummary) -> None:
        # Update Summary
        try:
            self.query_one("#acc-volume", Label).update(f"Volume 30d: ${data.volume_30d}")
            self.query_one("#acc-fees", Label).update(f"Fees 30d: ${data.fees_30d}")
            self.query_one("#acc-tier", Label).update(f"Fee Tier: {data.fee_tier}")
        except Exception:
            pass

        # Update Balances
        table = self.query_one(DataTable)
        table.clear()

        for bal in data.balances:
            table.add_row(bal.asset, bal.total, bal.available)
