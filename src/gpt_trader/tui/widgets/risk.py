from gpt_trader.tui.models import RiskData
from textual.app import ComposeResult
from textual.containers import Grid
from textual.widgets import Label, Static


class RiskWidget(Static):
    """Widget to display risk management status."""

    def compose(self) -> ComposeResult:
        yield Label("RISK MANAGEMENT", classes="header")
        with Grid(classes="risk-grid"):
            # Max Leverage
            yield Label("Max Leverage:", classes="risk-label")
            yield Label("-", id="max-leverage", classes="risk-value")

            # Daily Loss Limit
            yield Label("Daily Loss Limit:", classes="risk-label")
            yield Label("-", id="loss-limit", classes="risk-value")

            # Reduce Only Mode
            yield Label("Reduce Only:", classes="risk-label")
            yield Label("OFF", id="reduce-only", classes="risk-value")

            # Active Guards (Placeholder for now)
            yield Label("Active Guards:", classes="risk-label")
            yield Label("None", id="active-guards", classes="risk-value")

    def update_risk(self, data: RiskData) -> None:
        # Update Max Leverage
        self.query_one("#max-leverage", Label).update(f"{data.max_leverage}x")

        # Update Daily Loss Limit
        self.query_one("#loss-limit", Label).update(f"{data.daily_loss_limit_pct:.1%}")

        # Update Reduce Only
        reduce_only_label = self.query_one("#reduce-only", Label)
        if data.reduce_only_mode:
            reduce_only_label.update(f"ON ({data.reduce_only_reason})")
            reduce_only_label.add_class("risk-alert")
        else:
            reduce_only_label.update("OFF")
            reduce_only_label.remove_class("risk-alert")

        # Update Active Guards
        guards_label = self.query_one("#active-guards", Label)
        if data.active_guards:
            guards_label.update(", ".join(data.active_guards))
        else:
            guards_label.update("None")
