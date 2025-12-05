from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from gpt_trader.tui.responsive import calculate_modal_width


class ConfigModal(ModalScreen):
    """Modal to edit critical configuration."""

    CSS = """
    ConfigModal {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    .field-label {
        text-align: right;
        padding-top: 1;
    }

    .field-input {
        width: 100%;
    }

    #config-actions {
        column-span: 2;
        layout: horizontal;
        align: center middle;
        margin-top: 2;
    }

    #btn-save {
        margin-right: 2;
    }
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        # Cache initial values
        self._initial_leverage = "1.0"
        self._initial_loss_limit = "0.02"

        if hasattr(self.config, "risk"):
            self._initial_leverage = str(getattr(self.config.risk, "max_leverage", 1.0))
            self._initial_loss_limit = str(getattr(self.config.risk, "daily_loss_limit_pct", 0.02))

    def compose(self) -> ComposeResult:
        with Grid(id="dialog"):
            yield Label("Max Leverage:", classes="field-label")
            yield Input(self._initial_leverage, id="input-leverage", classes="field-input")

            yield Label("Daily Loss Limit (%):", classes="field-label")
            yield Input(self._initial_loss_limit, id="input-loss-limit", classes="field-input")

            with Static(id="config-actions"):
                yield Button("Save", variant="primary", id="btn-save")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        """Set dynamic width based on terminal size."""
        width = calculate_modal_width(self.app.size.width, "small")
        self.query_one("#dialog").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss()
        elif event.button.id == "btn-save":
            self._save_config()
            self.dismiss()

    def _save_config(self) -> None:
        """Update configuration objects."""
        try:
            new_leverage = float(self.query_one("#input-leverage", Input).value)
            new_loss_limit = float(self.query_one("#input-loss-limit", Input).value)

            if hasattr(self.config, "risk"):
                self.config.risk.max_leverage = new_leverage
                self.config.risk.daily_loss_limit_pct = new_loss_limit

                # Also update active risk manager if possible
                # This requires access to the bot or risk manager instance, which we don't have directly here
                # Ideally, ConfigModal should take a callback or the bot instance.
                # For now, we update the config object which is shared.

                # Notify user (via app)
                self.app.notify("Configuration updated successfully.", title="Config")

        except ValueError:
            self.app.notify("Invalid input values.", severity="error", title="Config Error")
