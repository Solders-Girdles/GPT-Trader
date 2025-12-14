from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from gpt_trader.tui.responsive import calculate_modal_width


class PanicModal(ModalScreen):
    """Modal for emergency panic button confirmation."""

    def compose(self) -> ComposeResult:
        with Vertical(id="panic-dialog"):
            yield Label("⚠️ EMERGENCY STOP ⚠️", id="panic-title")
            yield Label(
                "This will STOP the bot and CLOSE ALL POSITIONS immediately.\n"
                "This action cannot be undone.\n\n"
                "Type 'FLATTEN' to confirm.",
                id="panic-message",
            )
            yield Input(placeholder="Type FLATTEN", id="panic-input")
            with Static(id="panic-actions"):
                yield Button("FLATTEN & STOP", id="btn-panic-confirm", disabled=True)
                yield Button("Cancel", id="btn-panic-cancel")

    def on_mount(self) -> None:
        """Set dynamic width based on terminal size."""
        width = calculate_modal_width(self.app.size.width, "small")
        self.query_one("#panic-dialog").styles.width = width

    def on_input_changed(self, event: Input.Changed) -> None:
        """Enable confirm button only when input matches."""
        confirm_btn = self.query_one("#btn-panic-confirm", Button)
        if event.value == "FLATTEN":
            confirm_btn.disabled = False
        else:
            confirm_btn.disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-panic-cancel":
            self.dismiss(False)
        elif event.button.id == "btn-panic-confirm":
            self.dismiss(True)
