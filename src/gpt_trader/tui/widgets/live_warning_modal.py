"""Warning modal for live trading mode."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from gpt_trader.tui.responsive import calculate_modal_width


class LiveWarningModal(ModalScreen):
    """Warning shown when starting in live trading mode."""

    BINDINGS = [
        ("c", "continue_live", "Continue"),
        ("q", "quit_app", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose warning layout."""
        with Container(id="live-warning-modal"):
            with Vertical():
                yield Label("⚠ LIVE TRADING MODE ⚠", id="warning-title")
                yield Label("You are about to start the bot in LIVE mode.")
                yield Label("Real orders will be placed on Coinbase.")
                yield Label("REAL MONEY IS AT RISK.", id="risk-warning")
                yield Label("")
                yield Label("Press 'C' to continue or 'Q' to quit")
                with Container(id="warning-buttons"):
                    yield Button("Continue [C]", variant="error", id="continue-btn")
                    yield Button("Quit [Q]", variant="primary", id="quit-btn")

    def on_mount(self) -> None:
        """Set dynamic width based on terminal size."""
        width = calculate_modal_width(self.app.size.width, "medium")
        self.query_one("#live-warning-modal").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "continue-btn":
            self.dismiss(True)
        elif event.button.id == "quit-btn":
            self.dismiss(False)

    async def action_continue_live(self) -> None:
        """Dismiss modal and continue."""
        self.dismiss(True)

    async def action_quit_app(self) -> None:
        """Quit application."""
        self.dismiss(False)
