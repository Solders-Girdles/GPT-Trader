"""Modal showing detailed mode information."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from gpt_trader.tui.responsive import calculate_modal_width


class ModeInfoModal(ModalScreen):
    """Modal displaying detailed information about current mode."""

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, mode: str) -> None:
        """
        Initialize mode info modal.

        Args:
            mode: Current bot mode (demo, paper, read_only, live)
        """
        super().__init__()
        self.mode = mode

    def compose(self) -> ComposeResult:
        """Compose modal layout."""
        config = {
            "demo": {
                "title": "Demo Mode - Mock Data",
                "data_source": "Synthetic price simulation",
                "execution": "No orders (simulated only)",
                "risk": "Zero risk - no real money",
                "use_case": "UI development and widget testing",
                "update_rate": "2 seconds (fast)",
            },
            "paper": {
                "title": "Paper Trading - Real Data",
                "data_source": "Live Coinbase REST API",
                "execution": "Simulated (no real orders sent)",
                "risk": "Zero risk - no real money",
                "use_case": "Strategy testing with real market data",
                "update_rate": "30 seconds (strategy cycle time)",
            },
            "read_only": {
                "title": "Observation Mode - Read Only",
                "data_source": "Live Coinbase REST API",
                "execution": "Blocked (orders rejected)",
                "risk": "Zero risk - no real money",
                "use_case": "Strategy observation and validation",
                "update_rate": "15 seconds (frequent)",
            },
            "live": {
                "title": "⚠ LIVE TRADING MODE ⚠",
                "data_source": "Live Coinbase REST API",
                "execution": "REAL ORDERS EXECUTED",
                "risk": "⚠ REAL MONEY AT RISK ⚠",
                "use_case": "Production trading",
                "update_rate": "30-60 seconds (strategy cycle)",
            },
        }

        info = config.get(self.mode, config["demo"])

        with Container(id="mode-info-modal"):
            with Vertical():
                yield Label(info["title"], id="mode-title")
                yield Static(f"Data Source: {info['data_source']}")
                yield Static(f"Execution: {info['execution']}")
                yield Static(f"Risk Level: {info['risk']}")
                yield Static(f"Use Case: {info['use_case']}")
                yield Static(f"Update Rate: {info['update_rate']}")
                yield Button("Close", variant="primary", id="close-btn")

    def on_mount(self) -> None:
        """Set dynamic width based on terminal size."""
        width = calculate_modal_width(self.app.size.width, "medium")
        self.query_one("#mode-info-modal").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close-btn":
            self.dismiss()

    async def action_dismiss(self) -> None:
        """Dismiss the modal."""
        self.dismiss()
