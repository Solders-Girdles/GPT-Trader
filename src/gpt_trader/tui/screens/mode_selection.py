"""Mode selection home screen for TUI launch."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Label

from gpt_trader.tui.services.preferences_service import get_preferences_service
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")

# Mode display names for UI
MODE_DISPLAY_NAMES = {
    "demo": "Demo",
    "paper": "Paper",
    "read_only": "Observe",
    "live": "Live",
}


class ModeSelectionScreen(Screen):
    """Home screen for selecting bot operating mode.

    Shows mode options with the last used mode highlighted for quick resume.
    Press Enter to resume last mode or press 1-4 to select a specific mode.
    """

    BINDINGS = [
        ("enter", "resume_last", "Resume Last"),
        ("1", "select_demo", "Demo"),
        ("2", "select_paper", "Paper"),
        ("3", "select_observe", "Observe"),
        ("4", "select_live", "Live"),
        ("s", "setup_api", "Setup API"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        """Initialize mode selection screen."""
        super().__init__()
        self._last_mode: str | None = None

    def on_mount(self) -> None:
        """Load last mode preference on mount."""
        prefs = get_preferences_service()
        self._last_mode = prefs.get_mode()
        if self._last_mode:
            logger.debug("Last used mode: %s", self._last_mode)
            # Update resume banner visibility
            try:
                banner = self.query_one("#resume-banner", Container)
                banner.display = True
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        """Compose the mode selection layout."""
        yield Footer()

        # Load last mode for initial compose
        prefs = get_preferences_service()
        self._last_mode = prefs.get_mode()

        with Container(id="mode-selection-container"):
            yield Label("GPT-TRADER TERMINAL", id="selection-title")

            # Quick resume banner (shown if last mode exists)
            with Container(id="resume-banner", classes="resume-banner"):
                if self._last_mode:
                    mode_name = MODE_DISPLAY_NAMES.get(self._last_mode, self._last_mode)
                    yield Label(
                        f"Press [Enter] to resume {mode_name} mode",
                        id="resume-hint",
                        classes="resume-hint",
                    )
                else:
                    yield Label(
                        "Select your trading mode to continue",
                        id="resume-hint",
                        classes="resume-hint",
                    )

            with Vertical():
                # Demo Mode
                with Container(classes="mode-option mode-demo"):
                    yield Label("[1] DEMO MODE - Mock Data", classes="mode-option-header")
                    yield Label(
                        "Practice with simulated market data and virtual trading",
                        classes="mode-option-description",
                    )
                    yield Label(
                        "• No real exchanges or credentials needed",
                        classes="mode-option-description",
                    )
                    yield Label(
                        "• Safe environment for testing strategies",
                        classes="mode-option-description",
                    )

                # Paper Trading Mode
                with Container(classes="mode-option mode-paper"):
                    yield Label(
                        "[2] PAPER MODE - Real Data, Simulated Trading",
                        classes="mode-option-header",
                    )
                    yield Label(
                        "Connect to Coinbase for real market data with paper trading",
                        classes="mode-option-description",
                    )
                    yield Label(
                        "• Live market prices and order books", classes="mode-option-description"
                    )
                    yield Label(
                        "• Orders simulated locally (no real execution)",
                        classes="mode-option-description",
                    )
                    yield Label(
                        "• Requires Coinbase API credentials", classes="mode-option-description"
                    )

                # Observation Mode
                with Container(classes="mode-option mode-observe"):
                    yield Label("[3] OBSERVE MODE - Read Only", classes="mode-option-header")
                    yield Label(
                        "Monitor real Coinbase account and positions (no trading)",
                        classes="mode-option-description",
                    )
                    yield Label(
                        "• View live account balances and positions",
                        classes="mode-option-description",
                    )
                    yield Label(
                        "• All trading operations disabled", classes="mode-option-description"
                    )
                    yield Label(
                        "• Requires Coinbase API credentials", classes="mode-option-description"
                    )

                # Live Trading Mode
                with Container(classes="mode-option mode-live"):
                    yield Label("[4] LIVE MODE - Real Trading", classes="mode-option-header")
                    yield Label(
                        "Execute real trades on Coinbase with real money",
                        classes="mode-option-description",
                    )
                    yield Label("• REAL MONEY AT RISK", classes="mode-option-description")
                    yield Label(
                        "• Orders executed on Coinbase exchange", classes="mode-option-description"
                    )
                    yield Label(
                        "• Requires Coinbase API credentials with trading permissions",
                        classes="mode-option-description",
                    )

                # First-time setup hint
                with Container(classes="setup-hint-container"):
                    yield Label(
                        "First time? Press [S] to setup Coinbase API credentials",
                        classes="setup-hint",
                    )

                yield Button("Quit [Q]", variant="default", id="quit-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "quit-button":
            self.app.exit()

    def action_select_demo(self) -> None:
        """Select demo mode."""
        logger.info("User selected DEMO mode from selection screen")
        self.dismiss("demo")

    def action_select_paper(self) -> None:
        """Select paper trading mode."""
        logger.info("User selected PAPER mode from selection screen")
        self.dismiss("paper")

    def action_select_observe(self) -> None:
        """Select observation mode."""
        logger.info("User selected OBSERVE mode from selection screen")
        self.dismiss("read_only")

    def action_select_live(self) -> None:
        """Select live trading mode."""
        logger.info("User selected LIVE mode from selection screen")
        self.dismiss("live")

    def action_resume_last(self) -> None:
        """Resume last used mode.

        If no last mode exists, defaults to demo mode.
        """
        if self._last_mode:
            mode_name = MODE_DISPLAY_NAMES.get(self._last_mode, self._last_mode)
            logger.info(f"User resumed last mode: {self._last_mode} ({mode_name})")
            self.dismiss(self._last_mode)
        else:
            # No last mode - default to demo
            logger.info("No last mode found, defaulting to DEMO")
            self.dismiss("demo")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_setup_api(self) -> None:
        """Launch the API setup wizard.

        Returns "setup" to signal the app to show the wizard flow.
        """
        logger.info("User requested API setup wizard from mode selection")
        self.dismiss("setup")
