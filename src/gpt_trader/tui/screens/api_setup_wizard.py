"""Interactive wizard for setting up Coinbase CDP API credentials.

This screen provides step-by-step guidance for users to create and configure
their API credentials, including creating a key at the Developer Portal,
copying credentials, and setting up environment variables.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

# Total number of wizard steps
TOTAL_STEPS = 5


def _build_step_dots(current: int, total: int) -> str:
    """Build visual step indicator like ● ○ ○ ○ ○.

    Args:
        current: Current step (1-indexed).
        total: Total number of steps.

    Returns:
        String of dots showing progress.
    """
    dots = []
    for i in range(1, total + 1):
        if i < current:
            dots.append("●")  # Completed
        elif i == current:
            dots.append("◉")  # Current (larger filled)
        else:
            dots.append("○")  # Pending
    return " ".join(dots)


# Content for each wizard step
STEP_CONTENT: dict[int, dict[str, str | list[str]]] = {
    1: {
        "title": "Welcome to API Setup",
        "content": [
            "This wizard will help you set up your Coinbase CDP API credentials.",
            "",
            "You will need:",
            "  - A Coinbase account",
            "  - Access to the Coinbase Developer Portal",
            "  - ~5 minutes to complete the setup",
            "",
            "We'll guide you through:",
            "  1. Creating a CDP API key with the correct settings",
            "  2. Copying your credentials securely",
            "  3. Setting up environment variables",
            "  4. Verifying the connection",
        ],
    },
    2: {
        "title": "Create Your API Key",
        "content": [
            "Open the Coinbase Developer Portal:",
            "",
            "  https://portal.cdp.coinbase.com/",
            "",
            "Steps to create your key:",
            "",
            "  1. Sign in with your Coinbase account",
            "  2. Go to 'API Keys' section",
            "  3. Click 'Create API Key'",
            "",
            "  4. Set permissions under 'Coinbase App & Advanced Trade':",
            "     - View (read-only): Required for all modes except Demo",
            "     - Trade: Required only for Live trading",
            "",
            "  5. IMPORTANT - Expand 'Advanced Settings':",
            "     Select 'ECDSA' under Signature algorithm",
            "     (Ed25519 is recommended but NOT YET SUPPORTED)",
            "",
            "  6. Click 'Create' and proceed to the next step",
        ],
    },
    3: {
        "title": "Copy Your Credentials",
        "content": [
            "After creating the key, you'll see two values:",
            "",
            "1. API Key Name (format):",
            "   organizations/abc123.../apiKeys/xyz789...",
            "",
            "2. Private Key (format):",
            "   -----BEGIN EC PRIVATE KEY-----",
            "   ...(multiple lines of base64)...",
            "   -----END EC PRIVATE KEY-----",
            "",
            "IMPORTANT:",
            "  - Copy BOTH values now - private key shown only once!",
            "  - Store them securely (password manager recommended)",
            "  - Never share or commit these to version control",
        ],
    },
    4: {
        "title": "Configure Environment Variables",
        "content": [
            "Set the following environment variables:",
            "",
            "Option A: Add to your shell profile (~/.bashrc, ~/.zshrc):",
            "",
            '  export COINBASE_CDP_API_KEY="organizations/..."',
            '  export COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC..."',
            "",
            "Option B: Create a .env file in your project:",
            "",
            '  COINBASE_CDP_API_KEY="organizations/..."',
            '  COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC..."',
            "",
            "For production, use _PROD suffix:",
            "  COINBASE_PROD_CDP_API_KEY",
            "  COINBASE_PROD_CDP_PRIVATE_KEY",
            "",
            "After setting, restart terminal or run 'source ~/.bashrc'",
        ],
    },
    5: {
        "title": "Verify Your Setup",
        "content": [
            "Let's verify your credentials are working.",
            "",
            "When you click 'Verify', we will:",
            "  - Check your API key format",
            "  - Test the connection to Coinbase",
            "  - Verify your permissions",
            "",
            "If verification fails:",
            "  - Check that environment variables are set",
            "  - Verify you selected ES256/ECDSA (not Ed25519)",
            "  - Ensure private key includes BEGIN/END markers",
            "",
            "Click 'Verify' to test your setup!",
        ],
    },
}


class APISetupWizardScreen(ModalScreen[str | None]):
    """Multi-step wizard for API credential setup.

    Guides users through creating a CDP API key and configuring
    environment variables for authentication.

    Returns:
        "verify" - User completed wizard and wants to re-validate
        None - User cancelled
    """

    BINDINGS = [
        Binding("left", "previous", "Back", show=True),
        Binding("right", "next", "Next", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("1", "goto_step_1", "Step 1", show=False),
        Binding("2", "goto_step_2", "Step 2", show=False),
        Binding("3", "goto_step_3", "Step 3", show=False),
        Binding("4", "goto_step_4", "Step 4", show=False),
        Binding("5", "goto_step_5", "Step 5", show=False),
    ]

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the wizard screen.

        Args:
            name: Optional widget name.
            id: Optional widget ID.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.current_step = 1

    def compose(self) -> ComposeResult:
        """Build the wizard layout."""
        with Container(id="wizard-container"):
            # Visual step progress (dots)
            yield Label(
                _build_step_dots(self.current_step, TOTAL_STEPS),
                classes="wizard-dots",
                id="step-dots",
            )

            # Header with step indicator and title
            with Container(classes="wizard-header"):
                yield Label(
                    f"Step {self.current_step} of {TOTAL_STEPS}",
                    classes="wizard-step-indicator",
                    id="step-indicator",
                )
                yield Label(
                    str(STEP_CONTENT[1]["title"]),
                    classes="wizard-title",
                    id="wizard-title",
                )

            # Scrollable content area
            with VerticalScroll(id="wizard-content"):
                content = STEP_CONTENT[1]["content"]
                if isinstance(content, list):
                    for line in content:
                        yield Static(line, classes="wizard-text")

            # Navigation buttons
            with Container(id="wizard-buttons"):
                yield Button("Back", id="back-btn", disabled=True)
                yield Button("Next", id="next-btn")
                yield Button("Cancel", id="cancel-btn", variant="error")

    def _update_content(self) -> None:
        """Update wizard content for the current step."""
        step_data = STEP_CONTENT[self.current_step]

        # Update step dots
        self.query_one("#step-dots", Label).update(_build_step_dots(self.current_step, TOTAL_STEPS))

        # Update step indicator
        self.query_one("#step-indicator", Label).update(
            f"Step {self.current_step} of {TOTAL_STEPS}"
        )

        # Update title
        self.query_one("#wizard-title", Label).update(str(step_data["title"]))

        # Update content
        content_container = self.query_one("#wizard-content", VerticalScroll)
        content_container.remove_children()

        content = step_data["content"]
        if isinstance(content, list):
            for line in content:
                # Apply special styling based on content
                css_class = "wizard-text"
                if "https://" in line:
                    css_class = "wizard-text wizard-url"
                elif line.strip().startswith("export ") or "COINBASE_" in line:
                    css_class = "wizard-text wizard-code"
                elif "IMPORTANT" in line or "Do NOT" in line:
                    css_class = "wizard-text wizard-important"
                content_container.mount(Static(line, classes=css_class))

        # Update button states
        back_btn = self.query_one("#back-btn", Button)
        next_btn = self.query_one("#next-btn", Button)

        # Disable back on first step
        back_btn.disabled = self.current_step == 1

        # Change next button label on final step
        if self.current_step == TOTAL_STEPS:
            next_btn.label = "Verify"
        else:
            next_btn.label = "Next"

    def action_next(self) -> None:
        """Move to next step or trigger verification."""
        if self.current_step == TOTAL_STEPS:
            # Final step - trigger verification
            self.dismiss("verify")
        else:
            self.current_step += 1
            self._update_content()

    def action_previous(self) -> None:
        """Move to previous step."""
        if self.current_step > 1:
            self.current_step -= 1
            self._update_content()

    def action_cancel(self) -> None:
        """Cancel and close wizard."""
        self.dismiss(None)

    def _goto_step(self, step: int) -> None:
        """Jump to a specific step."""
        if 1 <= step <= TOTAL_STEPS:
            self.current_step = step
            self._update_content()

    def action_goto_step_1(self) -> None:
        """Jump to step 1."""
        self._goto_step(1)

    def action_goto_step_2(self) -> None:
        """Jump to step 2."""
        self._goto_step(2)

    def action_goto_step_3(self) -> None:
        """Jump to step 3."""
        self._goto_step(3)

    def action_goto_step_4(self) -> None:
        """Jump to step 4."""
        self._goto_step(4)

    def action_goto_step_5(self) -> None:
        """Jump to step 5."""
        self._goto_step(5)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.action_previous()
        elif event.button.id == "next-btn":
            self.action_next()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
