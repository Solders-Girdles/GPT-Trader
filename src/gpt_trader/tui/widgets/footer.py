"""Custom footer with contextual keybinding hints."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Static


class ContextualFooter(Static):
    """
    Footer that displays contextual keybindings and hints.

    Shows relevant shortcuts based on current UI state and focused widget.
    """

    def compose(self) -> ComposeResult:
        """Compose the footer layout."""
        with Horizontal(id="contextual-footer"):
            # Bot control shortcuts
            with Horizontal(classes="footer-group"):
                yield Label("[S]", classes="footer-key")
                yield Label("Start/Stop", classes="footer-label")

            yield Label("|", classes="footer-separator")

            with Horizontal(classes="footer-group"):
                yield Label("[P]", classes="footer-key")
                yield Label("Panic", classes="footer-label")

            yield Label("|", classes="footer-separator")

            # Navigation shortcuts
            with Horizontal(classes="footer-group"):
                yield Label("[C]", classes="footer-key")
                yield Label("Config", classes="footer-label")

            yield Label("|", classes="footer-separator")

            with Horizontal(classes="footer-group"):
                yield Label("[L]", classes="footer-key")
                yield Label("Logs", classes="footer-label")

            yield Label("|", classes="footer-separator")

            # Utility shortcuts
            with Horizontal(classes="footer-group"):
                yield Label("[?]", classes="footer-key")
                yield Label("Help", classes="footer-label")

            # Quit (right-aligned)
            with Horizontal(classes="footer-group footer-group-right"):
                yield Label("[Q]", classes="footer-key")
                yield Label("Quit", classes="footer-label")
