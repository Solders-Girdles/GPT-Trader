"""
Full-screen log viewer with filtering and search.

This screen provides an expanded view of system logs with search
functionality and navigation.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label

from gpt_trader.tui.widgets import LogWidget


class FullLogsScreen(Screen):
    """Full-screen log viewer with filtering, search, and expanded view.

    Provides a dedicated screen for viewing and searching through
    system logs with regex support.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("/", "focus_search", "Search"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the full logs screen layout."""
        yield Header(show_clock=True, classes="app-header", icon="*", time_format="%H:%M:%S")
        yield Label("FULL SYSTEM LOGS", classes="header")

        # Search bar
        with Horizontal(id="search-bar"):
            yield Input(placeholder="Search logs (regex supported)...", id="log-search-input")
            yield Label("", id="search-match-count")

        yield LogWidget(id="full-logs", compact_mode=False)  # Expanded mode for full logs screen
        yield Footer()

    def action_dismiss(self, result: object = None) -> None:
        """Close the full logs screen and return to main view."""
        self.app.pop_screen()

    def action_focus_search(self) -> None:
        """Focus the search input."""
        search_input = self.query_one("#log-search-input", Input)
        search_input.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "log-search-input":
            # Update match count label
            match_count_label = self.query_one("#search-match-count", Label)
            search_text = event.value.strip()
            if search_text:
                # Placeholder for future search implementation
                match_count_label.update(f"[Press Enter to search: '{search_text}']")
            else:
                match_count_label.update("")
