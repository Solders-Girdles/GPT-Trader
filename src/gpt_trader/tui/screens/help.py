"""Help screen showing comprehensive keyboard shortcut reference."""

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static


class HelpScreen(ModalScreen):
    """
    Comprehensive keyboard shortcut reference overlay.

    Displays all available shortcuts organized by category:
    - Essential Controls (quit, start/stop)
    - Navigation & Views (logs, system, config)
    - Emergency Actions (panic button)
    - Screen-Specific (full-screen modals)
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("?", "dismiss", "Close"),  # Same key that opens help
    ]

    def compose(self) -> ComposeResult:
        """Compose help screen with categorized shortcuts."""
        with Container(id="help-container"):
            with VerticalScroll():
                yield Label("⌨ KEYBOARD SHORTCUTS", classes="help-title")

                # Essential Controls
                yield Label("Essential Controls", classes="help-category")
                yield self._create_shortcut("Q", "Quit application")
                yield self._create_shortcut("S", "Start/Stop bot")
                yield self._create_shortcut("?", "Show this help screen")

                # Tile Navigation
                yield Label("Tile Navigation", classes="help-category")
                yield self._create_shortcut("↑/↓/←/→", "Move focus between tiles")
                yield self._create_shortcut("ENTER", "Open focused tile details")
                yield self._create_shortcut("ESC", "Return from detail screen")

                # Navigation & Views
                yield Label("Navigation & Views", classes="help-category")
                yield self._create_shortcut("M", "Show market overlay")
                yield self._create_shortcut("D", "Show details overlay")
                yield self._create_shortcut("W", "Edit watchlist symbols")
                yield self._create_shortcut("L", "Focus log viewer (scroll/read)")
                yield self._create_shortcut("1", "Show full logs screen")
                yield self._create_shortcut("2", "Show system details screen")
                yield self._create_shortcut("A", "Show alert history")
                yield self._create_shortcut("I", "Show mode information")

                # Trading Stats (Account Tile)
                yield Label("Trading Stats (Account Tile focused)", classes="help-category")
                yield self._create_shortcut("w", "Cycle time window (1h/4h/24h/7d)")
                yield self._create_shortcut("W", "Reset to all-time stats")

                # Data Tables
                yield Label("Data Tables (Positions, Trades, Orders)", classes="help-category")
                yield self._create_shortcut("c", "Copy selected row to clipboard")
                yield self._create_shortcut("C", "Copy all rows with headers")
                yield self._create_shortcut("↑/↓", "Navigate rows")
                yield self._create_shortcut("ENTER", "Select/expand row")

                # Orders Table
                yield Label("Orders Table (when focused)", classes="help-category")
                yield self._create_shortcut("ENTER", "View order details (fills, fees, trades)")
                yield self._create_shortcut("S", "Cycle sort: Fill% → Age → None")
                yield self._create_shortcut("c", "Copy selected order to clipboard")

                # Trades Table
                yield Label("Trades Table (when focused)", classes="help-category")
                yield self._create_shortcut("f", "Cycle symbol filter")
                yield self._create_shortcut("F", "Clear all filters")

                # Risk Tile
                yield Label("Risk Tile (when focused)", classes="help-category")
                yield self._create_shortcut("ENTER/G", "View risk detail (guards, limits, score)")
                yield self._create_shortcut("L", "Focus log viewer")
                yield self._create_shortcut("D", "Reset daily risk tracking")

                # Accessibility
                yield Label("Accessibility", classes="help-category")
                yield self._create_shortcut("TAB", "Focus next widget")
                yield self._create_shortcut("SHIFT+TAB", "Focus previous widget")
                yield self._create_shortcut("ARROW KEYS", "Navigate tables/lists")
                yield self._create_shortcut("HOME/END", "Jump to start/end")
                yield self._create_shortcut("PAGEUP/DN", "Scroll by page")
                yield self._create_shortcut("J/K", "Navigate Portfolio tabs (when focused)")
                yield self._create_shortcut("1/2/3", "Jump to Portfolio tab 1/2/3 (when focused)")

                # Log Navigation (when log widget focused)
                yield Label("Log Navigation (when focused)", classes="help-category")
                yield self._create_shortcut("SPACE", "Pause/Resume log streaming")
                yield self._create_shortcut("V", "Cycle format (compact/verbose/JSON)")
                yield self._create_shortcut("J/K", "Scroll up/down one line")
                yield self._create_shortcut("CTRL+D/U", "Scroll half page down/up")
                yield self._create_shortcut("g/G", "Jump to top/bottom")
                yield self._create_shortcut("n/N", "Jump to next/previous error")
                yield self._create_shortcut("CTRL+T", "Cycle timestamp format")
                yield self._create_shortcut("CTRL+S", "Toggle startup section")
                yield self._create_shortcut("1-5", "Filter: All/Error/Warn/Info/Debug")
                yield self._create_shortcut("f", "Cycle level filter")
                yield self._create_shortcut("F", "Clear all filters")

                # Configuration & Display
                yield Label("Configuration & Display", classes="help-category")
                yield self._create_shortcut("C", "Open configuration modal")
                yield self._create_shortcut("R", "Reconnect to data source")
                yield self._create_shortcut("F", "Force refresh")
                yield self._create_shortcut("T", "Toggle theme")
                yield self._create_shortcut("CTRL+K", "Open command palette")
                yield self._create_shortcut("CTRL+P", "Toggle performance overlay")

                # Emergency Actions
                yield Label("Emergency Actions", classes="help-category")
                yield self._create_shortcut("P", "PANIC - Emergency stop & flatten positions")

                # Alert History Screen
                yield Label("Alert History (press A to open)", classes="help-category")
                yield self._create_shortcut("1-5", "Filter: All/Trade/System/Risk/Error")
                yield self._create_shortcut("y", "Copy selected alert")
                yield self._create_shortcut("Y", "Copy all alerts with headers")
                yield self._create_shortcut("x", "Clear alert history")
                yield self._create_shortcut("r", "Reset alert cooldowns")
                yield self._create_shortcut("ESC", "Close alert history")

                # Modal-Specific
                yield Label("Modal Controls (when overlay open)", classes="help-category")
                yield self._create_shortcut("ESC", "Close current modal/overlay")
                yield self._create_shortcut("Q", "Close current modal/overlay")

                # Mode Selection (startup only)
                yield Label("Mode Selection (Startup Screen)", classes="help-category")
                yield self._create_shortcut("1", "Select Demo mode")
                yield self._create_shortcut("2", "Select Paper Trading mode")
                yield self._create_shortcut("3", "Select Observation (read-only) mode")
                yield self._create_shortcut("4", "Select Live Trading mode")

                # Footer
                yield Label("Press ESC, Q, or ? to close this help screen", classes="help-footer")

    def _create_shortcut(self, key: str, description: str) -> Static:
        """
        Create a shortcut display row.

        Args:
            key: The keyboard key (e.g., "Q", "ESC")
            description: Description of what the key does

        Returns:
            Static widget displaying the shortcut
        """
        shortcut = Static(classes="help-shortcut")

        # Create inner widgets
        key_label = Label(f"[{key}]", classes="help-key")
        desc_label = Label(description, classes="help-description")

        # Compose shortcut layout
        def compose_shortcut() -> ComposeResult:
            yield key_label
            yield desc_label

        shortcut.compose = compose_shortcut
        return shortcut

    async def action_dismiss(self) -> None:
        """Close help screen and return to previous screen."""
        self.dismiss()
