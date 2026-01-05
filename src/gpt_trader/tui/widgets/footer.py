"""Custom footer with contextual keybinding hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.events import ResponsiveStateChanged
from gpt_trader.tui.responsive_state import ResponsiveState

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


class ContextualFooter(Static):
    """
    Footer that displays contextual keybindings and hints.

    Shows relevant shortcuts based on current UI state and focused widget.
    Adapts to terminal width using priority-based visibility.

    Priority tiers (P0 highest, P3 lowest):
        - P0: Essential (start/stop, reconnect, quit) - always visible
        - P1: Important (market, details, panic, config, logs) - visible at 120+ cols
        - P2: Helpful (refresh, alerts, full logs, system) - visible at 140+ cols
        - P3: Nice-to-have (info, help, theme) - visible at 160+ cols
    """

    # Use percentage-based sizing for children - height is set via global CSS ID selector
    SCOPED_CSS = False  # Disable scoping to allow nested selectors

    # Styles moved to styles/widgets/status_bar.tcss (ContextualFooter section)

    # Responsive design property
    responsive_state = reactive(ResponsiveState.STANDARD)

    # Bot running state for Start/Stop label
    bot_running = reactive(False)

    def on_responsive_state_changed(self, event: ResponsiveStateChanged) -> None:
        """Update footer shortcuts when responsive state changes."""
        self.responsive_state = event.state

    def compose(self) -> ComposeResult:
        """Compose the footer layout with priority classes.

        Priority tiers for log-centric layout:
        - P0: Essential (start/stop, reconnect, quit) - always visible
        - P1: Important (market, details, panic, config, logs) - visible at 120+ cols
        - P2: Helpful (refresh, alerts, full logs, system) - visible at 140+ cols
        - P3: Nice-to-have (info, help, theme) - visible at 160+ cols
        """
        with Horizontal(id="contextual-footer"):
            # P0: Bot control (essential)
            with Horizontal(classes="footer-group p0"):
                yield Label("S", classes="footer-key")
                yield Label("Start", classes="footer-label", id="start-stop-label")

            yield Label("│", classes="footer-separator p0")

            # P0: Reconnect (essential for recovery at any width)
            with Horizontal(classes="footer-group p0"):
                yield Label("R", classes="footer-key")
                yield Label("Reconnect", classes="footer-label")

            yield Label("│", classes="footer-separator p1")

            # P1: Market overlay (important)
            with Horizontal(classes="footer-group p1"):
                yield Label("M", classes="footer-key")
                yield Label("Market", classes="footer-label")

            yield Label("│", classes="footer-separator p1")

            # P1: Details overlay (important)
            with Horizontal(classes="footer-group p1"):
                yield Label("D", classes="footer-key")
                yield Label("Details", classes="footer-label")

            yield Label("│", classes="footer-separator p1")

            # P1: Panic (important)
            with Horizontal(classes="footer-group p1"):
                yield Label("P", classes="footer-key")
                yield Label("Panic", classes="footer-label")

            yield Label("│", classes="footer-separator p1")

            # P1: Config (important)
            with Horizontal(classes="footer-group p1"):
                yield Label("C", classes="footer-key")
                yield Label("Config", classes="footer-label")

            yield Label("│", classes="footer-separator p1")

            # P1: Logs focus (important)
            with Horizontal(classes="footer-group p1"):
                yield Label("L", classes="footer-key")
                yield Label("Logs", classes="footer-label")

            yield Label("│", classes="footer-separator p2")

            # P2: Refresh (helpful)
            with Horizontal(classes="footer-group p2"):
                yield Label("F", classes="footer-key")
                yield Label("Refresh", classes="footer-label")

            yield Label("│", classes="footer-separator p2")

            # P2: Alerts (helpful)
            with Horizontal(classes="footer-group p2"):
                yield Label("A", classes="footer-key")
                yield Label("Alerts", classes="footer-label")

            yield Label("│", classes="footer-separator p2")

            # P2: Full Logs (helpful)
            with Horizontal(classes="footer-group p2"):
                yield Label("1", classes="footer-key")
                yield Label("Full", classes="footer-label")

            yield Label("│", classes="footer-separator p2")

            # P2: System (helpful)
            with Horizontal(classes="footer-group p2"):
                yield Label("2", classes="footer-key")
                yield Label("System", classes="footer-label")

            yield Label("│", classes="footer-separator p3")

            # P3: Mode Info (nice-to-have)
            with Horizontal(classes="footer-group p3"):
                yield Label("I", classes="footer-key")
                yield Label("Info", classes="footer-label")

            yield Label("│", classes="footer-separator p3")

            # P3: Help (nice-to-have)
            with Horizontal(classes="footer-group p3"):
                yield Label("?", classes="footer-key")
                yield Label("Help", classes="footer-label")

            yield Label("│", classes="footer-separator p3")

            # P3: Theme toggle (nice-to-have)
            with Horizontal(classes="footer-group p3"):
                yield Label("T", classes="footer-key")
                yield Label("Theme", classes="footer-label")

            # P0: Quit (essential, right-aligned)
            with Horizontal(classes="footer-group footer-group-right p0"):
                yield Label("Q", classes="footer-key")
                yield Label("Quit", classes="footer-label")

    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Toggle footer shortcuts based on responsive state.

        Shows/hides shortcuts by priority tier to optimize space usage
        at different terminal widths.

        Args:
            state: ResponsiveState enum value

        Visibility by state:
            - COMPACT (100-119): P0 only (start/stop, reconnect, quit)
            - STANDARD (120-139): P0 + P1 (add market, details, panic, config, logs)
            - COMFORTABLE (140-159): P0 + P1 + P2 (add refresh, alerts, full logs, system)
            - WIDE (160+): P0 + P1 + P2 + P3 (add info, help, theme)
        """
        # Define visibility mapping for each state
        visibility = {
            ResponsiveState.COMPACT: ["p0"],
            ResponsiveState.STANDARD: ["p0", "p1"],
            ResponsiveState.COMFORTABLE: ["p0", "p1", "p2"],
            ResponsiveState.WIDE: ["p0", "p1", "p2", "p3"],
        }

        visible_priorities = visibility.get(state, ["p0", "p1"])

        try:
            # Query all footer groups and separators with priority classes
            for element in self.query(".footer-group, .footer-separator"):
                # Get the priority class for this element (p0, p1, p2, or p3)
                priority_classes = element.classes & {"p0", "p1", "p2", "p3"}

                if priority_classes:
                    priority = next(iter(priority_classes))
                    element.display = priority in visible_priorities
        except Exception:
            # Widget might not be mounted yet
            pass

    def on_mount(self) -> None:
        """Register with StateRegistry for state updates."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry.

        Updates the Start/Stop label based on bot running state.

        Args:
            state: Current TuiState.
        """
        self.bot_running = state.running

    def watch_bot_running(self, running: bool) -> None:
        """Update Start/Stop label when bot running state changes.

        Args:
            running: True if bot is running, False if stopped.
        """
        try:
            label = self.query_one("#start-stop-label", Label)
            label.update("Stop" if running else "Start")
        except Exception:
            # Widget might not be mounted yet
            pass
