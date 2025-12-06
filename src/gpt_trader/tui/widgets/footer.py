"""Custom footer with contextual keybinding hints."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.widgets.mode_indicator import ModeIndicator


class ContextualFooter(Static):
    """
    Footer that displays contextual keybindings and hints.

    Shows relevant shortcuts based on current UI state and focused widget.
    Adapts to terminal width using priority-based visibility.

    Priority tiers (P0 highest, P3 lowest):
        - P0: Essential (data source, start/stop, quit) - always visible
        - P1: Important (panic, logs, config) - visible at 120+ cols
        - P2: Helpful (full logs, system) - visible at 140+ cols
        - P3: Nice-to-have (help) - visible at 160+ cols
    """

    # Use percentage-based sizing for children - height is set via global CSS ID selector
    SCOPED_CSS = False  # Disable scoping to allow nested selectors

    DEFAULT_CSS = """
    ContextualFooter > Horizontal {
        height: 100%;
        width: 100%;
    }
    """

    # Responsive design property
    responsive_state = reactive(ResponsiveState.STANDARD)

    def compose(self) -> ComposeResult:
        """Compose the footer layout with priority classes.

        Priority tiers for log-centric layout:
        - P0: Essential (data source, start/stop, market, details, quit)
        - P1: Important (panic, config, logs)
        - P2: Helpful (full logs, system)
        - P3: Nice-to-have (help, mode info)
        """
        with Horizontal(id="contextual-footer"):
            # P0: Data source info (essential, left side)
            with Horizontal(classes="footer-group footer-group-left p0"):
                yield Label("", id="data-source-info", classes="footer-data-source")

            yield Label("│", classes="footer-separator p0")

            # P0: Bot control (essential)
            with Horizontal(classes="footer-group p0"):
                yield Label("S", classes="footer-key")
                yield Label("Start", classes="footer-label", id="start-stop-label")

            yield Label("│", classes="footer-separator p0")

            # P0: Market overlay (essential for new layout)
            with Horizontal(classes="footer-group p0"):
                yield Label("M", classes="footer-key")
                yield Label("Market", classes="footer-label")

            yield Label("│", classes="footer-separator p0")

            # P0: Details overlay (essential for new layout)
            with Horizontal(classes="footer-group p0"):
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

            # P0: Quit (essential, right-aligned)
            with Horizontal(classes="footer-group footer-group-right p0"):
                yield Label("Q", classes="footer-key")
                yield Label("Quit", classes="footer-label")

    def update_data_source_info(self, mode: str, connection_healthy: bool) -> None:
        """
        Update data source information in footer.

        Args:
            mode: Current bot mode (demo, paper, read_only, live)
            connection_healthy: True if connection is healthy
        """
        try:
            info_label = self.query_one("#data-source-info", Label)

            status_icon = "🟢" if connection_healthy else "🔴"
            config = ModeIndicator.MODE_CONFIG.get(mode, {})
            description = config.get("description", "Unknown mode")

            info_label.update(f"{status_icon} {description}")
        except Exception:
            # Widget might not be mounted yet
            pass

    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Toggle footer shortcuts based on responsive state.

        Shows/hides shortcuts by priority tier to optimize space usage
        at different terminal widths.

        Args:
            state: ResponsiveState enum value

        Visibility by state:
            - COMPACT (100-119): P0 only (data source, start/stop, quit)
            - STANDARD (120-139): P0 + P1 (add panic, logs, config)
            - COMFORTABLE (140-159): P0 + P1 + P2 (add full logs, system)
            - WIDE (160+): P0 + P1 + P2 + P3 (add help)
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
