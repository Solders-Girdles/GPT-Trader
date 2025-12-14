"""
Performance monitoring screen.

Full-screen modal overlay for viewing detailed performance metrics.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label

from gpt_trader.tui.widgets.performance_dashboard import PerformanceDashboardWidget


class PerformanceScreen(ModalScreen[None]):
    """Full performance metrics screen.

    Modal overlay that shows detailed performance metrics.
    Toggle with Ctrl+P, dismiss with Escape or Q.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    # Styles moved to styles/widgets/performance.tcss (PerformanceScreen section)

    def compose(self) -> ComposeResult:
        with Container(id="perf-modal"):
            yield Label("TUI Performance Monitor", classes="perf-title")
            yield PerformanceDashboardWidget(compact=False, id="perf-dashboard")
            yield Label("Press ESC or Q to close", classes="perf-footer")

    def action_dismiss(self) -> None:
        """Dismiss the modal screen."""
        self.dismiss(None)
