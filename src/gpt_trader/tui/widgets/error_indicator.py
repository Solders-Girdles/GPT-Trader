"""Error Indicator Widget for persistent error tracking in the TUI."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__, component="tui")


@dataclass
class ErrorEntry:
    """Represents a single error in the error tracker."""

    widget: str
    method: str
    error: str
    timestamp: float


class ErrorIndicatorWidget(Static):
    """
    Displays persistent error tracking with collapsible details.

    Features:
    - Compact badge showing error count
    - Collapsible to show error details
    - Max 10 errors (FIFO queue)
    - Clear/acknowledge button
    - Red theme for visibility
    """

    DEFAULT_CSS = """
    ErrorIndicatorWidget {
        layout: vertical;
        height: auto;
        max-height: 15;
        min-height: 1;
        background: #E08580;  /* Theme error background */
        color: #F0EDE9;  /* Theme primary text */
        border: thick #D4744F;  /* Theme emphasized border */
        padding: 0 1;
    }

    ErrorIndicatorWidget.collapsed {
        max-height: 1;
    }

    ErrorIndicatorWidget.hidden {
        display: none;
    }

    ErrorIndicatorWidget Horizontal {
        height: auto;
        width: 1fr;
    }

    ErrorIndicatorWidget Vertical {
        height: auto;
        width: 1fr;
    }

    ErrorIndicatorWidget Label {
        width: 1fr;
        height: auto;
    }

    ErrorIndicatorWidget .error-badge {
        background: #3D3833;  /* Theme elevated bg */
        color: #F0EDE9;  /* Theme primary text */
        text-style: bold;
    }

    ErrorIndicatorWidget .error-detail {
        color: #B8B4AF;  /* Theme secondary text */
        height: auto;
        margin: 0 0 0 2;
    }

    ErrorIndicatorWidget Button {
        min-width: 10;
    }
    """

    # Reactive properties
    error_count = reactive(0)
    is_collapsed = reactive(True)

    def __init__(self, max_errors: int = 10) -> None:
        """
        Initialize error indicator widget.

        Args:
            max_errors: Maximum number of errors to track (FIFO queue)
        """
        super().__init__(id="error-indicator")
        self._errors: deque[ErrorEntry] = deque(maxlen=max_errors)
        self._max_errors = max_errors

    def compose(self) -> ComposeResult:
        """Compose error indicator widget."""
        with Horizontal(id="error-header"):
            yield Label("⚠️  0 errors", id="error-badge", classes="error-badge")
            yield Button("Toggle", id="toggle-btn", variant="error")
            yield Button("Clear", id="clear-btn", variant="error")

        with Vertical(id="error-details"):
            yield Label("No errors", id="error-list")

    def on_mount(self) -> None:
        """Set initial collapsed state."""
        self.add_class("collapsed")
        self.add_class("hidden")  # Start hidden when no errors

    def add_error(self, widget: str, method: str, error: str) -> None:
        """
        Add an error to the tracker.

        Args:
            widget: Widget class name where error occurred
            method: Method name where error occurred
            error: Error message
        """
        entry = ErrorEntry(widget=widget, method=method, error=error, timestamp=time.time())

        self._errors.append(entry)
        self.error_count = len(self._errors)

        logger.debug(
            f"Error tracked: {widget}.{method} - {error} "
            f"(total: {self.error_count}/{self._max_errors})"
        )

        self._update_display()

    def clear_errors(self) -> None:
        """Clear all tracked errors."""
        count = len(self._errors)
        self._errors.clear()
        self.error_count = 0

        logger.debug(f"Cleared {count} tracked errors")

        self._update_display()

    def _update_display(self) -> None:
        """Update the error display with current errors."""
        # Update badge (only if mounted)
        try:
            badge = self.query_one("#error-badge", Label)
        except Exception:
            # Widget not mounted yet, skip display update
            return
        if self.error_count == 0:
            badge.update("⚠️  0 errors")
            self.add_class("hidden")
        elif self.error_count == 1:
            badge.update("⚠️  1 error")
            self.remove_class("hidden")
        else:
            badge.update(f"⚠️  {self.error_count} errors")
            self.remove_class("hidden")

        # Update details
        if not self.is_collapsed and self.error_count > 0:
            error_list = self.query_one("#error-list", Label)
            details_lines = []

            for entry in self._errors:
                # Format timestamp as HH:MM:SS
                time_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
                error_preview = entry.error[:50] + ("..." if len(entry.error) > 50 else "")
                details_lines.append(f"[{time_str}] {entry.widget}.{entry.method}: {error_preview}")

            error_list.update("\n".join(details_lines))
        else:
            error_list = self.query_one("#error-list", Label)
            error_list.update("No errors" if self.error_count == 0 else "Collapsed")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "toggle-btn":
            self.is_collapsed = not self.is_collapsed

            if self.is_collapsed:
                self.add_class("collapsed")
                event.button.label = "Expand"
            else:
                self.remove_class("collapsed")
                event.button.label = "Collapse"

            self._update_display()

        elif event.button.id == "clear-btn":
            self.clear_errors()
