"""
Validation Indicator Widget.

Displays state validation status to the user. Shows warnings and errors
from the validation layer, allowing users to see when incoming data
has issues while still operating the TUI.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual.reactive import reactive
from textual.widgets import Static

from gpt_trader.tui.events import StateValidationFailed, StateValidationPassed

if TYPE_CHECKING:
    from gpt_trader.tui.state_management.validators import FieldValidationError


class ValidationIndicatorWidget(Static):
    """Widget that displays validation status.

    Shows a compact indicator when there are validation warnings or errors.
    Clicking the indicator can expand to show details.

    Attributes:
        error_count: Number of current validation errors
        warning_count: Number of current validation warnings
        last_errors: Most recent validation errors (for display)
        last_update_time: Timestamp of last validation event
    """

    # Styles moved to styles/widgets/validation.tcss

    error_count = reactive(0)
    warning_count = reactive(0)
    last_errors: list[FieldValidationError] = []
    last_update_time: float = 0.0

    # Auto-hide after this many seconds of being valid
    AUTO_HIDE_DELAY = 5.0

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the validation indicator."""
        super().__init__(*args, **kwargs)
        self.last_errors = []
        self.last_update_time = 0.0
        self._hide_timer = None

    def compose(self) -> None:
        """No child widgets needed."""
        yield from []

    def on_mount(self) -> None:
        """Hide by default when mounted."""
        self.add_class("hidden")

    def on_state_validation_failed(self, event: StateValidationFailed) -> None:
        """Handle validation failure events.

        Args:
            event: The validation failure event with error details
        """
        self.last_errors = list(event.errors)
        self.last_update_time = time.time()

        # Count errors and warnings
        errors = [e for e in event.errors if e.severity == "error"]
        warnings = [e for e in event.errors if e.severity == "warning"]

        self.error_count = len(errors)
        self.warning_count = len(warnings)

        # Update visibility and styling
        self._update_display()

        # Cancel any pending hide timer
        if self._hide_timer:
            self._hide_timer.stop()
            self._hide_timer = None

    def on_state_validation_passed(self, event: StateValidationPassed) -> None:
        """Handle validation success events.

        Args:
            event: The validation success event
        """
        self.error_count = 0
        self.warning_count = 0
        self.last_errors = []
        self.last_update_time = time.time()

        # Update display to show valid status briefly
        self._update_display()

        # Schedule auto-hide after delay
        if self._hide_timer:
            self._hide_timer.stop()
        self._hide_timer = self.set_timer(self.AUTO_HIDE_DELAY, self._auto_hide)

    def _update_display(self) -> None:
        """Update the widget display based on current state."""
        # Remove all state classes first
        self.remove_class("hidden", "has-errors", "has-warnings", "valid")

        if self.error_count > 0:
            self.add_class("has-errors")
            self.update(self._format_error_message())
        elif self.warning_count > 0:
            self.add_class("has-warnings")
            self.update(self._format_warning_message())
        else:
            self.add_class("valid")
            self.update("✓ Data valid")

    def _format_error_message(self) -> str:
        """Format error message for display."""
        if self.error_count == 1 and self.last_errors:
            error = self.last_errors[0]
            return f"✗ {error.field}: {error.message}"
        return f"✗ {self.error_count} validation error(s)"

    def _format_warning_message(self) -> str:
        """Format warning message for display."""
        if self.warning_count == 1 and self.last_errors:
            warning = next(
                (e for e in self.last_errors if e.severity == "warning"),
                None,
            )
            if warning:
                return f"⚠ {warning.field}: {warning.message}"
        return f"⚠ {self.warning_count} validation warning(s)"

    def _auto_hide(self) -> None:
        """Auto-hide the widget after being valid for a while."""
        if self.error_count == 0 and self.warning_count == 0:
            self.add_class("hidden")

    def watch_error_count(self, new_count: int) -> None:
        """React to error count changes."""
        self._update_display()

    def watch_warning_count(self, new_count: int) -> None:
        """React to warning count changes."""
        if self.error_count == 0:  # Only update if no errors
            self._update_display()

    def get_validation_summary(self) -> str:
        """Get a summary of current validation state.

        Returns:
            Human-readable summary string
        """
        if self.error_count > 0:
            return f"{self.error_count} errors, {self.warning_count} warnings"
        elif self.warning_count > 0:
            return f"{self.warning_count} warnings"
        return "Valid"

    def get_error_details(self) -> list[dict]:
        """Get detailed information about current errors.

        Returns:
            List of error detail dictionaries
        """
        return [
            {
                "field": e.field,
                "message": e.message,
                "severity": e.severity,
                "value": str(e.value) if e.value is not None else None,
            }
            for e in self.last_errors
        ]
