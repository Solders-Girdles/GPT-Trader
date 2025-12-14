"""
Responsive manager for handling terminal resize events.

Manages responsive state transitions based on terminal width,
with debouncing to prevent excessive updates during rapid resizing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.tui.events import ResponsiveStateChanged
from gpt_trader.tui.responsive import calculate_responsive_state
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App
    from textual.timer import Timer

logger = get_logger(__name__, component="tui")

# Debounce delay for resize events (seconds)
RESIZE_DEBOUNCE_DELAY = 0.1


class ResponsiveManager:
    """Manager for responsive terminal state.

    Handles terminal resize events with debouncing and propagates
    responsive state changes to the app and its widgets.

    Attributes:
        app: Reference to the parent Textual app.
        current_state: Current responsive state enum.
        current_width: Current terminal width in columns.
    """

    def __init__(self, app: App) -> None:
        """Initialize the responsive manager.

        Args:
            app: The parent Textual app.
        """
        self.app = app
        self.current_state: ResponsiveState = ResponsiveState.STANDARD
        self.current_width: int = 120
        self._resize_timer: Timer | None = None

    def initialize(self, width: int) -> ResponsiveState:
        """Initialize responsive state from terminal width.

        Should be called during app mount to set initial state.

        Args:
            width: Initial terminal width in columns.

        Returns:
            The calculated responsive state.
        """
        self.current_width = width
        self.current_state = calculate_responsive_state(width)
        logger.debug("Initial responsive state: %s (width: %s)", self.current_state, width)
        return self.current_state

    def handle_resize(self, new_width: int) -> None:
        """Handle terminal resize with debouncing.

        Schedules a debounced update to prevent excessive repaints
        during rapid resizing.

        Args:
            new_width: New terminal width in columns.
        """
        # Cancel any pending resize timer
        if self._resize_timer is not None:
            self._resize_timer.stop()
            self._resize_timer = None

        # Schedule debounced update
        self._resize_timer = self.app.set_timer(
            RESIZE_DEBOUNCE_DELAY,
            lambda: self._apply_resize(new_width),
        )

    def _apply_resize(self, width: int) -> None:
        """Apply the resize after debounce delay.

        Args:
            width: New terminal width in columns.
        """
        old_state = self.current_state
        new_state = calculate_responsive_state(width)

        self.current_width = width
        self._resize_timer = None

        if new_state != old_state:
            self.current_state = new_state
            logger.debug(
                "Responsive state changed: %s -> %s (width: %s)", old_state, new_state, width
            )

            # Brief user feedback when breakpoint changes
            try:
                self.app.notify(f"Layout: {new_state.value.title()}", timeout=1)
            except Exception:
                pass

            # Post event for interested widgets
            self.app.post_message(ResponsiveStateChanged(state=new_state, width=width))

            # Propagate to app's reactive property if it exists
            if hasattr(self.app, "responsive_state"):
                self.app.responsive_state = new_state  # type: ignore[attr-defined]

    def propagate_to_screen(self) -> None:
        """Propagate current responsive state to the active screen.

        Should be called when the screen changes or when state needs
        to be synchronized.
        """
        try:
            screen = self.app.screen
            if hasattr(screen, "responsive_state"):
                screen.responsive_state = self.current_state  # type: ignore[attr-defined]
                logger.debug(f"Propagated responsive state to screen: {self.current_state}")
        except Exception:
            # Screen might not be mounted yet
            pass

    def cleanup(self) -> None:
        """Cleanup any pending timers."""
        if self._resize_timer is not None:
            self._resize_timer.stop()
            self._resize_timer = None
