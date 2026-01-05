"""Value flash animation utilities for visual feedback on data changes.

Provides a mixin and helper functions to create brief visual highlights
when values update, giving users feedback that data is flowing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.timer import Timer
from textual.widgets import Label, Static

if TYPE_CHECKING:
    from textual.widget import Widget


class ValueFlashMixin:
    """Mixin to add value flash capability to widgets.

    Add this mixin to any widget that displays values which update over time.
    Call `flash_value()` when a value changes to create a brief visual highlight.

    Example:
        class MyWidget(ValueFlashMixin, Static):
            def update_price(self, new_price: float) -> None:
                old_price = self._price
                self._price = new_price
                if old_price != new_price:
                    direction = "up" if new_price > old_price else "down"
                    self.flash_value("#price-label", direction)
    """

    # Cache for previous values to detect changes
    _flash_cache: dict[str, Any]
    _flash_timers: dict[str, Timer]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._flash_cache = {}
        self._flash_timers = {}

    def flash_value(
        self: Widget,
        selector: str,
        direction: str = "neutral",
        duration: float = 0.5,
    ) -> None:
        """Flash a child widget to indicate value change.

        Args:
            selector: CSS selector for the widget to flash (e.g., "#price-label").
            direction: "up" for positive change, "down" for negative, "neutral" for either.
            duration: How long the flash should last in seconds.
        """
        try:
            widget = self.query_one(selector)
        except Exception:
            return

        # Cancel any existing flash timer on this element
        if selector in self._flash_timers:
            try:
                self._flash_timers[selector].stop()
            except Exception:
                pass

        # Determine flash class
        if direction == "up":
            flash_class = "value-changed-up"
        elif direction == "down":
            flash_class = "value-changed-down"
        else:
            flash_class = "value-flash"

        # Apply flash class
        widget.add_class(flash_class)

        # Schedule removal using Textual's timer
        def remove_flash() -> None:
            try:
                widget.remove_class(flash_class)
            except Exception:
                pass  # Widget may have been removed

        try:
            self._flash_timers[selector] = self.set_timer(duration, remove_flash)
        except Exception:
            # Widget not mounted yet
            pass

    def flash_if_changed(
        self: Widget,
        selector: str,
        key: str,
        new_value: Any,
        duration: float = 0.5,
    ) -> bool:
        """Flash a widget if the value has changed from the cached value.

        Args:
            selector: CSS selector for the widget to flash.
            key: Cache key to track the value.
            new_value: The new value to compare.
            duration: Flash duration in seconds.

        Returns:
            True if the value changed and flash was triggered, False otherwise.
        """
        old_value = self._flash_cache.get(key)
        self._flash_cache[key] = new_value

        if old_value is None:
            # First value - no flash
            return False

        if old_value == new_value:
            return False

        # Determine direction for numeric values
        direction = "neutral"
        try:
            if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                direction = "up" if new_value > old_value else "down"
        except (TypeError, ValueError):
            pass

        self.flash_value(selector, direction, duration)
        return True


def flash_label(
    label: Label | Static,
    direction: str = "neutral",
    duration: float = 0.5,
) -> None:
    """Flash a label widget directly.

    Standalone function for cases where the mixin isn't used.
    Uses Textual's set_timer for proper cleanup.

    Args:
        label: The label widget to flash.
        direction: "up", "down", or "neutral".
        duration: Flash duration in seconds.
    """
    if direction == "up":
        flash_class = "value-changed-up"
    elif direction == "down":
        flash_class = "value-changed-down"
    else:
        flash_class = "value-flash"

    label.add_class(flash_class)

    def remove_flash() -> None:
        try:
            label.remove_class(flash_class)
        except Exception:
            pass

    # Use Textual's set_timer for proper integration with event loop
    try:
        label.set_timer(duration, remove_flash)
    except Exception:
        pass  # Widget may not be mounted
