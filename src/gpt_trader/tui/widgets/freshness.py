"""Data freshness indicator widget.

Shows "Last updated: Xs ago" with visual warnings for stale data.
Designed to be embedded in tile headers for at-a-glance data currency.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual.reactive import reactive
from textual.widgets import Label, Static

if TYPE_CHECKING:
    pass


class FreshnessIndicator(Static):
    """Compact indicator showing data freshness with stale warnings.

    Displays relative time since last update (e.g., "2s ago", "1m ago").
    Changes color when data becomes stale based on configurable threshold.

    Example usage:
        yield FreshnessIndicator(stale_threshold=10, id="market-freshness")
        # Later:
        self.query_one("#market-freshness").mark_updated()
    """

    SCOPED_CSS = False  # Use global styles

    # Reactive property for last update timestamp
    last_update: float = reactive(0.0)

    # Thresholds
    stale_threshold: int = 10  # Seconds before warning
    critical_threshold: int = 30  # Seconds before critical

    def __init__(
        self,
        stale_threshold: int = 10,
        critical_threshold: int = 30,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        base_classes = "freshness-indicator"
        merged_classes = f"{base_classes} {classes}" if classes else base_classes
        super().__init__(id=id, classes=merged_classes)
        self.stale_threshold = stale_threshold
        self.critical_threshold = critical_threshold
        self._update_timer = None

    def on_mount(self) -> None:
        """Start periodic refresh of relative time display."""
        self._update_timer = self.set_interval(1.0, self._refresh_display)

    def on_unmount(self) -> None:
        """Clean up timer."""
        if self._update_timer:
            self._update_timer.stop()

    def mark_updated(self, timestamp: float | None = None) -> None:
        """Mark data as freshly updated.

        Args:
            timestamp: Unix timestamp of update. If None, uses current time.
        """
        self.last_update = timestamp if timestamp is not None else time.time()

    def watch_last_update(self, timestamp: float) -> None:
        """React to timestamp changes."""
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Update the display with current relative time."""
        if self.last_update <= 0:
            self.update("[dim]--[/dim]")
            self.remove_class("fresh", "stale", "critical")
            return

        age = time.time() - self.last_update

        # Format relative time
        if age < 60:
            time_str = f"{int(age)}s"
        elif age < 3600:
            time_str = f"{int(age / 60)}m"
        else:
            time_str = f"{int(age / 3600)}h"

        # Determine severity and color
        self.remove_class("fresh", "stale", "critical")

        if age >= self.critical_threshold:
            self.add_class("critical")
            self.update(f"[red]{time_str} ago[/red]")
        elif age >= self.stale_threshold:
            self.add_class("stale")
            self.update(f"[yellow]{time_str} ago[/yellow]")
        else:
            self.add_class("fresh")
            self.update(f"[dim]{time_str} ago[/dim]")


class CompactFreshnessChip(Label):
    """Ultra-compact freshness indicator for tight spaces.

    Shows just the age (e.g., "5s", "2m") with color coding.
    Supports fresh/stale/critical states for consistency with FreshnessIndicator.
    """

    last_update: float = reactive(0.0)
    stale_threshold: int = 10
    critical_threshold: int = 30

    def __init__(
        self,
        stale_threshold: int = 10,
        critical_threshold: int = 30,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        base_classes = "timestamp-chip"
        merged_classes = f"{base_classes} {classes}" if classes else base_classes
        super().__init__("--", id=id, classes=merged_classes)
        self.stale_threshold = stale_threshold
        self.critical_threshold = critical_threshold
        self._update_timer = None

    def on_mount(self) -> None:
        """Start periodic refresh."""
        self._update_timer = self.set_interval(1.0, self._refresh)

    def on_unmount(self) -> None:
        """Clean up timer."""
        if self._update_timer:
            self._update_timer.stop()

    def mark_updated(self, timestamp: float | None = None) -> None:
        """Mark data as freshly updated."""
        self.last_update = timestamp if timestamp is not None else time.time()

    def watch_last_update(self, _timestamp: float) -> None:
        """React to timestamp changes."""
        self._refresh()

    def _refresh(self) -> None:
        """Update display."""
        if self.last_update <= 0:
            self.update("--")
            self.remove_class("stale", "fresh", "critical")
            return

        age = time.time() - self.last_update

        # Format compact time
        if age < 60:
            time_str = f"{int(age)}s"
        elif age < 3600:
            time_str = f"{int(age / 60)}m"
        else:
            time_str = f"{int(age / 3600)}h"

        # Determine severity and update styling
        self.remove_class("stale", "fresh", "critical")

        if age >= self.critical_threshold:
            self.add_class("critical")
            self.update(f"[red]{time_str}[/red]")
        elif age >= self.stale_threshold:
            self.add_class("stale")
            self.update(f"[yellow]{time_str}[/yellow]")
        else:
            self.add_class("fresh")
            self.update(f"[dim]{time_str}[/dim]")
