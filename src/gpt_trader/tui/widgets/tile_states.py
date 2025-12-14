"""Reusable tile state components for dashboard-style widgets.

These helpers provide consistent loading, empty, and banner states without
inline CSS or hard-coded colors. Styling lives in TCSS.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.widgets.spinner import LoadingSpinner


class TileBanner(Static):
    """Inline banner for tiles (info/warning/error)."""

    text = reactive("")
    severity = reactive("info")  # info, warning, error

    def __init__(
        self,
        text: str = "",
        severity: str = "info",
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.text = text
        self.severity = severity

    def compose(self) -> ComposeResult:
        yield Label("", id="tile-banner-label")

    def on_mount(self) -> None:
        self._apply_state()

    def watch_text(self, _text: str) -> None:
        self._apply_state()

    def watch_severity(self, _severity: str) -> None:
        self._apply_state()

    def update_banner(self, text: str, severity: str = "info") -> None:
        """Update banner text/severity and visibility."""
        self.text = text
        self.severity = severity

    def _apply_state(self) -> None:
        if not self.text:
            self.add_class("hidden")
        else:
            self.remove_class("hidden")

        for sev in ("info", "warning", "error"):
            self.remove_class(sev)
        self.add_class((self.severity or "info").lower())

        try:
            self.query_one("#tile-banner-label", Label).update(self.text)
        except Exception:
            pass


class TileEmptyState(Vertical):
    """Centered empty-state view for tiles."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        icon: str = "○",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        base_classes = "empty-state-container"
        merged_classes = f"{base_classes} {classes}" if classes else base_classes
        super().__init__(id=id, classes=merged_classes)
        self._title = title
        self._subtitle = subtitle
        self._icon = icon

    def compose(self) -> ComposeResult:
        yield Label(self._icon, classes="empty-icon")
        yield Label(self._title, classes="empty-title")
        yield Label(self._subtitle, classes="empty-subtitle")


class TileLoadingState(Vertical):
    """Centered loading/skeleton view for tiles."""

    def __init__(
        self,
        message: str = "Loading...",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        base_classes = "empty-state-container loading-container"
        merged_classes = f"{base_classes} {classes}" if classes else base_classes
        super().__init__(id=id, classes=merged_classes)
        self._message = message

    def compose(self) -> ComposeResult:
        yield LoadingSpinner(message=self._message)


def tile_empty_state(title: str, subtitle: str, icon: str = "○") -> TileEmptyState:
    """Convenience factory for empty states."""
    return TileEmptyState(title=title, subtitle=subtitle, icon=icon)


def tile_loading_state(message: str = "Loading...") -> TileLoadingState:
    """Convenience factory for loading states."""
    return TileLoadingState(message=message)
