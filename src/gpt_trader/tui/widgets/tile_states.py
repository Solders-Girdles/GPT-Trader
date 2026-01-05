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
    """Centered empty-state view for tiles with optional action hints.

    Supports dynamic updates via update_state() or individual property setters.
    """

    # Reactive properties for dynamic updates
    title = reactive("")
    subtitle = reactive("")
    icon = reactive("○")

    def __init__(
        self,
        title: str,
        subtitle: str,
        icon: str = "○",
        *,
        actions: list[str] | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Create an empty state view.

        Args:
            title: Main title text.
            subtitle: Descriptive subtitle.
            icon: Unicode icon character.
            actions: List of action hints like "[S] Start", "[R] Refresh".
            id: Widget ID.
            classes: Additional CSS classes.
        """
        base_classes = "empty-state-container"
        merged_classes = f"{base_classes} {classes}" if classes else base_classes
        super().__init__(id=id, classes=merged_classes)
        self._actions = actions or []
        # Set reactive properties after super().__init__
        self.title = title
        self.subtitle = subtitle
        self.icon = icon

    def compose(self) -> ComposeResult:
        yield Label(self.icon, id="empty-icon", classes="empty-icon")
        yield Label(self.title, id="empty-title", classes="empty-title")
        yield Label(self.subtitle, id="empty-subtitle", classes="empty-subtitle")

        # Add action hints container (can be updated later)
        from textual.containers import Horizontal

        with Horizontal(id="empty-actions-container", classes="empty-actions") as container:
            for action in self._actions:
                yield Label(action, classes="action-hint")

    def on_mount(self) -> None:
        """Hide actions container if empty on initial mount."""
        if not self._actions:
            try:
                from textual.containers import Horizontal

                container = self.query_one("#empty-actions-container", Horizontal)
                container.display = False
            except Exception:
                pass

    def watch_title(self, new_title: str) -> None:
        """Update title label when reactive property changes."""
        try:
            self.query_one("#empty-title", Label).update(new_title)
        except Exception:
            pass

    def watch_subtitle(self, new_subtitle: str) -> None:
        """Update subtitle label when reactive property changes."""
        try:
            self.query_one("#empty-subtitle", Label).update(new_subtitle)
        except Exception:
            pass

    def watch_icon(self, new_icon: str) -> None:
        """Update icon label when reactive property changes."""
        try:
            self.query_one("#empty-icon", Label).update(new_icon)
        except Exception:
            pass

    def update_state(
        self,
        title: str | None = None,
        subtitle: str | None = None,
        icon: str | None = None,
        actions: list[str] | None = None,
    ) -> None:
        """Update empty state content dynamically.

        Args:
            title: New title text (None to keep current).
            subtitle: New subtitle text (None to keep current).
            icon: New icon character (None to keep current).
            actions: New action hints (None to keep current).
        """
        if title is not None:
            self.title = title
        if subtitle is not None:
            self.subtitle = subtitle
        if icon is not None:
            self.icon = icon
        if actions is not None:
            self._update_actions(actions)

    def _update_actions(self, actions: list[str]) -> None:
        """Update action hints dynamically."""
        from textual.containers import Horizontal

        try:
            container = self.query_one("#empty-actions-container", Horizontal)
            container.remove_children()

            if actions:
                container.display = True
                for action in actions:
                    container.mount(Label(action, classes="action-hint"))
            else:
                container.display = False

            self._actions = actions
        except Exception:
            pass


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


def tile_empty_state(
    title: str,
    subtitle: str,
    icon: str = "○",
    *,
    actions: list[str] | None = None,
) -> TileEmptyState:
    """Convenience factory for empty states.

    Args:
        title: Main title text.
        subtitle: Descriptive subtitle.
        icon: Unicode icon character.
        actions: Optional list of action hints like "[S] Start", "[R] Refresh".

    Returns:
        A TileEmptyState widget.
    """
    return TileEmptyState(title=title, subtitle=subtitle, icon=icon, actions=actions)


def tile_loading_state(message: str = "Loading...") -> TileLoadingState:
    """Convenience factory for loading states."""
    return TileLoadingState(message=message)
