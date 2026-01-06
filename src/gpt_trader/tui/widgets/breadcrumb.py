"""Breadcrumb navigation widget for screen hierarchy awareness.

Shows the current navigation path and allows quick navigation back to
parent screens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Label, Static

if TYPE_CHECKING:
    pass


class BreadcrumbWidget(Static):
    """Breadcrumb navigation showing screen hierarchy.

    Displays a path like: Dashboard > Position Details

    The breadcrumb automatically integrates with the screen stack,
    or can be configured manually with a path list.

    Example:
        yield BreadcrumbWidget(path=["Dashboard", "Position Details"])
    """

    SCOPED_CSS = False

    # The navigation path as a list of screen names
    path: list[str] = reactive([], always_update=True)  # type: ignore[var-annotated]

    def __init__(
        self,
        path: list[str] | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize breadcrumb.

        Args:
            path: Navigation path as list of screen names.
            id: Widget ID.
            classes: Additional CSS classes.
        """
        base_classes = "breadcrumb-widget"
        merged_classes = f"{base_classes} {classes}" if classes else base_classes
        super().__init__(id=id, classes=merged_classes)
        self._path = path or []

    def compose(self) -> ComposeResult:
        with Horizontal(classes="breadcrumb-container"):
            yield Label("", id="breadcrumb-content", classes="breadcrumb-content")

    def on_mount(self) -> None:
        """Update display on mount."""
        self._render_path()

    def watch_path(self, new_path: list[str]) -> None:
        """React to path changes."""
        self._path = new_path
        self._render_path()

    def set_path(self, path: list[str]) -> None:
        """Set the navigation path.

        Args:
            path: List of screen names from root to current.
        """
        self._path = path
        self._render_path()

    def push(self, name: str) -> None:
        """Add a screen to the path.

        Args:
            name: Screen name to add.
        """
        self._path.append(name)
        self._render_path()

    def pop(self) -> str | None:
        """Remove the last screen from path.

        Returns:
            The removed screen name, or None if path was empty.
        """
        if self._path:
            removed = self._path.pop()
            self._render_path()
            return removed
        return None

    def _render_path(self) -> None:
        """Render the breadcrumb path."""
        try:
            label = self.query_one("#breadcrumb-content", Label)
            if not self._path:
                label.update("")
                return

            # Build breadcrumb string with separators
            parts = []
            for i, name in enumerate(self._path):
                if i < len(self._path) - 1:
                    # Parent screens - dim color
                    parts.append(f"[dim]{name}[/dim]")
                else:
                    # Current screen - accent color
                    parts.append(f"[cyan]{name}[/cyan]")

            # Join with arrow separator
            breadcrumb_str = " [dim]›[/dim] ".join(parts)
            label.update(breadcrumb_str)
        except Exception:
            pass


class ScreenBreadcrumb(BreadcrumbWidget):
    """Breadcrumb that auto-detects path from screen stack.

    Automatically reads the app's screen stack to build the path.
    Optionally takes a current screen name to append.

    Example:
        yield ScreenBreadcrumb(current="Position Details")
    """

    def __init__(
        self,
        current: str | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize screen-aware breadcrumb.

        Args:
            current: Name of the current screen to display.
            id: Widget ID.
            classes: Additional CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._current = current

    def on_mount(self) -> None:
        """Build path from screen stack."""
        path = self._build_path_from_stack()
        if self._current:
            path.append(self._current)
        self._path = path
        self._render_path()

    def _build_path_from_stack(self) -> list[str]:
        """Build breadcrumb path from app's screen stack.

        Returns:
            List of screen names from root to parent.
        """
        path = []
        try:
            # Get screen stack from app
            if hasattr(self.app, "screen_stack"):
                for screen in self.app.screen_stack:
                    # Get screen title or class name
                    name = getattr(screen, "TITLE", None)
                    if not name:
                        name = screen.__class__.__name__
                        # Remove "Screen" suffix for cleaner display
                        if name.endswith("Screen"):
                            name = name[:-6]
                        # Add spaces before capitals for readability
                        name = "".join(f" {c}" if c.isupper() else c for c in name).strip()
                    path.append(name)
        except Exception:
            pass
        return path
