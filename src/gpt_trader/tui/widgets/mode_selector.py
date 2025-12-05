"""Mode selector dropdown widget."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Label, Select, Static


class ModeSelector(Static):
    """Dropdown selector for bot operating mode."""

    # Reactive properties
    current_mode = reactive("demo")
    enabled = reactive(True)

    # Mode configurations (prompt, value) tuples for Select widget
    MODES = [
        ("DEMO - Mock Data", "demo"),
        ("PAPER - Real Data", "paper"),
        ("OBSERVE - Read Only", "read_only"),
        ("⚠ LIVE TRADING ⚠", "live"),
    ]

    class ModeChanged(Message):
        """Posted when user selects a new mode."""

        def __init__(self, new_mode: str) -> None:
            self.new_mode = new_mode
            super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the mode selector layout."""
        with Horizontal(id="mode-selector-container"):
            yield Label("Mode:", classes="mode-selector-label")
            yield Select(
                self.MODES,
                value=self.current_mode,
                id="mode-select",
                allow_blank=False,
            )

    def on_mount(self) -> None:
        """Initialize selector with current mode."""
        select = self.query_one("#mode-select", Select)
        select.value = self.current_mode

    def watch_enabled(self, enabled: bool) -> None:
        """Update selector enabled/disabled state."""
        try:
            select = self.query_one("#mode-select", Select)
            select.disabled = not enabled
        except Exception:
            pass  # Widget might not be mounted yet

    def watch_current_mode(self, mode: str) -> None:
        """Update selector when mode changes externally."""
        try:
            select = self.query_one("#mode-select", Select)
            if select.value != mode:
                select.value = mode
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle mode selection change."""
        from textual.widgets._select import NoSelection

        if (
            event.select.id == "mode-select"
            and event.value is not None
            and not isinstance(event.value, NoSelection)
        ):
            new_mode = str(event.value)
            if new_mode != self.current_mode:
                # Post message to app for handling
                self.post_message(self.ModeChanged(new_mode))
