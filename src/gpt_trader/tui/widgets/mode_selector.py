"""Mode selector dropdown widget."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Label, Select, Static
from textual.widgets._select import SelectCurrent

from gpt_trader.tui.events import (
    ModeSelectorEnabledChanged,
    ModeSelectorLoadingChanged,
    ModeSelectorValueChanged,
)


class ModeSelector(Static):
    """Dropdown selector for bot operating mode."""

    # Styles moved to styles/widgets/status_bar.tcss (ModeSelector section)

    # Loading spinner frames (Braille dots)
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    # Reactive properties
    current_mode: str = reactive("demo")
    enabled: bool = reactive(True)
    loading: bool = reactive(False)

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
            select = Select(
                self.MODES,
                value=self.current_mode,
                id="mode-select",
                allow_blank=False,
            )
            select.can_focus = True
            yield select
            yield Label("", id="mode-loading", classes="loading-indicator")

    def on_mount(self) -> None:
        """Initialize selector with current mode."""
        select = self.query_one("#mode-select", Select)
        select.value = self.current_mode
        try:
            # Force the current label to render for debugging (some terminals clip styles)
            select_current = select.query_one(SelectCurrent)
            select_current.update("DEMO")
        except Exception:
            pass
        self._spinner_frame = 0
        self._spinner_timer = None

    def watch_loading(self, loading: bool) -> None:
        """Update loading state - show/hide spinner and disable selector."""
        try:
            loading_label = self.query_one("#mode-loading", Label)
            select = self.query_one("#mode-select", Select)

            if loading:
                # Disable selector and start spinner animation
                select.disabled = True
                loading_label.update(self.SPINNER_FRAMES[0])
                self._spinner_frame = 0
                self._spinner_timer = self.set_interval(0.1, self._animate_spinner)
            else:
                # Re-enable selector (unless externally disabled) and stop spinner
                if self.enabled:
                    select.disabled = False
                loading_label.update("")
                if self._spinner_timer:
                    self._spinner_timer.stop()
                    self._spinner_timer = None
        except Exception:
            pass  # Widget might not be mounted yet

    def _animate_spinner(self) -> None:
        """Advance spinner animation frame."""
        try:
            loading_label = self.query_one("#mode-loading", Label)
            self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_FRAMES)
            loading_label.update(self.SPINNER_FRAMES[self._spinner_frame])
        except Exception:
            pass

    def watch_enabled(self, enabled: bool) -> None:
        """Update selector enabled/disabled state."""
        try:
            select = self.query_one("#mode-select", Select)
            # Don't enable if currently loading
            if self.loading:
                select.disabled = True
            else:
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

    def on_mode_selector_enabled_changed(self, event: ModeSelectorEnabledChanged) -> None:
        """Handle enable/disable request from event system.

        This handler replaces direct property access from BotLifecycleManager.
        """
        self.enabled = event.enabled

    def on_mode_selector_value_changed(self, event: ModeSelectorValueChanged) -> None:
        """Handle mode value change request from event system.

        This handler replaces direct property access from BotLifecycleManager.
        """
        self.current_mode = event.mode

    def on_mode_selector_loading_changed(self, event: ModeSelectorLoadingChanged) -> None:
        """Handle loading state change request from event system.

        Shows/hides spinner indicator during mode switching operations.
        """
        self.loading = event.loading
