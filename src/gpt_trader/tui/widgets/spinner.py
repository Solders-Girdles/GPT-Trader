"""Loading spinner widget."""

from textual.app import ComposeResult
from textual.widgets import Label, Static


class LoadingSpinner(Static):
    """
    Animated loading spinner for data fetching states.

    Uses Unicode spinner characters: ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
    """

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Loading...") -> None:
        super().__init__()
        self.message = message
        self.frame_index = 0

    def compose(self) -> ComposeResult:
        yield Label(id="spinner-display", classes="loading-spinner")
        yield Label(self.message, id="spinner-message", classes="loading-message")

    def on_mount(self) -> None:
        """Start spinner animation."""
        self.set_interval(0.1, self._update_frame)

    def _update_frame(self) -> None:
        """Advance spinner frame."""
        spinner_char = self.SPINNER_FRAMES[self.frame_index]
        self.query_one("#spinner-display", Label).update(spinner_char)
        self.frame_index = (self.frame_index + 1) % len(self.SPINNER_FRAMES)
