"""Loading spinner widget with multiple animation styles."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Static

# Standardized animation intervals for visual consistency
SPINNER_INTERVAL = 0.15  # Regular spinner animation (6-7 fps)


class LoadingSpinner(Static):
    """Animated loading spinner for data fetching states.

    Supports multiple spinner styles:
    - "dots": Unicode Braille pattern (default)
    - "bar": Progress bar animation
    - "pulse": Pulsing dot
    - "bounce": Bouncing dots

    Args:
        message: Primary loading message.
        subtitle: Optional secondary message with more detail.
        style: Spinner animation style.
    """

    SPINNER_STYLES = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "bar": ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", "▎"],
        "pulse": ["○", "◔", "◑", "◕", "●", "◕", "◑", "◔"],
        "bounce": ["⠁", "⠂", "⠄", "⠂"],
        "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    }

    def __init__(
        self,
        message: str = "Loading...",
        subtitle: str | None = None,
        style: str = "dots",
    ) -> None:
        super().__init__()
        self.message = message
        self.subtitle = subtitle
        self.style = style
        self.frame_index = 0
        self._frames = self.SPINNER_STYLES.get(style, self.SPINNER_STYLES["dots"])

    def compose(self) -> ComposeResult:
        with Vertical(classes="spinner-content"):
            yield Label(id="spinner-display", classes="loading-spinner")
            yield Label(self.message, id="spinner-message", classes="loading-message")
            if self.subtitle:
                yield Label(self.subtitle, id="spinner-subtitle", classes="loading-subtitle")

    def on_mount(self) -> None:
        """Start spinner animation."""
        self._timer = self.set_interval(SPINNER_INTERVAL, self._update_frame)

    def on_unmount(self) -> None:
        """Stop spinner animation."""
        if hasattr(self, "_timer") and self._timer:
            self._timer.stop()

    def _update_frame(self) -> None:
        """Advance spinner frame."""
        spinner_char = self._frames[self.frame_index]
        try:
            self.query_one("#spinner-display", Label).update(f"[cyan]{spinner_char}[/cyan]")
        except Exception:
            pass
        self.frame_index = (self.frame_index + 1) % len(self._frames)

    def update_message(self, message: str, subtitle: str | None = None) -> None:
        """Update the loading message dynamically.

        Args:
            message: New primary message.
            subtitle: Optional new subtitle.
        """
        self.message = message
        try:
            self.query_one("#spinner-message", Label).update(message)
            if subtitle is not None:
                self.subtitle = subtitle
                try:
                    self.query_one("#spinner-subtitle", Label).update(subtitle)
                except Exception:
                    pass
        except Exception:
            pass
