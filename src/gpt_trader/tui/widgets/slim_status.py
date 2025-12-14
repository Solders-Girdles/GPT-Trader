"""Slim status bar widget for log-centric TUI layout.

A single-line status bar showing essential bot metrics:
- Running state indicator
- Equity with P&L
- Position count
- Uptime
- Start/Stop controls
- Mode selector
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, Select, Static

from gpt_trader.tui.events import ResponsiveStateChanged
from gpt_trader.tui.responsive_state import ResponsiveState


class SlimStatusWidget(Static):
    """Single-line status bar for bot monitoring.

    Displays essential metrics in a compact horizontal layout:
    ● RUNNING | $10,234.56 (+$123.45) | Pos: 2 | 02:34:12 | [Start] [Stop] [Mode ▼]

    Uses cached widget references for efficient updates without repeated DOM queries.
    """

    # Use percentage-based sizing for children - height is set via global CSS ID selector
    SCOPED_CSS = False  # Disable scoping to allow nested selectors

    # Styles moved to styles/widgets/status_bar.tcss (SlimStatusWidget section)

    # Reactive properties
    running = reactive(False)
    uptime = reactive(0.0)
    equity = reactive("0.00")
    pnl = reactive("0.00")
    position_count = reactive(0)
    data_source_mode = reactive("demo")
    error_badge_count = reactive(0)

    # Responsive design property
    responsive_state = reactive(ResponsiveState.STANDARD)

    def on_responsive_state_changed(self, event: ResponsiveStateChanged) -> None:
        """Update layout when responsive state changes."""
        self.responsive_state = event.state

    # Cached widget references
    _status_indicator: Label | None = None
    _status_text: Label | None = None
    _equity_value: Label | None = None
    _pnl_value: Label | None = None
    _positions: Label | None = None
    _uptime_label: Label | None = None
    _start_btn: Button | None = None
    _stop_btn: Button | None = None
    _mode_select: Select | None = None
    _error_badge: Label | None = None

    # Mode options for selector
    MODE_OPTIONS = [
        ("Demo Mode", "demo"),
        ("Paper Trading", "paper"),
        ("Read Only", "read_only"),
        ("Live Trading", "live"),
    ]

    class ToggleBotPressed(Message):
        """Message sent when start/stop button is pressed."""

    class ModeChanged(Message):
        """Message sent when mode is changed."""

        def __init__(self, mode: str) -> None:
            self.mode = mode
            super().__init__()

    def compose(self) -> ComposeResult:
        """Compose single-line status bar with visual hierarchy."""
        with Horizontal(id="slim-status-bar"):
            # Running indicator (colored dot + text)
            yield Label("", id="status-indicator", classes="status-indicator status-stopped")
            yield Label("STOPPED", id="status-text", classes="status-text status-stopped")
            yield Static("│", classes="slim-separator")

            # Equity with label
            yield Label("Eq", classes="slim-label")
            yield Label("$0.00", id="equity-value", classes="slim-value")
            yield Static("│", classes="slim-separator")

            # P&L with label
            yield Label("P/L", classes="slim-label")
            yield Label("+$0.00", id="pnl-value", classes="slim-value pnl-neutral")
            yield Static("│", classes="slim-separator")

            # Position count
            yield Label("Pos", classes="slim-label")
            yield Label("0", id="positions-count", classes="slim-value positions-count")
            yield Static("│", classes="slim-separator")

            # Uptime
            yield Label("00:00:00", id="uptime", classes="uptime")
            yield Static("│", classes="slim-separator")

            # Control buttons
            yield Button("Start", id="slim-start-btn", variant="success", classes="slim-btn")
            yield Button(
                "Stop", id="slim-stop-btn", variant="error", disabled=True, classes="slim-btn"
            )
            yield Static("│", classes="slim-separator")

            # Mode selector (compact)
            yield Select(
                self.MODE_OPTIONS,
                id="slim-mode-select",
                value="demo",
                allow_blank=False,
            )

            # Error badge (hidden by default)
            yield Label("", id="error-badge", classes="error-badge hidden")

    def on_mount(self) -> None:
        """Cache widget references on mount for efficient updates."""
        self._status_indicator = self.query_one("#status-indicator", Label)
        self._status_text = self.query_one("#status-text", Label)
        self._equity_value = self.query_one("#equity-value", Label)
        self._pnl_value = self.query_one("#pnl-value", Label)
        self._positions = self.query_one("#positions-count", Label)
        self._uptime_label = self.query_one("#uptime", Label)
        self._start_btn = self.query_one("#slim-start-btn", Button)
        self._stop_btn = self.query_one("#slim-stop-btn", Button)
        self._mode_select = self.query_one("#slim-mode-select", Select)
        self._error_badge = self.query_one("#error-badge", Label)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id in ("slim-start-btn", "slim-stop-btn"):
            self.post_message(self.ToggleBotPressed())

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle mode selection changes."""
        if event.select.id == "slim-mode-select" and event.value:
            self.post_message(self.ModeChanged(str(event.value)))

    def watch_running(self, running: bool) -> None:
        """Update UI when bot running state changes."""
        indicator = self._status_indicator or self.query_one("#status-indicator", Label)
        text = self._status_text or self.query_one("#status-text", Label)
        start_btn = self._start_btn or self.query_one("#slim-start-btn", Button)
        stop_btn = self._stop_btn or self.query_one("#slim-stop-btn", Button)
        mode_select = self._mode_select

        if running:
            indicator.update("")
            indicator.remove_class("status-stopped")
            indicator.add_class("status-running")
            text.update("RUNNING")
            text.remove_class("status-stopped")
            text.add_class("status-running")
            start_btn.disabled = True
            stop_btn.disabled = False
            if mode_select:
                mode_select.disabled = True
        else:
            indicator.update("")
            indicator.remove_class("status-running")
            indicator.add_class("status-stopped")
            text.update("STOPPED")
            text.remove_class("status-running")
            text.add_class("status-stopped")
            start_btn.disabled = False
            stop_btn.disabled = True
            if mode_select:
                mode_select.disabled = False

    def watch_uptime(self, uptime: float) -> None:
        """Update uptime display."""
        m, s = divmod(int(uptime), 60)
        h, m = divmod(m, 60)
        label = self._uptime_label or self.query_one("#uptime", Label)
        label.update(f"{h:02d}:{m:02d}:{s:02d}")

    def watch_equity(self, equity: str) -> None:
        """Update equity display."""
        label = self._equity_value or self.query_one("#equity-value", Label)
        label.update(f"${equity}")

    def watch_pnl(self, pnl: str) -> None:
        """Update P&L display with color coding."""
        label = self._pnl_value or self.query_one("#pnl-value", Label)

        try:
            pnl_float = float(pnl)
            if pnl_float >= 0:
                label.update(f"+${pnl}")
            else:
                label.update(f"-${abs(pnl_float):.2f}")

            # Color code based on P&L
            label.remove_class("pnl-positive", "pnl-negative", "pnl-neutral")
            if pnl_float > 0:
                label.add_class("pnl-positive")
            elif pnl_float < 0:
                label.add_class("pnl-negative")
            else:
                label.add_class("pnl-neutral")
        except ValueError:
            label.update(f"${pnl}")

    def watch_position_count(self, count: int) -> None:
        """Update position count display."""
        label = self._positions or self.query_one("#positions-count", Label)
        if count == 0:
            label.update("No Pos")
            label.remove_class("has-positions")
        else:
            label.update(f"Pos: {count}")
            label.add_class("has-positions")

    def watch_data_source_mode(self, mode: str) -> None:
        """Update mode selector to reflect current mode."""
        select = self._mode_select
        if select and select.value != mode:
            select.value = mode

    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Adjust layout based on terminal width."""
        # In compact mode, abbreviate labels
        if state == ResponsiveState.COMPACT:
            # Could hide uptime or use shorter labels
            pass
        # Future: implement responsive adjustments

    def watch_error_badge_count(self, count: int) -> None:
        """Update error badge visibility and count."""
        badge = self._error_badge or self.query_one("#error-badge", Label)
        if count == 0:
            badge.add_class("hidden")
            badge.update("")
        else:
            badge.remove_class("hidden")
            badge.update(f"⚠{count}")
