from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

from gpt_trader.tui.events import ResponsiveStateChanged
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.widgets.mode_indicator import ModeIndicator
from gpt_trader.tui.widgets.mode_selector import ModeSelector


class BotStatusWidget(Static):
    """Displays high-level bot status with interactive controls.

    Uses cached widget references for efficient updates without repeated DOM queries.
    """

    uptime = reactive(0.0)
    running = reactive(False)
    equity = reactive("0.00")
    pnl = reactive("0.00")
    margin_usage = reactive("0.00%")
    heartbeat = reactive(0.0)  # Smooth pulse (0.0-1.0 via sine wave)

    # System health metrics
    connection_status = reactive("UNKNOWN")
    api_latency = reactive(0.0)
    cpu_usage = reactive("0%")
    rate_limit_usage = reactive("0%")

    # Responsive design property
    responsive_state = reactive(ResponsiveState.STANDARD)

    def on_responsive_state_changed(self, event: ResponsiveStateChanged) -> None:
        """Update layout when responsive state changes."""
        self.responsive_state = event.state

    # Cached widget references (populated in on_mount)
    _status_label: Label | None = None
    _start_btn: Button | None = None
    _stop_btn: Button | None = None
    _uptime_label: Label | None = None
    _equity_label: Label | None = None
    _pnl_label: Label | None = None
    _margin_label: Label | None = None
    _heartbeat_indicator: Label | None = None
    _conn_indicator: Label | None = None
    _latency_label: Label | None = None
    _cpu_label: Label | None = None
    _rate_limit_label: Label | None = None

    class ToggleBotPressed(Message):
        """Message sent when start/stop button is pressed."""

    def compose(self) -> ComposeResult:
        # Flattened status bar - all direct children of single Horizontal
        with Horizontal(id="status-bar-container"):
            # Mode controls
            yield ModeIndicator(id="mode-indicator")
            yield ModeSelector(id="mode-selector")
            yield Static("|", classes="separator")

            # Bot control buttons
            yield Button("Start", id="start-btn", variant="success")
            yield Button("Stop", id="stop-btn", variant="primary", disabled=True)
            yield Static("|", classes="separator")

            # Status
            yield Label("Status:", classes="status-label")
            yield Label("STOPPED", id="status-value", classes="status-value status-stopped")
            yield Label("", id="heartbeat-indicator", classes="heartbeat-pulse")
            yield Static("|", classes="separator")

            # Uptime
            yield Label("Uptime:", classes="status-label uptime-group")
            yield Label("00:00:00", id="uptime-value", classes="status-value uptime-group")
            yield Static("|", classes="separator uptime-group")

            # Performance metrics
            yield Label("Equity:", classes="status-label")
            yield Label("$0.00", id="equity-value", classes="status-value")
            yield Label("P&L:", classes="status-label")
            yield Label("$0.00", id="pnl-value", classes="status-value pnl-neutral")
            yield Label("Margin:", id="margin-label", classes="status-label margin-group")
            yield Label("0.00%", id="margin-value", classes="status-value margin-group")
            yield Static("|", classes="separator system-health-group")

            # System health
            yield Label("", id="sys-conn-indicator", classes="status-unknown system-health-group")
            yield Label("API:", classes="status-label system-health-group")
            yield Label("0ms", id="sys-latency", classes="status-value system-health-group")
            yield Label("CPU:", classes="status-label system-health-group")
            yield Label("0%", id="sys-cpu", classes="status-value system-health-group")
            yield Label("Rate:", classes="status-label system-health-group")
            yield Label("0%", id="sys-rate-limit", classes="status-value system-health-group")

    def on_mount(self) -> None:
        """Cache widget references on mount for efficient updates."""
        self._status_label = self.query_one("#status-value", Label)
        self._start_btn = self.query_one("#start-btn", Button)
        self._stop_btn = self.query_one("#stop-btn", Button)
        self._uptime_label = self.query_one("#uptime-value", Label)
        self._equity_label = self.query_one("#equity-value", Label)
        self._pnl_label = self.query_one("#pnl-value", Label)
        self._margin_label = self.query_one("#margin-value", Label)
        self._heartbeat_indicator = self.query_one("#heartbeat-indicator", Label)
        self._conn_indicator = self.query_one("#sys-conn-indicator", Label)
        self._latency_label = self.query_one("#sys-latency", Label)
        self._cpu_label = self.query_one("#sys-cpu", Label)
        self._rate_limit_label = self.query_one("#sys-rate-limit", Label)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id in ("start-btn", "stop-btn"):
            self.post_message(self.ToggleBotPressed())

    @safe_update
    def watch_running(self, running: bool) -> None:
        """Update UI when bot running state changes."""
        _last = getattr(self, "_last_running", object())
        if running == _last:
            return
        self._last_running = running

        # Use cached references (fall back to query if not cached yet)
        label = self._status_label or self.query_one("#status-value", Label)
        start_btn = self._start_btn or self.query_one("#start-btn", Button)
        stop_btn = self._stop_btn or self.query_one("#stop-btn", Button)

        if running:
            label.update("RUNNING")
            label.remove_class("status-stopped")
            label.add_class("status-running")
            start_btn.disabled = True
            stop_btn.disabled = False
        else:
            label.update("STOPPED")
            label.remove_class("status-running")
            label.add_class("status-stopped")
            start_btn.disabled = False
            stop_btn.disabled = True

    @safe_update
    def watch_uptime(self, uptime: float) -> None:
        """Update uptime display."""
        _last = getattr(self, "_last_uptime", object())
        if uptime == _last:
            return
        self._last_uptime = uptime

        m, s = divmod(int(uptime), 60)
        h, m = divmod(m, 60)
        label = self._uptime_label or self.query_one("#uptime-value", Label)
        label.update(f"{h:02d}:{m:02d}:{s:02d}")

    @safe_update
    def watch_equity(self, equity: str) -> None:
        """Update equity display."""
        _last = getattr(self, "_last_equity", object())
        if equity == _last:
            return
        self._last_equity = equity

        label = self._equity_label or self.query_one("#equity-value", Label)
        label.update(f"${equity}")
        self._flash_value(label)

    @safe_update
    def watch_pnl(self, pnl: str) -> None:
        """Update P&L display with color coding."""
        _last = getattr(self, "_last_pnl", object())
        if pnl == _last:
            return
        self._last_pnl = pnl

        pnl_label = self._pnl_label or self.query_one("#pnl-value", Label)
        pnl_label.update(f"${pnl}")

        # Color code based on value
        try:
            pnl_float = float(pnl)
            pnl_label.remove_class("pnl-positive", "pnl-negative", "pnl-neutral")
            if pnl_float > 0:
                pnl_label.add_class("pnl-positive")
            elif pnl_float < 0:
                pnl_label.add_class("pnl-negative")
            else:
                pnl_label.add_class("pnl-neutral")
        except ValueError:
            pass

        self._flash_value(pnl_label)

    @safe_update
    def watch_margin_usage(self, margin: str) -> None:
        """Update margin usage display."""
        _last = getattr(self, "_last_margin_usage", object())
        if margin == _last:
            return
        self._last_margin_usage = margin

        label = self._margin_label or self.query_one("#margin-value", Label)
        label.update(f"{margin}")
        self._flash_value(label)

    def _flash_value(self, label: Label) -> None:
        """
        Apply a brief flash animation to a value label.

        Adds the 'value-updating' class which triggers a flash effect,
        then removes it after 300ms to return to normal styling.

        Only applies animation if widget is mounted and has an active app.
        """
        # Only animate if widget is mounted and has an event loop
        if not self.is_mounted or not self.app:
            return

        label.add_class("value-updating")
        try:
            self.set_timer(0.3, lambda: label.remove_class("value-updating"))
        except RuntimeError:
            # No event loop available (e.g., in tests), skip animation
            label.remove_class("value-updating")

    @safe_update
    def watch_heartbeat(self, heartbeat: float) -> None:
        """Update heartbeat with smooth opacity transition."""
        _last = getattr(self, "_last_heartbeat", object())
        if heartbeat == _last:
            return
        self._last_heartbeat = heartbeat

        indicator = self._heartbeat_indicator
        if not indicator:
            return

        # Map 0.0-1.0 to opacity via sine wave
        # Range: 0.3 (dim) to 1.0 (bright)
        opacity = 0.3 + (0.7 * heartbeat)
        indicator.styles.opacity = opacity

    @safe_update
    def watch_connection_status(self, status: str) -> None:
        """Update connection status display via indicator icon."""
        _last = getattr(self, "_last_connection_status", object())
        if status == _last:
            return
        self._last_connection_status = status

        indicator = self._conn_indicator
        if not indicator:
            return

        # Remove all status classes and add appropriate one
        indicator.remove_class("status-connected", "status-disconnected", "status-unknown")

        if status == "CONNECTED":
            indicator.update("")
            indicator.add_class("status-connected")
        elif status == "DISCONNECTED":
            indicator.update("")
            indicator.add_class("status-disconnected")
        else:
            indicator.update("")
            indicator.add_class("status-unknown")

    @safe_update
    def watch_api_latency(self, latency: float) -> None:
        """Update API latency display."""
        _last = getattr(self, "_last_api_latency", object())
        if latency == _last:
            return
        self._last_api_latency = latency

        label = self._latency_label
        if label:
            label.update(f"{latency:.0f}ms")

    @safe_update
    def watch_cpu_usage(self, cpu: str) -> None:
        """Update CPU usage display."""
        _last = getattr(self, "_last_cpu_usage", object())
        if cpu == _last:
            return
        self._last_cpu_usage = cpu

        label = self._cpu_label
        if label:
            label.update(cpu)

    @safe_update
    def watch_rate_limit_usage(self, rate_limit: str) -> None:
        """Update rate limit usage display."""
        _last = getattr(self, "_last_rate_limit_usage", object())
        if rate_limit == _last:
            return
        self._last_rate_limit_usage = rate_limit

        label = self._rate_limit_label
        if label:
            label.update(rate_limit)

    @safe_update
    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Update status bar layout based on responsive state.

        Adjusts visibility and labels to optimize space usage across
        different terminal widths. Uses CSS classes for hiding elements
        to avoid layout conflicts with Python display toggles.

        Args:
            state: ResponsiveState enum value

        Breakpoint behaviors:
            - COMPACT (100-119): Icon buttons, hide system health, margin & uptime
            - STANDARD (120-139): Short labels, hide system health & margin
            - COMFORTABLE (140-159): Full labels, all metrics visible
            - WIDE (160+): Expanded labels
        """
        _last = getattr(self, "_last_responsive_state", object())
        if state == _last:
            return
        self._last_responsive_state = state

        # Use cached references (fall back to query if not cached yet)
        start_btn = self._start_btn
        stop_btn = self._stop_btn

        if not start_btn or not stop_btn:
            return  # Not mounted yet

        # Update button labels based on state
        if state in (ResponsiveState.COMPACT, ResponsiveState.STANDARD):
            start_btn.label = "Start"
            stop_btn.label = "Stop"
        else:  # COMFORTABLE or WIDE
            start_btn.label = "Start Bot"
            stop_btn.label = "Stop Bot"

        # Use CSS classes for visibility toggling
        hide_system_health = state in (ResponsiveState.COMPACT, ResponsiveState.STANDARD)
        hide_margin = state in (ResponsiveState.COMPACT, ResponsiveState.STANDARD)
        hide_uptime = state == ResponsiveState.COMPACT

        # Toggle visibility via CSS class (batch update)
        for elem in self.query(".system-health-group"):
            elem.set_class(hide_system_health, "hidden")

        for elem in self.query(".margin-group"):
            elem.set_class(hide_margin, "hidden")

        for elem in self.query(".uptime-group"):
            elem.set_class(hide_uptime, "hidden")
