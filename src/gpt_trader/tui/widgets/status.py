from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

from gpt_trader.tui.widgets.mode_indicator import ModeIndicator
from gpt_trader.tui.widgets.mode_selector import ModeSelector


class BotStatusWidget(Static):
    """Displays high-level bot status with interactive controls."""

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

    # Responsive design property
    responsive_state = reactive("standard")

    class ToggleBotPressed(Message):
        """Message sent when start/stop button is pressed."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="status-bar-container"):
            # Left side: Mode controls + Bot control buttons
            with Container(id="bot-controls", classes="status-section"):
                # Mode controls (compact horizontal layout)
                with Horizontal(classes="mode-controls-compact"):
                    yield ModeIndicator(id="mode-indicator")
                    yield ModeSelector(id="mode-selector")

                # Bot control buttons
                with Horizontal(classes="button-group"):
                    yield Button("▶ Start", id="start-btn", variant="success")
                    yield Button("⏹ Stop", id="stop-btn", variant="primary", disabled=True)

            # Center: Status info (single row)
            with Container(id="status-info", classes="status-section"):
                with Horizontal(classes="status-row"):
                    yield Label("Status:", classes="status-label")
                    yield Label("STOPPED", id="status-value", classes="status-value status-stopped")
                    yield Label("●", id="heartbeat-indicator", classes="heartbeat-pulse")
                    yield Label("Uptime:", classes="status-label")
                    yield Label("00:00:00", id="uptime-value", classes="status-value")

            # Right side: Performance metrics (single row, cycles removed)
            with Container(id="performance-metrics", classes="status-section"):
                with Horizontal(classes="status-row"):
                    yield Label("Equity:", classes="status-label")
                    yield Label("$0.00", id="equity-value", classes="status-value")
                    yield Label("P&L:", classes="status-label")
                    yield Label("$0.00", id="pnl-value", classes="status-value pnl-neutral")
                    yield Label("Margin:", classes="status-label")
                    yield Label("0.00%", id="margin-value", classes="status-value")

            # Far right: System health (single row)
            with Container(id="system-health-section", classes="status-section"):
                with Horizontal(classes="status-row"):
                    yield Label("●", id="sys-conn-indicator", classes="status-unknown")
                    yield Label(
                        "UNKNOWN", id="sys-connection-status", classes="status-value status-unknown"
                    )
                    yield Label("API:", classes="status-label")
                    yield Label("0ms", id="sys-latency", classes="status-value")
                    yield Label("CPU:", classes="status-label")
                    yield Label("0%", id="sys-cpu", classes="status-value")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id in ("start-btn", "stop-btn"):
            self.post_message(self.ToggleBotPressed())

    def watch_running(self, running: bool) -> None:
        """Update UI when bot running state changes."""
        # Update status label
        label = self.query_one("#status-value", Label)
        if running:
            label.update("RUNNING")
            label.remove_class("status-stopped")
            label.add_class("status-running")
        else:
            label.update("STOPPED")
            label.remove_class("status-running")
            label.add_class("status-stopped")

        # Update button states
        start_btn = self.query_one("#start-btn", Button)
        stop_btn = self.query_one("#stop-btn", Button)

        if running:
            start_btn.disabled = True
            stop_btn.disabled = False
        else:
            start_btn.disabled = False
            stop_btn.disabled = True

    def watch_uptime(self, uptime: float) -> None:
        """Update uptime display."""
        m, s = divmod(int(uptime), 60)
        h, m = divmod(m, 60)
        self.query_one("#uptime-value", Label).update(f"{h:02d}:{m:02d}:{s:02d}")

    def watch_equity(self, equity: str) -> None:
        """Update equity display."""
        label = self.query_one("#equity-value", Label)
        label.update(f"${equity}")
        self._flash_value(label)

    def watch_pnl(self, pnl: str) -> None:
        """Update P&L display with color coding."""
        pnl_label = self.query_one("#pnl-value", Label)
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

    def watch_margin_usage(self, margin: str) -> None:
        """Update margin usage display."""
        label = self.query_one("#margin-value", Label)
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

    def watch_heartbeat(self, heartbeat: float) -> None:
        """Update heartbeat with smooth opacity transition."""
        indicator = self.query_one("#heartbeat-indicator", Label)

        # Map 0.0-1.0 to opacity via sine wave
        # Range: 0.3 (dim) to 1.0 (bright)
        opacity = 0.3 + (0.7 * heartbeat)

        # Update CSS with inline opacity
        indicator.styles.opacity = opacity

    def watch_connection_status(self, status: str) -> None:
        """Update connection status display."""
        try:
            status_label = self.query_one("#sys-connection-status", Label)
            indicator = self.query_one("#sys-conn-indicator", Label)

            status_label.update(status)

            # Remove all status classes
            for cls in ["status-connected", "status-disconnected", "status-unknown"]:
                status_label.remove_class(cls)
                indicator.remove_class(cls)

            # Add appropriate class
            if status == "CONNECTED":
                status_label.add_class("status-connected")
                indicator.add_class("status-connected")
            elif status == "DISCONNECTED":
                status_label.add_class("status-disconnected")
                indicator.add_class("status-disconnected")
            else:
                status_label.add_class("status-unknown")
                indicator.add_class("status-unknown")
        except Exception:
            pass  # Widget not mounted yet

    def watch_api_latency(self, latency: float) -> None:
        """Update API latency display."""
        try:
            label = self.query_one("#sys-latency", Label)
            label.update(f"{latency:.0f}ms")
        except Exception:
            pass  # Widget not mounted yet

    def watch_cpu_usage(self, cpu: str) -> None:
        """Update CPU usage display."""
        try:
            label = self.query_one("#sys-cpu", Label)
            label.update(cpu)
        except Exception:
            pass  # Widget not mounted yet

    def watch_responsive_state(self, state: str) -> None:
        """Update status bar layout based on responsive state.

        Adjusts visibility and labels to optimize space usage across
        different terminal widths.

        Args:
            state: Responsive state ("compact", "standard", "comfortable", "wide")

        Breakpoint behaviors:
            - compact (100-119): Icon buttons, hide system health & margin
            - standard (120-139): Short labels, hide API latency & margin
            - comfortable (140-159): Full labels, all metrics visible
            - wide (160+): Expanded labels ("Portfolio" vs "Equity")
        """
        try:
            # Update button labels based on state
            start_btn = self.query_one("#start-btn", Button)
            stop_btn = self.query_one("#stop-btn", Button)

            if state == "compact":
                # Icon-only buttons for narrow terminals
                start_btn.label = "▶"
                stop_btn.label = "⏹"
            elif state == "standard":
                # Short labels
                start_btn.label = "▶ Start"
                stop_btn.label = "⏹ Stop"
            else:  # comfortable or wide
                # Full labels
                start_btn.label = "▶ Start Bot"
                stop_btn.label = "⏹ Stop"

            # Toggle system health section visibility
            system_health = self.query_one("#system-health-section")
            if state in ("compact", "standard"):
                # Hide entire system health section at narrow widths
                system_health.display = False
            else:
                system_health.display = True

            # Toggle margin display visibility
            # Find the margin label's parent container
            try:
                margin_label = self.query_one("#margin-value")
                margin_container = margin_label.parent

                if state in ("compact", "standard"):
                    # Hide margin at narrow widths (less critical metric)
                    if margin_container:
                        margin_container.display = False
                else:
                    if margin_container:
                        margin_container.display = True
            except Exception:
                # Margin label might not exist yet
                pass

            # TODO: Implement expanded labels for "wide" state
            # e.g., "Equity" → "Portfolio", "P&L" → "Daily P&L"
            # Requires updating Label text, not just button labels

        except Exception:
            # Silently handle if widgets not mounted yet
            pass
