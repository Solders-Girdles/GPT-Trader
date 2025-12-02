from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label, Static


class BotStatusWidget(Static):
    """Displays high-level bot status with interactive controls."""

    uptime = reactive(0.0)
    cycle_count = reactive(0)
    running = reactive(False)
    equity = reactive("0.00")
    pnl = reactive("0.00")
    margin_usage = reactive("0.00%")

    class ToggleBotPressed(Message):
        """Message sent when start/stop button is pressed."""

    class PanicPressed(Message):
        """Message sent when panic button is pressed."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="status-bar-container"):
            # Left side: Bot control buttons
            with Container(id="bot-controls", classes="status-section"):
                with Horizontal(classes="button-group"):
                    yield Button("▶ Start", id="start-btn", variant="success")
                    yield Button("⏹ Stop", id="stop-btn", variant="primary", disabled=True)
                    yield Button("🚨 Panic", id="panic-btn", variant="error")

            # Center: Status info
            with Container(id="status-info", classes="status-section"):
                with Vertical():
                    with Horizontal(classes="status-row"):
                        yield Label("Status:", classes="status-label")
                        yield Label(
                            "STOPPED", id="status-value", classes="status-value status-stopped"
                        )
                    with Horizontal(classes="status-row"):
                        yield Label("Uptime:", classes="status-label")
                        yield Label("00:00:00", id="uptime-value", classes="status-value")

            # Right side: Performance metrics
            with Container(id="performance-metrics", classes="status-section"):
                with Vertical():
                    with Horizontal(classes="status-row"):
                        yield Label("Equity:", classes="status-label")
                        yield Label("$0.00", id="equity-value", classes="status-value")
                        yield Label("P&L:", classes="status-label")
                        yield Label("$0.00", id="pnl-value", classes="status-value pnl-neutral")
                    with Horizontal(classes="status-row"):
                        yield Label("Cycles:", classes="status-label")
                        yield Label("0", id="cycles-value", classes="status-value")
                        yield Label("Margin:", classes="status-label")
                        yield Label("0.00%", id="margin-value", classes="status-value")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "start-btn":
            self.post_message(self.ToggleBotPressed())
        elif event.button.id == "stop-btn":
            self.post_message(self.ToggleBotPressed())
        elif event.button.id == "panic-btn":
            self.post_message(self.PanicPressed())

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

    def watch_cycle_count(self, count: int) -> None:
        """Update cycle count display."""
        self.query_one("#cycles-value", Label).update(str(count))

    def watch_equity(self, equity: str) -> None:
        """Update equity display."""
        self.query_one("#equity-value", Label).update(f"${equity}")

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

    def watch_margin_usage(self, margin: str) -> None:
        """Update margin usage display."""
        self.query_one("#margin-value", Label).update(f"{margin}")
