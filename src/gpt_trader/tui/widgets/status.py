from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Label, Static


class BotStatusWidget(Static):
    """Displays high-level bot status in a horizontal bar."""

    uptime = reactive(0.0)
    cycle_count = reactive(0)
    running = reactive(False)
    equity = reactive("0.00")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("GPT-TRADER v1.0", classes="status-item status-value")
            yield Label("|", classes="status-item")
            yield Label("Status: ", classes="status-item")
            yield Label("STOPPED", id="status-label", classes="status-value status-stopped")
            yield Label("|", classes="status-item")
            yield Label("Uptime: ", classes="status-item")
            yield Label("0s", id="uptime-label", classes="status-value")
            yield Label("|", classes="status-item")
            yield Label("Cycles: ", classes="status-item")
            yield Label("0", id="cycles-label", classes="status-value")
            yield Label("|", classes="status-item")
            yield Label("Equity: ", classes="status-item")
            yield Label("$0.00", id="equity-label", classes="status-value")

    def watch_running(self, running: bool) -> None:
        label = self.query_one("#status-label", Label)
        if running:
            label.update("RUNNING")
            label.remove_class("status-stopped")
            label.add_class("status-running")
        else:
            label.update("STOPPED")
            label.remove_class("status-running")
            label.add_class("status-stopped")

    def watch_uptime(self, uptime: float) -> None:
        # Format uptime as HH:MM:SS
        m, s = divmod(int(uptime), 60)
        h, m = divmod(m, 60)
        self.query_one("#uptime-label", Label).update(f"{h:02d}:{m:02d}:{s:02d}")

    def watch_cycle_count(self, count: int) -> None:
        self.query_one("#cycles-label", Label).update(str(count))

    def watch_equity(self, equity: str) -> None:
        self.query_one("#equity-label", Label).update(f"${equity}")
