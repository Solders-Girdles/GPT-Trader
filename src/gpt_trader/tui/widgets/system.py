from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import SystemStatus


class SystemHealthWidget(Static):
    """Widget to display system health and brokerage connection status."""

    DEFAULT_CSS = """
    SystemHealthWidget {
        background: #3b4252;  /* nord1 */
        border: solid #4c566a; /* nord3 */
        height: auto;
        padding: 1;
    }

    SystemHealthWidget .header {
        text-style: bold;
        color: #88c0d0; /* nord8 */
        margin-bottom: 1;
    }

    SystemHealthWidget .metric-row {
        height: 1;
        margin-bottom: 0;
    }

    SystemHealthWidget .label {
        color: #d8dee9; /* nord4 */
        width: 15;
    }

    SystemHealthWidget .value {
        color: #eceff4; /* nord6 */
        text-style: bold;
    }

    .status-connected {
        color: #a3be8c; /* nord14 */
    }

    .status-disconnected {
        color: #bf616a; /* nord11 */
    }

    .status-unknown {
        color: #ebcb8b; /* nord13 */
    }
    """

    system_data = reactive(SystemStatus())

    def compose(self) -> ComposeResult:
        yield Label("System Health", classes="header")

        with Vertical():
            with Horizontal(classes="metric-row"):
                yield Label("Connection:", classes="label")
                yield Label("UNKNOWN", id="connection-status", classes="value status-unknown")

            with Horizontal(classes="metric-row"):
                yield Label("Latency:", classes="label")
                yield Label("0ms", id="latency", classes="value")

            with Horizontal(classes="metric-row"):
                yield Label("Rate Limit:", classes="label")
                yield Label("0%", id="rate-limit", classes="value")

            with Horizontal(classes="metric-row"):
                yield Label("Memory:", classes="label")
                yield Label("0MB", id="memory", classes="value")

            with Horizontal(classes="metric-row"):
                yield Label("CPU:", classes="label")
                yield Label("0%", id="cpu", classes="value")

    @safe_update
    def update_system(self, data: SystemStatus) -> None:
        """Update the widget with new system data."""
        self.system_data = data

        # Update Connection Status
        conn_label = self.query_one("#connection-status", Label)
        conn_label.update(data.connection_status)

        conn_label.remove_class("status-connected")
        conn_label.remove_class("status-disconnected")
        conn_label.remove_class("status-unknown")

        if data.connection_status == "CONNECTED":
            conn_label.add_class("status-connected")
        elif data.connection_status == "DISCONNECTED":
            conn_label.add_class("status-disconnected")
        else:
            conn_label.add_class("status-unknown")

        # Update Metrics
        self.query_one("#latency", Label).update(f"{data.api_latency:.0f}ms")
        self.query_one("#rate-limit", Label).update(data.rate_limit_usage)
        self.query_one("#memory", Label).update(data.memory_usage)
        self.query_one("#cpu", Label).update(data.cpu_usage)
