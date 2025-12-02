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
        background: #2E2922;  /* bg-secondary */
        border: solid #3A3530; /* border-subtle */
        height: auto;
        padding: 1;
    }

    SystemHealthWidget .header {
        text-style: bold;
        color: #C15F3C; /* accent */
        margin-bottom: 1;
    }

    SystemHealthWidget .metric-row {
        height: 1;
        margin-bottom: 0;
    }

    SystemHealthWidget .label {
        color: #ABA8A5; /* text-secondary */
        width: 15;
    }

    SystemHealthWidget .value {
        color: #E8E6E3; /* text-primary */
        text-style: bold;
    }

    .status-connected {
        color: #7AA874; /* success */
    }

    .status-disconnected {
        color: #D4736E; /* error */
    }

    .status-unknown {
        color: #D8A657; /* warning */
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
