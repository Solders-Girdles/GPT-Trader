from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import SystemStatus


class SystemHealthWidget(Static):
    """Widget to display system health and brokerage connection status."""

    # Styles moved to styles/widgets/system.tcss

    system_data = reactive(SystemStatus())

    def __init__(self, compact_mode: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode

    def compose(self) -> ComposeResult:
        yield Label("SYSTEM", classes="widget-header")

        if self.compact_mode:
            # Compact horizontal layout - all metrics in one row
            with Horizontal(classes="compact-metrics"):
                yield Label("●", id="conn-indicator", classes="status-unknown")
                yield Label("UNKNOWN", id="connection-status", classes="value status-unknown")
                yield Label("|", classes="metric-separator")
                yield Label("0ms", id="latency", classes="value")
                yield Label("|", classes="metric-separator")
                yield Label("Rate: 0%", id="rate-limit", classes="value")
                yield Label("|", classes="metric-separator")
                yield Label("CPU: 0%", id="cpu", classes="value")
        else:
            # Full vertical layout (existing implementation)
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

        # Update connection indicator (only in compact mode)
        if self.compact_mode:
            try:
                conn_indicator = self.query_one("#conn-indicator", Label)
                conn_indicator.remove_class("status-connected")
                conn_indicator.remove_class("status-disconnected")
                conn_indicator.remove_class("status-unknown")

                if data.connection_status == "CONNECTED":
                    conn_indicator.add_class("status-connected")
                elif data.connection_status == "DISCONNECTED":
                    conn_indicator.add_class("status-disconnected")
                else:
                    conn_indicator.add_class("status-unknown")
            except Exception:
                pass  # Indicator doesn't exist in expanded mode

        # Update Metrics
        self.query_one("#latency", Label).update(f"{data.api_latency:.0f}ms")

        if self.compact_mode:
            # Compact mode shows labels inline with values
            self.query_one("#rate-limit", Label).update(f"Rate: {data.rate_limit_usage}")
            self.query_one("#cpu", Label).update(f"CPU: {data.cpu_usage}")
        else:
            # Expanded mode shows just values (labels are separate)
            self.query_one("#rate-limit", Label).update(data.rate_limit_usage)
            self.query_one("#memory", Label).update(data.memory_usage)
            self.query_one("#cpu", Label).update(data.cpu_usage)
