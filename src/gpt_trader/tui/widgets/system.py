from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import ExecutionMetrics, SystemStatus, WebSocketState


class SystemHealthWidget(Static):
    """Widget to display system health and brokerage connection status."""

    # Styles moved to styles/widgets/system.tcss

    system_data = reactive(SystemStatus())
    websocket_data = reactive(WebSocketState())

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
                yield Label("|", classes="metric-separator")
                yield Label("WS: --", id="ws-status", classes="value")
                yield Label("|", classes="metric-separator")
                yield Label("Vfail: 0", id="validation-failures", classes="value")
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

                with Horizontal(classes="metric-row"):
                    yield Label("Validation:", classes="label")
                    yield Label("OK", id="validation-failures", classes="value")

                with Horizontal(classes="metric-row"):
                    yield Label("WebSocket:", classes="label")
                    yield Label("--", id="ws-status", classes="value")

                # Execution Issues section (hidden by default, shown when issues exist)
                with Vertical(id="execution-issues", classes="execution-issues hidden"):
                    yield Label("Exec Issues:", classes="label execution-label")
                    yield Label("", id="exec-rejects", classes="value")
                    yield Label("", id="exec-retries", classes="value")

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

        # Update validation failure display
        self._update_validation_display(data)

    def _update_validation_display(self, data: SystemStatus) -> None:
        """Update the validation failure indicator.

        Shows total consecutive failures across all check types.
        Highlights in warning style if failures exist or if escalated.
        In expanded mode, shows which specific checks are failing.
        """
        try:
            vfail_label = self.query_one("#validation-failures", Label)

            # Calculate total failures across all check types
            total_failures = sum(data.validation_failures.values())

            # Update display text
            if self.compact_mode:
                if data.validation_escalated:
                    vfail_label.update(f"Vfail: {total_failures} ⚠")
                else:
                    vfail_label.update(f"Vfail: {total_failures}")
            else:
                # Expanded mode shows more detail
                if data.validation_escalated:
                    vfail_label.update(f"{total_failures} (ESCALATED)")
                elif total_failures > 0:
                    # Show which checks are failing
                    failing_checks = self._format_failing_checks(data.validation_failures)
                    vfail_label.update(failing_checks)
                else:
                    vfail_label.update("OK")

            # Update styling based on state
            vfail_label.remove_class("status-warning")
            vfail_label.remove_class("status-error")
            vfail_label.remove_class("status-ok")

            if data.validation_escalated:
                vfail_label.add_class("status-error")
            elif total_failures > 0:
                vfail_label.add_class("status-warning")
            else:
                vfail_label.add_class("status-ok")
        except Exception:
            pass  # Label may not exist yet during initial compose

    def _format_failing_checks(self, failures: dict[str, int]) -> str:
        """Format failing validation checks for display.

        Args:
            failures: Dict mapping check_type to consecutive failure count.

        Returns:
            Formatted string showing which checks are failing.
        """
        # Short names for check types
        check_names = {
            "mark_staleness": "mark",
            "slippage_guard": "slip",
            "order_preview": "prev",
        }

        parts = []
        for check_type, count in failures.items():
            if count > 0:
                short_name = check_names.get(check_type, check_type[:4])
                parts.append(f"{short_name}:{count}")

        if parts:
            return " ".join(parts)
        return "OK"

    @safe_update
    def update_execution_metrics(self, metrics: ExecutionMetrics) -> None:
        """Update execution issues display.

        Shows rejection and retry reason breakdowns when issues exist.
        Hidden when no issues are present.

        Args:
            metrics: ExecutionMetrics with rejection_reasons and retry_reasons.
        """
        if self.compact_mode:
            # Compact mode doesn't show execution details
            return

        try:
            issues_container = self.query_one("#execution-issues", Vertical)
            rejects_label = self.query_one("#exec-rejects", Label)
            retries_label = self.query_one("#exec-retries", Label)

            has_rejects = bool(metrics.rejection_reasons)
            has_retries = bool(metrics.retry_reasons)

            if not has_rejects and not has_retries:
                # No issues - hide section
                issues_container.add_class("hidden")
                return

            # Show section
            issues_container.remove_class("hidden")

            # Format rejection reasons (top 2)
            if has_rejects:
                rejects_text = self._format_reason_summary(metrics.top_rejection_reasons, "Rejects")
                rejects_label.update(rejects_text)
                rejects_label.remove_class("hidden")
            else:
                rejects_label.add_class("hidden")

            # Format retry reasons (top 2)
            if has_retries:
                retries_text = self._format_reason_summary(metrics.top_retry_reasons, "Retries")
                retries_label.update(retries_text)
                retries_label.remove_class("hidden")
            else:
                retries_label.add_class("hidden")

        except Exception:
            pass  # Section may not exist in compact mode

    def _format_reason_summary(self, reasons: list[tuple[str, int]], prefix: str) -> str:
        """Format reason breakdown for display.

        Shows top 2 reasons with counts, ellipsis if more.

        Args:
            reasons: List of (reason, count) tuples sorted by count.
            prefix: Label prefix (e.g., "Rejects", "Retries").

        Returns:
            Formatted string like "Rejects: rate_limit(3), timeout(1)"
        """
        if not reasons:
            return ""

        # Take top 2
        top_reasons = reasons[:2]
        parts = [f"{reason}({count})" for reason, count in top_reasons]

        # Add ellipsis if more
        if len(reasons) > 2:
            parts.append("…")

        return f"{prefix}: {', '.join(parts)}"

    @safe_update
    def update_websocket(self, data: WebSocketState) -> None:
        """Update WebSocket health display.

        Shows connection state, staleness indicators, and gap count.

        Args:
            data: WebSocketState with connection health metrics.
        """
        self.websocket_data = data

        try:
            ws_label = self.query_one("#ws-status", Label)

            # Build status string
            if self.compact_mode:
                status_text = self._format_ws_compact(data)
            else:
                status_text = self._format_ws_expanded(data)

            ws_label.update(status_text)

            # Update styling based on state
            ws_label.remove_class("status-connected")
            ws_label.remove_class("status-warning")
            ws_label.remove_class("status-error")
            ws_label.remove_class("status-unknown")

            if data.message_stale or data.heartbeat_stale:
                ws_label.add_class("status-error")
            elif data.connected:
                ws_label.add_class("status-connected")
            elif data.last_message_ts is not None:
                # Has had messages before but currently disconnected
                ws_label.add_class("status-warning")
            else:
                ws_label.add_class("status-unknown")

        except Exception:
            pass  # Widget may not exist yet

    def _format_ws_compact(self, data: WebSocketState) -> str:
        """Format WS status for compact display.

        Shows: WS: OK / STALE / gaps:N / --
        """
        if data.message_stale or data.heartbeat_stale:
            return "WS: STALE"
        elif data.gap_count > 0:
            return f"WS: gaps:{data.gap_count}"
        elif data.connected:
            return "WS: OK"
        elif data.last_message_ts is not None:
            return "WS: DISC"
        else:
            return "WS: --"

    def _format_ws_expanded(self, data: WebSocketState) -> str:
        """Format WS status for expanded display.

        Shows more detail: CONNECTED (gaps: N, reconnects: N)
        """
        import time

        if data.message_stale or data.heartbeat_stale:
            age = 0.0
            if data.message_stale and data.last_message_ts:
                age = time.time() - data.last_message_ts
            elif data.heartbeat_stale and data.last_heartbeat_ts:
                age = time.time() - data.last_heartbeat_ts
            return f"STALE ({age:.0f}s)"
        elif data.connected:
            details = []
            if data.gap_count > 0:
                details.append(f"gaps:{data.gap_count}")
            if data.reconnect_count > 0:
                details.append(f"reconn:{data.reconnect_count}")
            if details:
                return f"OK ({', '.join(details)})"
            return "OK"
        elif data.last_message_ts is not None:
            return "DISCONNECTED"
        else:
            return "--"
