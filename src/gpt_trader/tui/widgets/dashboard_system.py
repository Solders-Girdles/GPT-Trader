"""
System-related dashboard widgets.

Contains:
- SystemThresholds: Configurable thresholds for system monitor color coding
- DEFAULT_THRESHOLDS: Default SystemThresholds instance
- SystemMonitorWidget: Dashboard widget displaying system health metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.thresholds import DEFAULT_THRESHOLDS as PERF_THRESHOLDS
from gpt_trader.tui.thresholds import (
    get_error_rate_status,
    get_latency_status,
    get_status_color,
    get_status_icon,
)
from gpt_trader.tui.widgets.primitives import ProgressBarWidget
from gpt_trader.tui.widgets.value_flash import flash_label
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


@dataclass(frozen=True)
class SystemThresholds:
    """Configurable thresholds for system monitor color coding.

    All values define boundaries between OK/WARNING/CRITICAL states.
    Values aligned with shared thresholds in gpt_trader.tui.thresholds.
    """

    # Latency thresholds (milliseconds)
    latency_good: float = 50.0  # Below = OK
    latency_warn: float = 150.0  # Below = WARNING, above = CRITICAL

    # Rate limit thresholds (percentage)
    rate_limit_good: float = 50.0  # Below = OK
    rate_limit_warn: float = 80.0  # Below = WARNING, above = CRITICAL

    # CPU thresholds (percentage)
    cpu_warn: float = 50.0  # Below = OK
    cpu_critical: float = 80.0  # Below = WARNING, above = CRITICAL

    # Memory thresholds (MB)
    memory_max: float = 1024.0  # Max memory for percentage calculation
    memory_warn: float = 60.0  # Percentage threshold for WARNING
    memory_critical: float = 80.0  # Percentage threshold for CRITICAL


# Default thresholds instance
DEFAULT_THRESHOLDS = SystemThresholds()


class SystemMonitorWidget(Static):
    """
    Displays system health metrics in a compact panel.

    Shows: CPU, Memory, Latency, Connection Status, Rate Limit.
    Implements StateObserver to receive updates via StateRegistry broadcast.

    Styles are defined in styles/widgets/dashboard.tcss for centralized theming.

    Args:
        thresholds: Optional SystemThresholds for configurable color coding.
    """

    SCOPED_CSS = False  # Use global styles from dashboard.tcss

    cpu_usage = reactive(0.0)
    memory_usage = reactive("0MB")
    latency = reactive(0.0)
    connection_status = reactive("CONNECTING")
    rate_limit = reactive("0%")

    # Resilience metrics from CoinbaseClient
    latency_p50 = reactive(0.0)
    latency_p95 = reactive(0.0)
    error_rate_pct = reactive(0.0)
    cache_hit_rate_pct = reactive(0.0)
    circuit_state = reactive("OK")

    # Execution telemetry
    exec_success_rate = reactive(100.0)
    exec_latency_ms = reactive(0.0)
    exec_count = reactive(0)

    def __init__(
        self,
        thresholds: SystemThresholds | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        # Display signature cache for early-exit optimization
        self._last_display_signature: tuple | None = None

    def on_mount(self) -> None:
        """Register with StateRegistry on mount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def _compute_display_signature(self, state: TuiState) -> tuple:
        """Compute a signature from all fields displayed by this widget.

        Returns a tuple that can be compared for equality to detect changes.
        """
        sys = state.system_data
        res = state.resilience_data
        exec_data = state.execution_data

        # System metrics section
        sys_sig = (
            (
                getattr(sys, "cpu_usage", None),
                getattr(sys, "api_latency", None),
                getattr(sys, "memory_usage", None),
                getattr(sys, "connection_status", None),
                getattr(sys, "rate_limit_usage", None),
            )
            if sys
            else ()
        )

        # Resilience metrics section
        res_sig = (
            (
                getattr(res, "latency_p50_ms", None),
                getattr(res, "latency_p95_ms", None),
                getattr(res, "error_rate", None),
                getattr(res, "cache_hit_rate", None),
                getattr(res, "any_circuit_open", None),
                getattr(res, "last_update", None),
            )
            if res
            else ()
        )

        # Execution metrics section
        exec_sig = (
            (
                getattr(exec_data, "success_rate", None),
                getattr(exec_data, "avg_latency_ms", None),
                getattr(exec_data, "submissions_total", None),
            )
            if exec_data
            else ()
        )

        # Include running state for connection status logic
        return (sys_sig, res_sig, exec_sig, state.running, state.data_source_mode)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts system data from TuiState.system_data (SystemStatus).
        """
        if not state.system_data:
            return

        # Early exit if display signature unchanged
        sig = self._compute_display_signature(state)
        if sig == self._last_display_signature:
            return
        self._last_display_signature = sig

        system_data = state.system_data

        # Extract CPU usage (remove % suffix and convert to float)
        try:
            cpu_val = system_data.cpu_usage
            if isinstance(cpu_val, str) and cpu_val.endswith("%"):
                self.cpu_usage = float(cpu_val.rstrip("%"))
            elif isinstance(cpu_val, (int, float)):
                self.cpu_usage = float(cpu_val)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed to extract CPU usage: %s", e)

        # Extract latency
        try:
            latency_val = system_data.api_latency
            if isinstance(latency_val, (int, float)):
                self.latency = float(latency_val)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed to extract latency: %s", e)

        # Extract memory usage
        try:
            memory_val = system_data.memory_usage
            if memory_val is not None:
                self.memory_usage = str(memory_val)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract memory usage: %s", e)

        # Extract connection status
        try:
            if state.data_source_mode != "demo" and not state.running:
                self.connection_status = "STOPPED"
            else:
                conn_val = system_data.connection_status
                if conn_val is not None:
                    self.connection_status = str(conn_val)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract connection status: %s", e)

        # Extract rate limit usage
        try:
            rate_val = system_data.rate_limit_usage
            if rate_val is not None:
                self.rate_limit = str(rate_val)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract rate limit: %s", e)

        # Extract resilience metrics if available
        try:
            res = state.resilience_data
            if res and res.last_update > 0:
                self.latency_p50 = res.latency_p50_ms
                self.latency_p95 = res.latency_p95_ms
                self.error_rate_pct = res.error_rate * 100
                self.cache_hit_rate_pct = res.cache_hit_rate * 100
                self.circuit_state = "OPEN" if res.any_circuit_open else "OK"
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract resilience metrics: %s", e)

        # Extract execution telemetry if available
        try:
            exec_data = state.execution_data
            if exec_data and exec_data.submissions_total > 0:
                self.exec_success_rate = exec_data.success_rate
                self.exec_latency_ms = exec_data.avg_latency_ms
                self.exec_count = exec_data.submissions_total
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract execution metrics: %s", e)

    def compose(self) -> ComposeResult:
        yield Label("SYSTEM", classes="sys-header")

        # CPU with progress bar
        yield ProgressBarWidget(percentage=0.0, label="CPU", id="pb-cpu", classes="sys-row")

        # Memory with progress bar
        yield ProgressBarWidget(percentage=0.0, label="MEM", id="pb-memory", classes="sys-row")

        # Latency with color coding
        yield Label("Latency: 0ms", id="lbl-latency", classes="sys-metric")

        # Connection status
        yield Label("[yellow]○ Connecting...[/yellow]", id="lbl-conn", classes="sys-metric warning")

        # Rate limit with progress bar
        yield ProgressBarWidget(percentage=0.0, label="Rate", id="pb-rate", classes="sys-row")

        # Resilience metrics section
        with Container(id="resilience-section", classes="resilience-metrics"):
            yield Label("p50/p95: --ms / --ms", id="lbl-latency-pct", classes="sys-metric")
            yield Label("Errors: 0.0%", id="lbl-error-rate", classes="sys-metric")
            yield Label("Cache: --%", id="lbl-cache-hit", classes="sys-metric")
            yield Label("[green]Circuit: OK[/green]", id="lbl-circuit", classes="sys-metric")

        # Execution telemetry section
        with Container(id="execution-section", classes="execution-metrics"):
            yield Label("Exec: --% (0)", id="lbl-exec-rate", classes="sys-metric")
            yield Label("Exec Lat: --ms", id="lbl-exec-latency", classes="sys-metric")

    def watch_cpu_usage(self, val: float) -> None:
        _last = getattr(self, "_last_cpu_usage", object())
        if val == _last:
            return
        self._last_cpu_usage = val

        try:
            self.query_one("#pb-cpu", ProgressBarWidget).percentage = val
        except Exception as e:
            logger.debug("Failed to update CPU display: %s", e)

    def watch_memory_usage(self, val: str) -> None:
        _last = getattr(self, "_last_memory_usage", object())
        if val == _last:
            return
        self._last_memory_usage = val

        try:
            pb = self.query_one("#pb-memory", ProgressBarWidget)
            # Extract numeric value from "256MB" format
            try:
                memory_mb = float(val.rstrip("MB").rstrip("GB").rstrip("mb").rstrip("gb"))
                # Convert GB to MB if needed
                if "GB" in val.upper():
                    memory_mb *= 1024
                pct = min(100.0, (memory_mb / self.thresholds.memory_max) * 100)
                pb.percentage = pct
            except (ValueError, AttributeError):
                # If parsing fails, set to 0
                pb.percentage = 0.0
        except Exception as e:
            logger.debug("Failed to update memory display: %s", e)

    def watch_latency(self, val: float) -> None:
        _last = getattr(self, "_last_latency", object())
        if val == _last:
            return
        self._last_latency = val

        try:
            lbl = self.query_one("#lbl-latency", Label)
            # Color code using shared thresholds with icon for accessibility
            status = get_latency_status(val, PERF_THRESHOLDS)
            color = get_status_color(status)
            icon = get_status_icon(status)
            lbl.update(f"[{color}]{icon} Latency: {val:.0f}ms[/{color}]")
        except Exception as e:
            logger.debug("Failed to update latency display: %s", e)

    def watch_connection_status(self, val: str) -> None:
        _last = getattr(self, "_last_connection_status", object())
        if val == _last:
            return
        self._last_connection_status = val

        try:
            lbl = self.query_one("#lbl-conn", Label)
            status_upper = val.upper()
            if status_upper in ("CONNECTED", "OK", "HEALTHY"):
                lbl.update("[green]● Connected[/green]")
                lbl.remove_class("stopped", "warning", "bad")
                lbl.add_class("good")
                # Flash green when connection is established
                flash_label(lbl, direction="up", duration=0.6)
            elif status_upper in ("STOPPED", "IDLE"):
                lbl.update("[cyan]■ Stopped[/cyan]")
                lbl.remove_class("good", "warning", "bad")
                lbl.add_class("stopped")
            elif status_upper in ("CONNECTING", "RECONNECTING", "SYNCING", "--", "UNKNOWN"):
                lbl.update("[yellow]○ Connecting...[/yellow]")
                lbl.remove_class("stopped", "good", "bad")
                lbl.add_class("warning")
            else:
                lbl.update(f"[red]■ {val}[/red]")
                lbl.remove_class("stopped", "good", "warning")
                lbl.add_class("bad")
                # Flash red when connection has issues
                flash_label(lbl, direction="down", duration=0.6)
        except Exception as e:
            logger.debug("Failed to update connection status display: %s", e)

    def watch_rate_limit(self, val: str) -> None:
        _last = getattr(self, "_last_rate_limit", object())
        if val == _last:
            return
        self._last_rate_limit = val

        try:
            pb = self.query_one("#pb-rate", ProgressBarWidget)
            # Extract numeric value for progress bar
            try:
                pct = float(val.rstrip("%"))
                pb.percentage = pct
            except (ValueError, AttributeError):
                pb.percentage = 0.0
        except Exception as e:
            logger.debug("Failed to update rate limit display: %s", e)

    def watch_latency_p50(self, val: float) -> None:
        """Update the p50/p95 latency display."""
        _last = getattr(self, "_last_latency_p50", object())
        if val == _last:
            return
        self._last_latency_p50 = val

        try:
            lbl = self.query_one("#lbl-latency-pct", Label)
            lbl.update(f"p50/p95: {val:.0f}ms / {self.latency_p95:.0f}ms")
        except Exception as e:
            logger.debug("Failed to update latency percentiles: %s", e)

    def watch_latency_p95(self, val: float) -> None:
        """Update the p50/p95 latency display when p95 changes."""
        _last = getattr(self, "_last_latency_p95", object())
        if val == _last:
            return
        self._last_latency_p95 = val

        try:
            lbl = self.query_one("#lbl-latency-pct", Label)
            lbl.update(f"p50/p95: {self.latency_p50:.0f}ms / {val:.0f}ms")
        except Exception as e:
            logger.debug("Failed to update latency percentiles: %s", e)

    def watch_error_rate_pct(self, val: float) -> None:
        """Update the error rate display with color coding and icon for accessibility."""
        _last = getattr(self, "_last_error_rate_pct", object())
        if val == _last:
            return
        self._last_error_rate_pct = val

        try:
            lbl = self.query_one("#lbl-error-rate", Label)
            # Color code using shared thresholds with icon for accessibility
            status = get_error_rate_status(val, PERF_THRESHOLDS)
            color = get_status_color(status)
            icon = get_status_icon(status)
            lbl.update(f"[{color}]{icon} Errors: {val:.1f}%[/{color}]")
        except Exception as e:
            logger.debug("Failed to update error rate: %s", e)

    def watch_cache_hit_rate_pct(self, val: float) -> None:
        """Update the cache hit rate display."""
        _last = getattr(self, "_last_cache_hit_rate_pct", object())
        if val == _last:
            return
        self._last_cache_hit_rate_pct = val

        try:
            lbl = self.query_one("#lbl-cache-hit", Label)
            if val > 0:
                lbl.update(f"Cache: {val:.0f}%")
            else:
                lbl.update("Cache: --%")
        except Exception as e:
            logger.debug("Failed to update cache hit rate: %s", e)

    def watch_circuit_state(self, val: str) -> None:
        """Update the circuit breaker state display."""
        _last = getattr(self, "_last_circuit_state", object())
        if val == _last:
            return
        self._last_circuit_state = val

        try:
            lbl = self.query_one("#lbl-circuit", Label)
            if val == "OK":
                lbl.update("[green]Circuit: OK[/green]")
            else:
                lbl.update("[red]Circuit: OPEN[/red]")
        except Exception as e:
            logger.debug("Failed to update circuit state: %s", e)

    def watch_exec_success_rate(self, val: float) -> None:
        """Update the execution success rate display."""
        _last = getattr(self, "_last_exec_success_rate", object())
        if val == _last:
            return
        self._last_exec_success_rate = val

        try:
            lbl = self.query_one("#lbl-exec-rate", Label)
            # Color code based on success rate
            if val >= 95:
                color = "green"
                icon = "✓"
            elif val >= 80:
                color = "yellow"
                icon = "●"
            else:
                color = "red"
                icon = "✗"
            lbl.update(f"[{color}]{icon} Exec: {val:.0f}% ({self.exec_count})[/{color}]")
        except Exception as e:
            logger.debug("Failed to update execution rate: %s", e)

    def watch_exec_latency_ms(self, val: float) -> None:
        """Update the execution latency display."""
        _last = getattr(self, "_last_exec_latency_ms", object())
        if val == _last:
            return
        self._last_exec_latency_ms = val

        try:
            lbl = self.query_one("#lbl-exec-latency", Label)
            if val > 0:
                # Color code based on latency
                if val < 100:
                    color = "green"
                elif val < 500:
                    color = "yellow"
                else:
                    color = "red"
                lbl.update(f"[{color}]Exec Lat: {val:.0f}ms[/{color}]")
            else:
                lbl.update("Exec Lat: --ms")
        except Exception as e:
            logger.debug("Failed to update execution latency: %s", e)

    def watch_exec_count(self, val: int) -> None:
        """Update execution count in the rate display."""
        _last = getattr(self, "_last_exec_count", object())
        if val == _last:
            return
        self._last_exec_count = val

        # Triggers update via watch_exec_success_rate
        try:
            lbl = self.query_one("#lbl-exec-rate", Label)
            rate = self.exec_success_rate
            if rate >= 95:
                color = "green"
                icon = "✓"
            elif rate >= 80:
                color = "yellow"
                icon = "●"
            else:
                color = "red"
                icon = "✗"
            lbl.update(f"[{color}]{icon} Exec: {rate:.0f}% ({val})[/{color}]")
        except Exception as e:
            logger.debug("Failed to update execution count: %s", e)
