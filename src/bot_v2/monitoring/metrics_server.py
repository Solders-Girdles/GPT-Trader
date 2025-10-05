"""Prometheus metrics HTTP server for bot observability.

Provides /metrics endpoint for Prometheus scraping and /health endpoint
for Docker health checks and liveness probes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.orchestration.guardrails import GuardRailManager

logger = logging.getLogger(__name__)


class MetricsServer:
    """HTTP server exposing Prometheus metrics and health status.

    Runs in a background thread, provides /metrics and /health endpoints.
    Designed to be owned by PerpsBotLifecycle for lifecycle management.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        registry: CollectorRegistry | None = None,
        broker: IBrokerage | None = None,
    ) -> None:
        """Initialize metrics server.

        Args:
            host: Bind address (default: 0.0.0.0)
            port: Bind port (default: 9090)
            registry: Optional custom registry for testing isolation
            broker: Optional broker instance for health monitoring
        """
        self.host = host
        self.port = port
        self.registry = registry or CollectorRegistry()
        self._broker = broker
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._start_time = time.time()
        self._process = psutil.Process()
        self._guard_profile = "default"
        self._guard_states: dict[str, bool] = {}
        self._guardrails: GuardRailManager | None = None
        self._profile = "default"
        self._stream_name = "coinbase_ws"

        # Initialize metrics
        self._init_metrics()

        # Health status tracking
        self._last_cycle_timestamp: float | None = None
        self._background_tasks_running: int = 0
        self._streaming_connected: bool = False
        self._streaming_last_heartbeat: float | None = None
        self._streaming_last_message: float | None = None
        self._streaming_last_disconnect: float | None = None
        self._streaming_last_reconnect: float | None = None
        self._streaming_reconnect_attempts_total: int = 0
        self._streaming_reconnect_successes_total: int = 0
        self._streaming_reconnect_peak_attempt: int = 0
        self._streaming_fallback_active: bool = False

    def set_broker(self, broker: IBrokerage) -> None:
        """Update broker reference for health monitoring.

        Args:
            broker: Broker instance to monitor
        """
        self._broker = broker

    def set_profile(self, profile: str) -> None:
        """Update default profile label for metrics."""

        self._profile = profile

    def set_streaming_context(self, stream_name: str, profile: str | None = None) -> None:
        """Configure default labels for streaming metrics."""

        self._stream_name = stream_name
        if profile is not None:
            self.set_profile(profile)

    def _stream_labels(
        self, profile: str | None = None, stream: str | None = None
    ) -> tuple[str, str]:
        """Resolve label values for streaming metrics."""

        resolved_profile = profile or self._profile
        resolved_stream = stream or self._stream_name
        return resolved_profile, resolved_stream

    def register_guard_manager(
        self, guardrails: GuardRailManager, profile: str = "default"
    ) -> None:
        """Attach guard manager for guard state metrics."""

        self._guard_profile = profile
        self.set_profile(profile)
        self._guardrails = guardrails
        snapshot = guardrails.snapshot()
        for name, active in snapshot.items():
            self._update_guard_metric(name, active)
        self.error_streak_gauge.labels(profile=profile).set(guardrails.get_error_streak())
        self.daily_loss_gauge.labels(profile=profile).set(float(guardrails.get_daily_loss()))

        def _listener(name: str, active: bool) -> None:
            self._update_guard_metric(name, active)

        guardrails.register_listener(_listener)

    def record_guard_trip(self, guard: str, reason: str, profile: str | None = None) -> None:
        """Record a guard trip event."""

        label_profile = profile or self._guard_profile or self._profile
        self.guard_trip_counter.labels(profile=label_profile, guard=guard, reason=reason).inc()

    def _update_guard_metric(self, guard: str, active: bool) -> None:
        self._guard_states[guard] = active
        self.guard_state_gauge.labels(profile=self._guard_profile, guard=guard).set(
            1 if active else 0
        )

    def update_error_streak(self, streak: int, profile: str | None = None) -> None:
        """Update error streak gauge."""

        label_profile = profile or self._guard_profile or self._profile
        self.error_streak_gauge.labels(profile=label_profile).set(streak)

    def _init_metrics(self) -> None:
        """Initialize Prometheus metric collectors."""
        # Bot uptime
        self.uptime_gauge = Gauge(
            "bot_uptime_seconds",
            "Bot uptime in seconds",
            labelnames=["profile"],
            registry=self.registry,
        )

        # Cycle duration
        self.cycle_duration_histogram = Histogram(
            "bot_cycle_duration_seconds",
            "Bot cycle execution duration",
            labelnames=["profile"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Background tasks
        self.background_tasks_gauge = Gauge(
            "bot_background_tasks",
            "Number of active background tasks",
            labelnames=["profile", "task"],
            registry=self.registry,
        )

        # System metrics
        self.memory_gauge = Gauge(
            "bot_memory_used_bytes",
            "Process memory usage in bytes",
            labelnames=["profile"],
            registry=self.registry,
        )

        self.cpu_gauge = Gauge(
            "bot_cpu_percent",
            "Process CPU usage percentage",
            labelnames=["profile"],
            registry=self.registry,
        )

        # Order metrics
        self.order_attempts_counter = Counter(
            "bot_order_attempts_total",
            "Total order attempts by status",
            labelnames=["profile", "status"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_counter = Counter(
            "bot_errors_total",
            "Total errors by component and severity",
            labelnames=["component", "profile", "severity"],
            registry=self.registry,
        )

        self.guard_state_gauge = Gauge(
            "bot_guard_active",
            "Active guard states",
            labelnames=["profile", "guard"],
            registry=self.registry,
        )

        self.guard_trip_counter = Counter(
            "bot_guard_trips_total",
            "Guard rail trip events",
            labelnames=["profile", "guard", "reason"],
            registry=self.registry,
        )

        self.daily_loss_gauge = Gauge(
            "bot_guard_daily_loss_usd",
            "Current daily realized loss in USD",
            labelnames=["profile"],
            registry=self.registry,
        )

        self.error_streak_gauge = Gauge(
            "bot_guard_error_streak",
            "Current consecutive critical error streak",
            labelnames=["profile"],
            registry=self.registry,
        )

        # Streaming metrics
        self.streaming_connection_gauge = Gauge(
            "bot_streaming_connection_state",
            "Streaming connection state (1=connected, 0=disconnected)",
            labelnames=["profile", "stream"],
            registry=self.registry,
        )

        self.streaming_heartbeat_lag_gauge = Gauge(
            "bot_streaming_heartbeat_lag_seconds",
            "Seconds since last streaming heartbeat/message",
            labelnames=["profile", "stream"],
            registry=self.registry,
        )

        self.streaming_reconnect_counter = Counter(
            "bot_streaming_reconnect_total",
            "Streaming reconnect attempts partitioned by result",
            labelnames=["profile", "stream", "status"],
            registry=self.registry,
        )

        self.streaming_message_gap_histogram = Histogram(
            "bot_streaming_inter_message_seconds",
            "Elapsed seconds between streaming messages",
            labelnames=["profile", "stream"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        self.streaming_fallback_gauge = Gauge(
            "bot_streaming_fallback_active",
            "Indicates whether REST fallback polling is active",
            labelnames=["profile", "stream"],
            registry=self.registry,
        )

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self._running:
            logger.warning("MetricsServer already running")
            return

        self._running = True
        self._start_time = time.time()

        # Create request handler with access to this instance
        server_instance = self

        class MetricsHandler(BaseHTTPRequestHandler):
            """HTTP request handler for /metrics and /health endpoints."""

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress default HTTP logging."""
                pass

            def do_GET(self) -> None:
                """Handle GET requests."""
                try:
                    if self.path == "/metrics":
                        self._handle_metrics()
                    elif self.path == "/health":
                        self._handle_health()
                    else:
                        self.send_error(404, "Not Found")
                except Exception as e:
                    logger.error(f"Error handling request {self.path}: {e}")
                    self.send_error(500, "Internal Server Error")

            def _handle_metrics(self) -> None:
                """Serve Prometheus metrics."""
                # Update system metrics
                server_instance._update_system_metrics()

                # Generate metrics output
                metrics_output = generate_latest(server_instance.registry)

                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(metrics_output)

            def _handle_health(self) -> None:
                """Serve health check status."""
                now = time.time()
                uptime = now - server_instance._start_time
                last_heartbeat = server_instance._streaming_last_heartbeat
                last_message = server_instance._streaming_last_message
                heartbeat_lag = (
                    max(0.0, now - last_heartbeat) if last_heartbeat is not None else None
                )
                message_lag = max(0.0, now - last_message) if last_message is not None else None
                health_data: dict[str, Any] = {
                    "status": "ok",
                    "uptime_seconds": round(uptime, 2),
                    "background_tasks_running": server_instance._background_tasks_running,
                    "last_cycle_timestamp": server_instance._last_cycle_timestamp,
                    "streaming_connected": server_instance._streaming_connected,
                    "streaming": {
                        "stream_name": server_instance._stream_name,
                        "connected": server_instance._streaming_connected,
                        "last_message_timestamp": last_message,
                        "last_heartbeat_timestamp": last_heartbeat,
                        "heartbeat_lag_seconds": heartbeat_lag,
                        "message_lag_seconds": message_lag,
                        "last_disconnect_timestamp": server_instance._streaming_last_disconnect,
                        "last_reconnect_timestamp": server_instance._streaming_last_reconnect,
                        "reconnect_attempts_total": server_instance._streaming_reconnect_attempts_total,
                        "reconnect_successes_total": server_instance._streaming_reconnect_successes_total,
                        "reconnect_peak_attempt": server_instance._streaming_reconnect_peak_attempt,
                        "fallback_active": server_instance._streaming_fallback_active,
                    },
                    "guards": server_instance._guard_states,
                }

                # Include broker health if broker is available
                if server_instance._broker:
                    try:
                        broker_health = server_instance._broker.check_health()
                        health_data["broker"] = {
                            "connected": broker_health.connected,
                            "api_responsive": broker_health.api_responsive,
                            "last_check": broker_health.last_check_timestamp,
                            "error": broker_health.error_message,
                        }
                        # Update overall status if broker is unhealthy
                        if not broker_health.connected or not broker_health.api_responsive:
                            health_data["status"] = "degraded"
                    except Exception as exc:
                        logger.warning("Failed to check broker health: %s", exc)
                        health_data["broker"] = {"error": f"Health check failed: {exc}"}
                        health_data["status"] = "degraded"

                if server_instance._guardrails:
                    health_data["guard_error_streak"] = (
                        server_instance._guardrails.get_error_streak()
                    )
                    health_data["daily_loss_usd"] = float(
                        server_instance._guardrails.get_daily_loss()
                    )

                response = json.dumps(health_data, indent=2)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

        # Create and start server
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), MetricsHandler)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="MetricsServer",
            )
            self._thread.start()
            logger.info(f"MetricsServer started on {self.host}:{self.port}")
        except Exception as e:
            self._running = False
            logger.error(f"Failed to start MetricsServer: {e}")
            raise

    def stop(self) -> None:
        """Stop the metrics server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.shutdown()
            self._server.server_close()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("MetricsServer stopped")

    def _update_system_metrics(self) -> None:
        """Update system resource metrics (memory, CPU)."""
        try:
            # Memory usage
            mem_info = self._process.memory_info()
            self.memory_gauge.labels(profile=self._profile).set(mem_info.rss)

            # CPU usage (averaged over 0.1s)
            cpu_percent = self._process.cpu_percent(interval=0.1)
            self.cpu_gauge.labels(profile=self._profile).set(cpu_percent)

            # Daily loss (if guardrails manager attached)
            if self._guardrails is not None:
                daily_loss = float(self._guardrails.get_daily_loss())
                self.daily_loss_gauge.labels(profile=self._profile).set(daily_loss)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    # Public API for lifecycle integration

    def record_cycle_duration(self, duration_seconds: float, profile: str = "default") -> None:
        """Record a bot cycle duration.

        Args:
            duration_seconds: Cycle execution time
            profile: Bot profile name
        """
        self.cycle_duration_histogram.labels(profile=profile).observe(duration_seconds)
        self._last_cycle_timestamp = time.time()

    def update_background_tasks(self, task_name: str, count: int, profile: str = "default") -> None:
        """Update background task count.

        Args:
            task_name: Task identifier
            count: Current task count
            profile: Bot profile name
        """
        self.background_tasks_gauge.labels(profile=profile, task=task_name).set(count)
        self._background_tasks_running = count

    def record_order_attempt(self, status: str, profile: str = "default") -> None:
        """Record an order attempt.

        Args:
            status: Order status (attempted/success/failed)
            profile: Bot profile name
        """
        self.order_attempts_counter.labels(profile=profile, status=status).inc()

    def record_error(self, component: str, severity: str, profile: str = "default") -> None:
        """Record an error.

        Args:
            component: Component name where error occurred
            severity: Error severity level
            profile: Bot profile name
        """
        self.errors_counter.labels(component=component, profile=profile, severity=severity).inc()

    def update_streaming_status(
        self,
        connected: bool,
        *,
        profile: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Update streaming connection status.

        Args:
            connected: True if streaming is connected
        """
        label_profile, label_stream = self._stream_labels(profile, stream)
        self.streaming_connection_gauge.labels(profile=label_profile, stream=label_stream).set(
            1 if connected else 0
        )
        self._streaming_connected = connected
        now = time.time()
        if connected:
            self._streaming_last_message = now
        else:
            self._streaming_last_disconnect = now

    def update_uptime(self, profile: str = "default") -> None:
        """Update uptime gauge.

        Args:
            profile: Bot profile name
        """
        uptime = time.time() - self._start_time
        self.uptime_gauge.labels(profile=profile).set(uptime)

    def record_streaming_message(
        self,
        elapsed_since_last: float | None,
        *,
        timestamp: float | None = None,
        profile: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Record streaming message timing information."""

        label_profile, label_stream = self._stream_labels(profile, stream)

        event_time = timestamp if timestamp is not None else time.time()
        self._streaming_last_message = event_time
        self._streaming_last_heartbeat = event_time

        lag_seconds: float
        if elapsed_since_last is not None and elapsed_since_last >= 0:
            try:
                self.streaming_message_gap_histogram.labels(
                    profile=label_profile, stream=label_stream
                ).observe(elapsed_since_last)
            except Exception as exc:
                logger.debug("Failed to record streaming gap", exc_info=exc)
            lag_seconds = elapsed_since_last
        else:
            lag_seconds = max(0.0, time.time() - event_time)

        self.streaming_heartbeat_lag_gauge.labels(profile=label_profile, stream=label_stream).set(
            lag_seconds
        )

    def record_streaming_heartbeat(
        self,
        timestamp: float | None = None,
        *,
        profile: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Record a streaming heartbeat event."""

        label_profile, label_stream = self._stream_labels(profile, stream)
        event_time = timestamp if timestamp is not None else time.time()
        self._streaming_last_heartbeat = event_time
        lag_seconds = max(0.0, time.time() - event_time)
        self.streaming_heartbeat_lag_gauge.labels(profile=label_profile, stream=label_stream).set(
            lag_seconds
        )

    def record_streaming_reconnect(
        self,
        status: str,
        *,
        attempt: int | None = None,
        profile: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Record streaming reconnect attempts and outcomes."""

        label_profile, label_stream = self._stream_labels(profile, stream)
        self.streaming_reconnect_counter.labels(
            profile=label_profile, stream=label_stream, status=status
        ).inc()

        now = time.time()
        self._streaming_last_reconnect = now

        if status == "attempt":
            self._streaming_reconnect_attempts_total += 1
        elif status == "success":
            self._streaming_reconnect_successes_total += 1
            # Connection likely restored
            self.update_streaming_status(True, profile=label_profile, stream=label_stream)

        if attempt is not None:
            # Store highest observed attempt for diagnostics
            self._streaming_reconnect_peak_attempt = max(
                self._streaming_reconnect_peak_attempt, attempt
            )

    def update_streaming_fallback(
        self,
        active: bool,
        *,
        profile: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Record whether REST fallback polling is active."""

        label_profile, label_stream = self._stream_labels(profile, stream)
        self.streaming_fallback_gauge.labels(profile=label_profile, stream=label_stream).set(
            1 if active else 0
        )
        self._streaming_fallback_active = active

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
