"""
Lightweight HTTP health server for Kubernetes/Docker health checks.

Provides three endpoints:
- /health - Overall health status (liveness + readiness combined)
- /live - Liveness probe (is the process running?)
- /ready - Readiness probe (is the service ready to receive traffic?)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

from gpt_trader.config.constants import (
    HEALTH_CHECK_READ_TIMEOUT_SECONDS,
    METRICS_ENDPOINT_ENABLED,
)
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.performance.health import get_performance_health_check

logger = get_logger(__name__, component="health_server")

# Prometheus content type
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Default port for health server
DEFAULT_HEALTH_PORT = 8080


@dataclass
class HealthState:
    """Tracks application health state for probes."""

    ready: bool = False
    live: bool = True
    reason: str = "initializing"
    checks: dict[str, Any] = field(default_factory=dict)

    def set_ready(self, ready: bool, reason: str = "") -> None:
        """Update readiness state."""
        self.ready = ready
        if reason:
            self.reason = reason

    def set_live(self, live: bool, reason: str = "") -> None:
        """Update liveness state."""
        self.live = live
        if reason:
            self.reason = reason

    def add_check(self, name: str, status: bool, details: dict[str, Any] | None = None) -> None:
        """Add a named health check result."""
        self.checks[name] = {"status": "pass" if status else "fail", "details": details or {}}


def _build_health_response(state: HealthState) -> dict[str, Any]:
    """Build comprehensive health response combining all checks."""
    perf_health = get_performance_health_check()

    # Start with basic status
    overall_status = "healthy"
    if not state.live:
        overall_status = "unhealthy"
    elif not state.ready:
        overall_status = "starting"
    elif perf_health.get("status") == "degraded":
        overall_status = "degraded"
    elif perf_health.get("status") == "unhealthy":
        overall_status = "unhealthy"

    # Check individual health checks for severity escalation
    # Only escalate if we're currently healthy/degraded
    if overall_status in ("healthy", "degraded"):
        for check_name, check_result in state.checks.items():
            check_status = check_result.get("status", "pass")
            check_details = check_result.get("details", {})
            severity = check_details.get("severity", "warning")

            if check_status == "fail":
                if severity == "critical":
                    overall_status = "unhealthy"
                    break
                elif severity == "warning" and overall_status == "healthy":
                    overall_status = "degraded"

    # Compute execution health signals
    signals_summary = _build_signals_summary()

    # Escalate status based on signals
    if overall_status in ("healthy", "degraded"):
        signal_status = signals_summary.get("status", "OK")
        if signal_status == "CRIT":
            overall_status = "unhealthy"
        elif signal_status == "WARN" and overall_status == "healthy":
            overall_status = "degraded"

    return {
        "status": overall_status,
        "live": state.live,
        "ready": state.ready,
        "reason": state.reason,
        "checks": {
            **state.checks,
            "performance": perf_health,
        },
        "signals": signals_summary,
    }


def _build_signals_summary() -> dict[str, Any]:
    """Build health signals summary from execution metrics.

    Returns:
        Dictionary with status, message, and individual signals.
    """
    try:
        from gpt_trader.monitoring.health_checks import compute_execution_health_signals

        summary = compute_execution_health_signals()
        return summary.to_dict()
    except Exception as exc:
        logger.warning(
            "Failed to compute health signals",
            operation="health_signals",
            error=str(exc),
        )
        return {
            "status": "UNKNOWN",
            "message": f"Error computing signals: {exc}",
            "signals": [],
        }


def _build_liveness_response(state: HealthState) -> dict[str, Any]:
    """Build liveness probe response."""
    return {
        "status": "pass" if state.live else "fail",
        "live": state.live,
    }


def _build_readiness_response(state: HealthState) -> dict[str, Any]:
    """Build readiness probe response."""
    return {
        "status": "pass" if state.ready else "fail",
        "ready": state.ready,
        "reason": state.reason,
    }


def _build_metrics_response() -> str:
    """Build Prometheus metrics response.

    Returns:
        Prometheus text format string.
    """
    from gpt_trader.monitoring.metrics_collector import get_metrics_collector
    from gpt_trader.monitoring.metrics_exporter import format_prometheus

    summary = get_metrics_collector().get_metrics_summary()
    return format_prometheus(summary)


class HealthServer:
    """Async HTTP server for health check endpoints."""

    def __init__(
        self,
        host: str = "0.0.0.0",  # nosec B104 - Health server must bind to all interfaces for container orchestration
        port: int = DEFAULT_HEALTH_PORT,
        health_state: HealthState | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._health_state = health_state or HealthState()
        self._server: asyncio.Server | None = None
        self._running = False

    async def start(self) -> None:
        """Start the health server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_request,
            self.host,
            self.port,
        )
        logger.info(
            "Health server started",
            operation="health_server_start",
            host=self.host,
            port=self.port,
        )

    async def stop(self) -> None:
        """Stop the health server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Health server stopped", operation="health_server_stop")

    async def _handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming HTTP request."""
        try:
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=HEALTH_CHECK_READ_TIMEOUT_SECONDS
            )
            if not request_line:
                return

            request_text = request_line.decode("utf-8").strip()
            parts = request_text.split()
            if len(parts) < 2:
                await self._send_response(
                    writer, HTTPStatus.BAD_REQUEST, {"error": "Invalid request"}
                )
                return

            method, path = parts[0], parts[1]

            # Only handle GET requests
            if method != "GET":
                await self._send_response(
                    writer, HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Method not allowed"}
                )
                return

            # Route to appropriate handler
            if path == "/health":
                response = _build_health_response(self._health_state)
                status = (
                    HTTPStatus.OK
                    if response["status"] in ("healthy", "degraded")
                    else HTTPStatus.SERVICE_UNAVAILABLE
                )
                await self._send_response(writer, status, response)
            elif path == "/live":
                response = _build_liveness_response(self._health_state)
                status = HTTPStatus.OK if response["live"] else HTTPStatus.SERVICE_UNAVAILABLE
                await self._send_response(writer, status, response)
            elif path == "/ready":
                response = _build_readiness_response(self._health_state)
                status = HTTPStatus.OK if response["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
                await self._send_response(writer, status, response)
            elif path == "/metrics":
                if METRICS_ENDPOINT_ENABLED:
                    metrics_text = _build_metrics_response()
                    await self._send_text_response(
                        writer, HTTPStatus.OK, metrics_text, PROMETHEUS_CONTENT_TYPE
                    )
                else:
                    await self._send_response(
                        writer,
                        HTTPStatus.NOT_FOUND,
                        {
                            "error": "Metrics endpoint disabled",
                            "hint": "Set GPT_TRADER_METRICS_ENDPOINT_ENABLED=1",
                        },
                    )
            else:
                await self._send_response(
                    writer, HTTPStatus.NOT_FOUND, {"error": "Not found", "path": path}
                )

        except TimeoutError:
            logger.debug("Health request timeout", operation="health_request")
        except Exception as exc:
            logger.warning(
                "Health request error",
                operation="health_request",
                error=str(exc),
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as cleanup_error:
                logger.debug(
                    "Error closing writer during cleanup: %s",
                    cleanup_error,
                )

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: dict[str, Any],
    ) -> None:
        """Send HTTP JSON response."""
        body_bytes = json.dumps(body, indent=2).encode("utf-8")
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body_bytes

        writer.write(response)
        await writer.drain()

    async def _send_text_response(
        self,
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        """Send HTTP text response (for Prometheus metrics)."""
        body_bytes = body.encode("utf-8")
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body_bytes

        writer.write(response)
        await writer.drain()

    async def run_forever(self) -> None:
        """Run the server until stopped."""
        if self._server:
            await self._server.serve_forever()


# Convenience functions for application integration


async def start_health_server(
    host: str = "0.0.0.0",  # nosec B104 - Health server must bind to all interfaces for container orchestration
    port: int = DEFAULT_HEALTH_PORT,
    health_state: HealthState | None = None,
) -> HealthServer:
    """Start a health server and return the instance."""
    server = HealthServer(host=host, port=port, health_state=health_state)
    await server.start()
    return server


def mark_ready(
    state: HealthState,
    ready: bool = True,
    reason: str = "application_ready",
) -> None:
    """Mark the application as ready to receive traffic."""
    state.set_ready(ready, reason)
    logger.info(
        "Application readiness changed",
        operation="readiness_update",
        ready=ready,
        reason=reason,
    )


def mark_live(state: HealthState, live: bool = True, reason: str = "") -> None:
    """Mark the application as live/alive."""
    state.set_live(live, reason)


def add_health_check(
    state: HealthState,
    name: str,
    check_fn: Callable[[], tuple[bool, dict[str, Any]]],
) -> None:
    """Add a custom health check.

    Args:
        state: HealthState instance to update.
        name: Name of the health check
        check_fn: Function that returns (status, details)
    """
    try:
        status, details = check_fn()
        state.add_check(name, status, details)
    except Exception as exc:
        state.add_check(name, False, {"error": str(exc)})


__all__ = [
    "HealthServer",
    "HealthState",
    "start_health_server",
    "mark_ready",
    "mark_live",
    "add_health_check",
    "DEFAULT_HEALTH_PORT",
]
