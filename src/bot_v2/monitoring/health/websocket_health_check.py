"""WebSocket health check implementation."""

from __future__ import annotations

from typing import Any

from .base import HealthChecker, HealthCheckResult, HealthStatus


class WebSocketReconnectHealthCheck(HealthChecker):
    """Health check for detecting excessive WebSocket reconnections."""

    def __init__(
        self,
        websocket_handler: Any,
        max_reconnects_per_hour: int = 10,
        reconnect_loop_threshold: int = 5,
        name: str = "websocket_reconnects",
    ) -> None:
        super().__init__(name)
        self.websocket_handler = websocket_handler
        self.max_reconnects_per_hour = max_reconnects_per_hour
        self.reconnect_loop_threshold = reconnect_loop_threshold

    async def _do_check(self) -> HealthCheckResult:
        try:
            from datetime import datetime, timedelta

            # Get reconnect history
            reconnect_times = []
            if hasattr(self.websocket_handler, "reconnect_history"):
                reconnect_times = self.websocket_handler.reconnect_history or []
            elif hasattr(self.websocket_handler, "get_reconnect_count"):
                # If only count is available, create synthetic history
                count = self.websocket_handler.get_reconnect_count()
                if count > 0:
                    reconnect_times = [datetime.now()] * count

            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            one_minute_ago = now - timedelta(minutes=1)

            # Count recent reconnects
            recent_hour = sum(1 for t in reconnect_times if t >= one_hour_ago)
            recent_minute = sum(1 for t in reconnect_times if t >= one_minute_ago)

            # Check connection status
            is_connected = False
            if hasattr(self.websocket_handler, "is_connected"):
                is_connected = self.websocket_handler.is_connected()
            elif hasattr(self.websocket_handler, "connected"):
                is_connected = self.websocket_handler.connected

            # Determine health
            if recent_minute >= self.reconnect_loop_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"WebSocket reconnect loop detected: {recent_minute} reconnects in last minute"
            elif recent_hour >= self.max_reconnects_per_hour:
                status = HealthStatus.DEGRADED
                message = f"High reconnect rate: {recent_hour} reconnects in last hour"
            elif not is_connected:
                status = HealthStatus.DEGRADED
                message = "WebSocket disconnected"
            else:
                status = HealthStatus.HEALTHY
                message = f"WebSocket stable: {recent_hour} reconnects in last hour"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "is_connected": is_connected,
                    "reconnects_last_minute": recent_minute,
                    "reconnects_last_hour": recent_hour,
                    "max_reconnects_per_hour": self.max_reconnects_per_hour,
                    "reconnect_loop_threshold": self.reconnect_loop_threshold,
                    "total_reconnects": len(reconnect_times),
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"WebSocket health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )
