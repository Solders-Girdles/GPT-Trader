"""Concrete health check implementations."""

from __future__ import annotations

import time
from collections.abc import Callable
from numbers import Real
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from bot_v2.utilities.performance_monitoring import get_performance_health_check

from .base import HealthChecker, HealthCheckResult, HealthStatus


class DatabaseHealthCheck(HealthChecker):
    """Health check for database connectivity."""

    def __init__(self, connection_factory: Callable[[], Any], name: str = "database") -> None:
        super().__init__(name)
        self.connection_factory = connection_factory

    async def _do_check(self) -> HealthCheckResult:
        try:
            connection = self.connection_factory()
            start_time = time.time()
            await connection.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={
                    "query_time_ms": query_time,
                    "connection_type": type(connection).__name__,
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class APIHealthCheck(HealthChecker):
    """Health check for external API connectivity."""

    def __init__(self, api_client: Any, endpoint: str = "/health", name: str = "api") -> None:
        super().__init__(name)
        self.api_client = api_client
        self.endpoint = endpoint

    async def _do_check(self) -> HealthCheckResult:
        try:
            start_time = time.time()
            response = await self.api_client.get(self.endpoint)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "API endpoint responsive"
            else:
                status = HealthStatus.DEGRADED
                message = f"API returned status {response.status_code}"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "endpoint": self.endpoint,
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"API health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class BrokerageHealthCheck(HealthChecker):
    """Health check for brokerage connectivity."""

    def __init__(self, brokerage: Any, name: str = "brokerage") -> None:
        super().__init__(name)
        self.brokerage = brokerage

    async def _do_check(self) -> HealthCheckResult:
        try:
            if not self.brokerage.validate_connection():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Brokerage not connected",
                )

            start_time = time.time()
            balances = self.brokerage.list_balances()
            response_time = (time.time() - start_time) * 1000
            products = self.brokerage.list_products()

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Brokerage connection healthy",
                details={
                    "response_time_ms": response_time,
                    "balance_count": len(balances),
                    "product_count": len(products),
                    "account_id": self.brokerage.get_account_id(),
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Brokerage health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class MemoryHealthCheck(HealthChecker):
    """Health check for memory usage."""

    def __init__(
        self,
        warning_threshold_mb: float = 1000.0,
        critical_threshold_mb: float = 2000.0,
        name: str = "memory",
    ) -> None:
        super().__init__(name)
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb

    async def _do_check(self) -> HealthCheckResult:
        try:
            import sys

            module = sys.modules.get("bot_v2.monitoring.health_checks")
            psutil_module = getattr(module, "psutil", psutil)
            if psutil_module is None:
                raise ImportError

            try:
                process = psutil_module.Process()
            except Exception as exc:
                raise ImportError from exc
            memory_info = process.memory_info()
            rss_value = getattr(memory_info, "rss", None)
            if not isinstance(rss_value, Real):
                raise ImportError
            memory_mb = float(rss_value) / (1024 * 1024)

            if memory_mb > self.critical_threshold_mb:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_mb:.1f}MB"
            elif memory_mb > self.warning_threshold_mb:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.1f}MB"

            try:
                memory_percent = float(process.memory_percent())
            except Exception:
                memory_percent = 0.0

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "memory_mb": round(memory_mb, 1),
                    "warning_threshold_mb": self.warning_threshold_mb,
                    "critical_threshold_mb": self.critical_threshold_mb,
                    "memory_percent": memory_percent,
                },
            )
        except ImportError:  # pragma: no cover - optional dependency
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for memory monitoring",
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Memory health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class PerformanceHealthCheck(HealthChecker):
    """Health check based on performance metrics."""

    def __init__(
        self,
        slow_operation_threshold_s: float = 1.0,
        very_slow_operation_threshold_s: float = 5.0,
        name: str = "performance",
    ) -> None:
        super().__init__(name)
        self.slow_operation_threshold_s = slow_operation_threshold_s
        self.very_slow_operation_threshold_s = very_slow_operation_threshold_s

    async def _do_check(self) -> HealthCheckResult:
        try:
            import sys

            module = sys.modules.get("bot_v2.monitoring.health_checks")
            perf_fn = getattr(module, "get_performance_health_check", get_performance_health_check)
            perf_health = perf_fn()
            status = HealthStatus(perf_health["status"])
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=f"Performance status: {perf_health['status']}",
                details={
                    "issues": perf_health["issues"],
                    "metrics": perf_health["metrics"],
                    "slow_threshold_s": self.slow_operation_threshold_s,
                    "very_slow_threshold_s": self.very_slow_operation_threshold_s,
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Performance health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class StaleFillsHealthCheck(HealthChecker):
    """Health check for detecting unfilled orders that are too old."""

    def __init__(
        self,
        orders_store: Any,
        max_age_minutes: float = 10.0,
        name: str = "stale_fills",
    ) -> None:
        super().__init__(name)
        self.orders_store = orders_store
        self.max_age_minutes = max_age_minutes

    async def _do_check(self) -> HealthCheckResult:
        try:
            from datetime import datetime, timedelta

            # Get all open orders
            open_orders = getattr(self.orders_store, "get_open_orders", lambda: [])()
            if not open_orders:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="No open orders",
                    details={"open_orders": 0},
                )

            # Check for stale orders
            now = datetime.now()
            cutoff = now - timedelta(minutes=self.max_age_minutes)
            stale_orders = []

            for order in open_orders:
                # Get order timestamp
                order_time = None
                if hasattr(order, "created_at"):
                    order_time = order.created_at
                elif isinstance(order, dict):
                    ts = order.get("created_at") or order.get("timestamp")
                    if ts:
                        if isinstance(ts, str):
                            order_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        else:
                            order_time = ts

                if order_time and order_time < cutoff:
                    stale_orders.append({
                        "order_id": getattr(order, "id", order.get("id", "unknown")),
                        "symbol": getattr(order, "symbol", order.get("symbol", "unknown")),
                        "age_minutes": (now - order_time).total_seconds() / 60,
                    })

            if len(stale_orders) >= 5:
                status = HealthStatus.UNHEALTHY
                message = f"{len(stale_orders)} orders unfilled for >{self.max_age_minutes}min"
            elif stale_orders:
                status = HealthStatus.DEGRADED
                message = f"{len(stale_orders)} orders unfilled for >{self.max_age_minutes}min"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {len(open_orders)} open orders are recent"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "open_orders": len(open_orders),
                    "stale_orders": len(stale_orders),
                    "max_age_minutes": self.max_age_minutes,
                    "stale_order_details": stale_orders[:5],  # First 5
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Stale fills health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


class StaleMarksHealthCheck(HealthChecker):
    """Health check for detecting stale market data marks."""

    def __init__(
        self,
        market_data_service: Any,
        max_age_seconds: float = 30.0,
        name: str = "stale_marks",
    ) -> None:
        super().__init__(name)
        self.market_data_service = market_data_service
        self.max_age_seconds = max_age_seconds

    async def _do_check(self) -> HealthCheckResult:
        try:
            from datetime import datetime, timedelta

            # Get current marks
            marks = {}
            if hasattr(self.market_data_service, "get_all_marks"):
                marks = self.market_data_service.get_all_marks()
            elif hasattr(self.market_data_service, "marks"):
                marks = self.market_data_service.marks or {}

            if not marks:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="No market marks available",
                    details={"symbols": 0},
                )

            # Check staleness
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.max_age_seconds)
            stale_symbols = []
            total_symbols = len(marks)

            for symbol, mark_data in marks.items():
                # Extract timestamp
                mark_time = None
                if isinstance(mark_data, dict):
                    ts = mark_data.get("timestamp") or mark_data.get("time")
                    if ts:
                        if isinstance(ts, str):
                            mark_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        else:
                            mark_time = ts
                elif hasattr(mark_data, "timestamp"):
                    mark_time = mark_data.timestamp

                if mark_time and mark_time < cutoff:
                    stale_symbols.append({
                        "symbol": symbol,
                        "age_seconds": (now - mark_time).total_seconds(),
                    })

            stale_pct = (len(stale_symbols) / total_symbols * 100) if total_symbols > 0 else 0

            if stale_pct > 50:
                status = HealthStatus.UNHEALTHY
                message = f"{len(stale_symbols)}/{total_symbols} marks stale (>{self.max_age_seconds}s)"
            elif stale_symbols:
                status = HealthStatus.DEGRADED
                message = f"{len(stale_symbols)}/{total_symbols} marks stale (>{self.max_age_seconds}s)"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {total_symbols} marks are fresh"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "total_symbols": total_symbols,
                    "stale_symbols": len(stale_symbols),
                    "stale_percentage": round(stale_pct, 1),
                    "max_age_seconds": self.max_age_seconds,
                    "stale_symbol_details": stale_symbols[:5],  # First 5
                },
            )
        except Exception as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Stale marks health check failed: {exc}",
                details={"error_type": type(exc).__name__},
            )


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


__all__ = [
    "DatabaseHealthCheck",
    "APIHealthCheck",
    "BrokerageHealthCheck",
    "MemoryHealthCheck",
    "PerformanceHealthCheck",
    "StaleFillsHealthCheck",
    "StaleMarksHealthCheck",
    "WebSocketReconnectHealthCheck",
]
