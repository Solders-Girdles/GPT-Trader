"""Market data health check implementations (stale fills and marks)."""

from __future__ import annotations

from typing import Any

from .base import HealthChecker, HealthCheckResult, HealthStatus


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
