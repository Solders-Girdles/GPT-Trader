"""Broker open-order audit orchestration for the live trading engine."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.order_reconciliation import (
    OrderReconciliationService,
)
from gpt_trader.features.live_trade.engines.order_record_mapping import (
    get_order_field,
    parse_timestamp,
)
from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="order_audit")

BrokerCallProvider = Callable[[], Callable[..., Awaitable[Any]]]
AppendEvent = Callable[[str, dict[str, Any]], None]
CycleCountProvider = Callable[[], int]


class OrderAuditService:
    """Fetch broker open orders and feed reconciliation/status reporting."""

    def __init__(
        self,
        *,
        context: CoordinatorContext,
        broker_calls_provider: BrokerCallProvider,
        status_reporter: StatusReporter,
        reconciliation: OrderReconciliationService,
        append_event: AppendEvent,
        cycle_count_provider: CycleCountProvider,
    ) -> None:
        self._context = context
        self._broker_calls_provider = broker_calls_provider
        self._status_reporter = status_reporter
        self._reconciliation = reconciliation
        self._append_event = append_event
        self._cycle_count_provider = cycle_count_provider
        self._unfilled_order_alerts: dict[str, float] = {}

    async def audit_orders(self) -> None:
        """Audit broker open orders for reconciliation and status reporting."""
        broker = self._context.broker
        if broker is None:
            return
        try:
            orders = await self._load_open_orders(broker)
            await self._reconciliation.reconcile_open_orders(orders)

            if orders:
                logger.info(
                    "AUDIT: Found OPEN orders",
                    open_order_count=len(orders),
                    operation="order_audit",
                    stage="list",
                )

            self.record_unfilled_order_alerts(orders)
            self._status_reporter.update_orders(self._status_orders(orders))

            if self._cycle_count_provider() % 60 == 0:
                await self._update_account_metrics(broker)
        except Exception as exc:
            logger.warning(f"Failed to audit orders: {exc}")

    def record_unfilled_order_alerts(self, orders: list[Any]) -> None:
        risk_manager = self._context.risk_manager
        config = risk_manager.config if risk_manager else None
        threshold = getattr(config, "unfilled_order_alert_seconds", 300)
        if not orders or threshold <= 0:
            self._unfilled_order_alerts.clear()
            return

        now = time.time()
        active_ids: set[str] = set()

        for order in orders:
            order_id = get_order_field(order, "order_id", "id", "client_order_id", "client_id")
            if order_id is None:
                continue
            order_id_str = str(order_id)
            active_ids.add(order_id_str)
            if order_id_str in self._unfilled_order_alerts:
                continue

            status = get_order_field(order, "status")
            status_str = (status.value if hasattr(status, "value") else str(status or "")).upper()
            if status_str in {"FILLED", "CANCELLED", "CANCELED", "REJECTED"}:
                continue

            created_at = get_order_field(
                order, "created_time", "created_at", "submitted_at", "created"
            )
            created_ts = parse_timestamp(created_at)
            age_seconds = now - created_ts
            if age_seconds < threshold:
                continue

            payload = {
                "order_id": order_id_str,
                "symbol": get_order_field(order, "product_id", "symbol") or "",
                "side": get_order_field(order, "side") or "",
                "status": status_str,
                "created_time": created_ts,
                "age_seconds": age_seconds,
                "threshold_seconds": threshold,
                "timestamp": now,
            }
            self._append_event("unfilled_order_alert", payload)
            self._unfilled_order_alerts[order_id_str] = now

        stale_ids = [
            order_id for order_id in self._unfilled_order_alerts if order_id not in active_ids
        ]
        for order_id in stale_ids:
            del self._unfilled_order_alerts[order_id]

    async def _load_open_orders(self, broker: Any) -> list[Any]:
        response: Any = None
        module_name = getattr(broker, "__module__", "")
        if "coinbase" in module_name:
            client = getattr(broker, "client", None)
            client_list_orders = getattr(client, "list_orders", None)
            if callable(client_list_orders):
                response = await self._broker_call(client_list_orders, order_status="OPEN")
        if response is None:
            list_orders = getattr(broker, "list_orders", None)
            if callable(list_orders):
                try:
                    response = await self._broker_call(list_orders, order_status="OPEN")
                except TypeError:
                    response = await self._broker_call(list_orders, status=["OPEN"])

        if isinstance(response, dict):
            return list(response.get("orders", []))
        if isinstance(response, list):
            return list(response)
        return []

    async def _update_account_metrics(self, broker: Any) -> None:
        try:
            balances = await self._broker_call(broker.list_balances)
            summary = {}
            if hasattr(broker, "client") and hasattr(broker.client, "get_transaction_summary"):
                try:
                    summary = await self._broker_call(broker.client.get_transaction_summary)
                except Exception:
                    pass

            self._status_reporter.update_account(balances, summary)
        except Exception as exc:
            logger.warning(f"Failed to update account metrics: {exc}")

    def _status_orders(self, orders: list[Any]) -> list[dict[str, Any]]:
        status_orders: list[dict[str, Any]] = []
        for order in orders:
            if isinstance(order, dict):
                status_orders.append(order)
                continue
            status_orders.append(
                {
                    "order_id": get_order_field(order, "order_id", "id") or "",
                    "product_id": get_order_field(order, "product_id", "symbol") or "",
                    "side": get_order_field(order, "side") or "",
                    "status": get_order_field(order, "status") or "",
                    "price": get_order_field(order, "price"),
                    "size": get_order_field(order, "size", "quantity"),
                    "created_time": get_order_field(order, "created_time", "created_at"),
                    "filled_size": get_order_field(order, "filled_size", "filled_quantity"),
                    "average_filled_price": get_order_field(
                        order,
                        "average_filled_price",
                        "avg_fill_price",
                    ),
                }
            )
        return status_orders

    async def _broker_call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._broker_calls_provider()(function, *args, **kwargs)
