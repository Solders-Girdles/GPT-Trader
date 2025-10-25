"""Lifecycle helpers for order management."""

from __future__ import annotations

import inspect
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Order, OrderStatus


class OrderLifecycleMixin:
    """Shared helpers for tracking and updating order lifecycle state."""

    def _increment_order_stat(self, key: str) -> None:
        runtime_state = self.context.runtime_state
        if runtime_state is None:
            return
        stats = getattr(runtime_state, "order_stats", None)
        if isinstance(stats, dict):
            stats[key] = stats.get(key, 0) + 1

    async def _maybe_record_status_check(self, order: Order) -> None:
        broker = self.context.broker
        if broker is None:
            return
        get_status = getattr(broker, "get_order_status", None)
        if get_status is None:
            return
        try:
            result = get_status(order.id)
            if inspect.isawaitable(result):
                await result  # type: ignore[arg-type]
        except Exception as exc:
            payload = {
                "symbol": order.symbol,
                "order_id": str(order.id),
                "error": str(exc),
            }
            self._record_event("status_update_error", payload)
            self._record_event("system_error", payload)
            self._record_metric("error_count", 1.0, {"type": "status_update_error"})
            self._record_broker_error(exc, symbol=order.symbol, order_id=str(order.id))

    async def cancel_order(self, order_id: str) -> Any:
        from types import SimpleNamespace

        broker = getattr(self, "broker", None) or self.context.broker
        success = False
        if broker is not None and hasattr(broker, "cancel_order"):
            try:
                result = broker.cancel_order(order_id)
                result = await self._await_if_needed(result)
                success = bool(result)
            except Exception as exc:
                self._record_broker_error(exc, order_id=order_id)
        if success:
            self._record_event("order_cancelled", {"order_id": order_id})
        return SimpleNamespace(success=success)

    async def get_order_status(self, order_id: str) -> Any:
        broker = getattr(self, "broker", None) or self.context.broker
        if broker is None or not hasattr(broker, "get_order_status"):
            return None
        try:
            result = broker.get_order_status(order_id)
            return await self._await_if_needed(result)
        except Exception as exc:
            self._record_broker_error(exc, order_id=order_id)
            return None

    async def modify_order(
        self,
        modified_order: Order,
        risk_manager: Any | None = None,
        *,
        exec_engine: Any | None = None,
    ) -> Any:
        from types import SimpleNamespace

        await self.cancel_order(modified_order.id)

        engine = exec_engine or self._last_exec_engine
        if engine is None:
            runtime_state = self.context.runtime_state
            engine = getattr(runtime_state, "exec_engine", None) if runtime_state else None
        if engine is None:
            raise RuntimeError("Execution engine not available for order modification")

        placed = await self.place_order(
            engine,
            symbol=modified_order.symbol,
            side=modified_order.side,
            order_type=modified_order.type,
            quantity=modified_order.quantity,
            price=modified_order.price,
        )

        success = placed is not None and getattr(placed, "status", None) != OrderStatus.REJECTED
        if success:
            self._record_event(
                "order_modified",
                {
                    "order_id": str(getattr(placed, "id", modified_order.id)),
                    "symbol": modified_order.symbol,
                },
            )
        return SimpleNamespace(success=success, order=placed)


__all__ = ["OrderLifecycleMixin"]
