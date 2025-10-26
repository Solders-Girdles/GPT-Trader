"""Order placement mixin wrapper that leverages modular helpers."""

from __future__ import annotations

from typing import Any

from . import errors, finalization, workflow
from .resolution import resolve_order_from_result
from ..lifecycle import OrderLifecycleMixin


class OrderPlacementMixin(OrderLifecycleMixin):
    """Encapsulate order placement and reconciliation helpers."""

    async def place_order(self, exec_engine: Any, **kwargs: Any) -> Any:
        return await workflow.place_order(self, exec_engine, **kwargs)

    async def place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Any:
        return await workflow.place_order_inner(self, exec_engine, **kwargs)

    async def _resolve_order_from_result(self, exec_engine: Any, broker: Any, result: Any):
        return await resolve_order_from_result(self, exec_engine, broker, result)

    async def _finalize_successful_order(self, order, original_kwargs):
        return await finalization.finalize_successful_order(self, order, original_kwargs)

    def _normalize_partial_fill(self, order):
        return finalization.normalize_partial_fill(order)

    def _handle_failed_order(self, order, status_name: str) -> None:
        finalization.handle_failed_order(self, order, status_name)

    def _record_successful_order(self, order, kwargs):
        finalization.record_successful_order(self, order, kwargs)

    def _handle_order_error(self, symbol: str, kwargs: dict[str, Any], exc: Exception) -> None:
        errors.handle_order_error(self, symbol, kwargs, exc)

    def _handle_risk_callback(self, exc: Exception, symbol: str) -> None:
        errors.handle_risk_callback(self, exc, symbol)


__all__ = ["OrderPlacementMixin"]
