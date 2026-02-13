"""Execution "codec" for translating broker status payloads to persistence-friendly enums."""

from __future__ import annotations

from typing import Any

from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


class ExecutionStatusCodecError(ValueError):
    """Raised when an execution status cannot be translated to persistence."""


_STATUS_MAPPING: dict[str, StoreOrderStatus] = {
    "pending": StoreOrderStatus.PENDING,
    "submitted": StoreOrderStatus.OPEN,
    "open": StoreOrderStatus.OPEN,
    "new": StoreOrderStatus.OPEN,
    "partially_filled": StoreOrderStatus.PARTIALLY_FILLED,
    "partial_fill": StoreOrderStatus.PARTIALLY_FILLED,
    "partial": StoreOrderStatus.PARTIALLY_FILLED,
    "filled": StoreOrderStatus.FILLED,
    "cancelled": StoreOrderStatus.CANCELLED,
    "canceled": StoreOrderStatus.CANCELLED,
    "rejected": StoreOrderStatus.REJECTED,
    "expired": StoreOrderStatus.EXPIRED,
    "failed": StoreOrderStatus.FAILED,
    "retry": StoreOrderStatus.OPEN,
    "retrying": StoreOrderStatus.OPEN,
}


def execution_status_for_store(status: Any, *, context: str | None = None) -> StoreOrderStatus:
    """Return the persistence enum for the given execution-side status."""
    raw = status.value if hasattr(status, "value") else status
    normalized = (
        str(raw).strip().lower().replace("-", "_") if raw is not None else ""
    )
    if not normalized:
        # Missing status defaults to "open" so idempotent writes still succeed.
        return StoreOrderStatus.OPEN
    try:
        return _STATUS_MAPPING[normalized]
    except KeyError as exc:
        context_fragment = f" ({context})" if context else ""
        raise ExecutionStatusCodecError(
            f"Unsupported execution status '{normalized}'{context_fragment}"
        ) from exc


def execution_status_for_event(status: Any, *, context: str | None = None) -> str:
    """Return a serializable status string for events after codec translation."""
    store_status = execution_status_for_store(status, context=context)
    return store_status.value
