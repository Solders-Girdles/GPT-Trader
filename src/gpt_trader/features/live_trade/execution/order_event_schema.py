"""Typed normalization helpers for order event payloads."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping


class OrderEventSchemaError(ValueError):
    """Raised when an order event payload contains invalid or missing data."""


def _coerce_identifier(
    value: Any, *, field_name: str, event_type: str, allow_empty: bool = False
) -> str:
    if value is None:
        raise OrderEventSchemaError(f"{event_type}: missing required field '{field_name}'")
    normalized = getattr(value, "value", value)
    text = str(normalized)
    if not allow_empty and not text:
        raise OrderEventSchemaError(
            f"{event_type}: '{field_name}' must be a non-empty string, got {normalized!r}"
        )
    return text


def _coerce_decimal(value: Any, *, field_name: str, event_type: str) -> str:
    if value is None:
        raise OrderEventSchemaError(f"{event_type}: missing required field '{field_name}'")
    try:
        return str(Decimal(value))
    except Exception as exc:  # pragma: no cover - defensive
        raise OrderEventSchemaError(
            f"{event_type}: invalid decimal value for '{field_name}' ({value!r})"
        ) from exc


def _format_price(value: Any, *, event_type: str) -> str:
    if value is None:
        return "market"
    return _coerce_decimal(value, field_name="price", event_type=event_type)


def _normalize_preview_data(preview: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
    if not isinstance(preview, Mapping):
        raise OrderEventSchemaError(
            f"{event_type}: 'preview' must be a mapping/dict, got {type(preview).__name__}"
        )
    return dict(preview)


@dataclass(frozen=True)
class OrderPreviewEvent:
    symbol: Any
    side: Any
    order_type: Any
    quantity: Any
    price: Any | None
    preview: Mapping[str, Any]

    def serialize(self) -> dict[str, Any]:
        event_type = "order_preview"
        return {
            "event_type": event_type,
            "symbol": _coerce_identifier(self.symbol, field_name="symbol", event_type=event_type),
            "side": _coerce_identifier(self.side, field_name="side", event_type=event_type),
            "order_type": _coerce_identifier(
                self.order_type, field_name="order_type", event_type=event_type
            ),
            "quantity": _coerce_decimal(self.quantity, field_name="quantity", event_type=event_type),
            "price": _format_price(self.price, event_type=event_type),
            "preview": _normalize_preview_data(self.preview, event_type=event_type),
        }


@dataclass(frozen=True)
class OrderSubmissionAttemptEvent:
    client_order_id: Any
    symbol: Any
    side: Any
    order_type: Any
    quantity: Any
    price: Any | None

    def serialize(self) -> dict[str, Any]:
        event_type = "order_submission_attempt"
        return {
            "event_type": event_type,
            "client_order_id": _coerce_identifier(
                self.client_order_id,
                field_name="client_order_id",
                event_type=event_type,
                allow_empty=True,
            ),
            "symbol": _coerce_identifier(self.symbol, field_name="symbol", event_type=event_type),
            "side": _coerce_identifier(self.side, field_name="side", event_type=event_type),
            "order_type": _coerce_identifier(
                self.order_type, field_name="order_type", event_type=event_type
            ),
            "quantity": _coerce_decimal(self.quantity, field_name="quantity", event_type=event_type),
            "price": _format_price(self.price, event_type=event_type),
        }


@dataclass(frozen=True)
class OrderRejectionEvent:
    symbol: Any
    side: Any
    quantity: Any
    price: Any | None
    reason: Any
    reason_detail: Any | None
    client_order_id: Any

    def serialize(self) -> dict[str, Any]:
        event_type = "order_rejected"
        reason_text = _coerce_identifier(
            self.reason, field_name="reason", event_type=event_type
        )
        payload: dict[str, Any] = {
            "event_type": event_type,
            "symbol": _coerce_identifier(self.symbol, field_name="symbol", event_type=event_type),
            "side": _coerce_identifier(self.side, field_name="side", event_type=event_type),
            "quantity": _coerce_decimal(
                self.quantity, field_name="quantity", event_type=event_type
            ),
            "price": _format_price(self.price, event_type=event_type),
            "reason": reason_text,
            "reason_detail": self.reason_detail,
            "client_order_id": _coerce_identifier(
                self.client_order_id,
                field_name="client_order_id",
                event_type=event_type,
                allow_empty=True,
            ),
        }
        return payload


__all__ = [
    "OrderEventSchemaError",
    "OrderPreviewEvent",
    "OrderSubmissionAttemptEvent",
    "OrderRejectionEvent",
]
