"""Order-value extraction and coercion helpers.

Pure helpers for reading identifier and decimal fields out of broker order
objects (Mappings or attribute objects), used by the emergency-close / flatten
path. Extracted from TradingBot so the coercion logic is independently testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any


def extract_order_identifier(order: Any | None, field_names: tuple[str, ...]) -> str | None:
    if order is None:
        return None
    if isinstance(order, Mapping):
        for field_name in field_names:
            value = order.get(field_name)
            if value is not None:
                return str(getattr(value, "value", value))
        return None
    for field_name in field_names:
        value = getattr(order, field_name, None)
        if value is not None:
            return str(getattr(value, "value", value))
    return None


def extract_order_decimal(
    order: Any | None,
    field_names: tuple[str, ...],
) -> Decimal | None:
    if order is None:
        return None
    if isinstance(order, Mapping):
        values = (order.get(field_name) for field_name in field_names)
    else:
        values = (getattr(order, field_name, None) for field_name in field_names)
    for value in values:
        decimal_value = decimal_or_none(value)
        if decimal_value is not None:
            return decimal_value
    return None


def stringify_order_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(getattr(value, "value", value)))
    except Exception:
        return None


def decimal_or_zero(value: Any) -> Decimal:
    return decimal_or_none(value) or Decimal("0")
