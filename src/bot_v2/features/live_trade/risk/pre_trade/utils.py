"""Utility helpers for pre-trade validation."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_risk")


def coalesce_quantity(*values: Decimal | None) -> Decimal:
    """Pick the first non-None quantity value."""
    for value in values:
        if value is not None:
            if isinstance(value, Decimal):
                return value
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError, TypeError):
                continue
    raise TypeError("quantity must be provided")


def to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely coerce arbitrary inputs to Decimal."""
    if value in (None, "", "null"):
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


__all__ = ["coalesce_quantity", "to_decimal", "logger"]
