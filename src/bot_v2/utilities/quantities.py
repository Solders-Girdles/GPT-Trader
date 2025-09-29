"""Utility helpers for normalising quantity fields."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def quantity_from(source: Any, default: Decimal | None = Decimal("0")) -> Decimal | None:
    """Normalise quantity fields across common payload shapes."""

    if source is None:
        return default

    if isinstance(source, (Decimal, int, float, str)):
        try:
            return Decimal(str(source))
        except Exception:
            return default

    raw = getattr(source, "quantity", None)
    if raw is None:
        raw = getattr(source, "size", None)
    if raw is None:
        raw = getattr(source, "contracts", None)

    if raw is None and isinstance(source, dict):
        raw = source.get("quantity")
        if raw is None:
            raw = source.get("size")
        if raw is None:
            raw = source.get("contracts")

    if raw is None:
        return default

    try:
        return Decimal(str(raw))
    except Exception:
        return default


__all__ = ["quantity_from"]
