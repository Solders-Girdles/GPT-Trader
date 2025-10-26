"""Enumerations for order policy capabilities."""

from __future__ import annotations

from enum import Enum


class OrderTypeSupport(Enum):
    """Order type support levels."""

    SUPPORTED = "supported"
    GATED = "gated"
    UNSUPPORTED = "unsupported"


class TIFSupport(Enum):
    """Time-in-force support levels."""

    SUPPORTED = "supported"
    GATED = "gated"
    UNSUPPORTED = "unsupported"


__all__ = ["OrderTypeSupport", "TIFSupport"]
