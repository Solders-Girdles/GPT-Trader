"""Configuration and sizing modes for the advanced execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class SizingMode(Enum):
    """Position sizing strategy."""

    CONSERVATIVE = "conservative"  # Downsize to fit impact limit
    STRICT = "strict"  # Reject if can't fit
    AGGRESSIVE = "aggressive"  # Allow higher impact


@dataclass
class OrderConfig:
    """Configuration for advanced order workflows."""

    # Limit orders
    enable_limit_orders: bool = True
    limit_price_offset_bps: Decimal = Decimal("5")  # Offset from mid

    # Stop orders
    enable_stop_orders: bool = True
    stop_trigger_offset_pct: Decimal = Decimal("0.02")  # 2% from entry

    # Stop-limit orders
    enable_stop_limit: bool = True
    stop_limit_spread_bps: Decimal = Decimal("10")

    # Post-only protection
    enable_post_only: bool = True
    reject_on_cross: bool = True  # Reject if post-only would cross

    # TIF support
    enable_ioc: bool = True
    enable_fok: bool = False  # Gate until confirmed

    # Sizing
    sizing_mode: SizingMode = SizingMode.CONSERVATIVE
    max_impact_bps: Decimal = Decimal("15")


__all__ = ["SizingMode", "OrderConfig"]
