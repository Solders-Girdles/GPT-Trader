"""Order utility helpers for the advanced execution engine."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import OrderType
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_execution")


def normalize_quantity(quantity: Decimal | int) -> Decimal:
    """Normalize incoming quantity to Decimal."""

    return quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))


def validate_stop_order_requirements(
    engine: "AdvancedExecutionEngine",
    *,
    symbol: str,
    order_type: OrderType,
    stop_price: Decimal | None,
) -> bool:
    """Ensure stop orders satisfy configuration requirements."""

    if order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
        return True

    if not engine.config.enable_stop_orders:
        logger.warning("Stop orders disabled by configuration; rejecting %s", symbol)
        engine.order_metrics["rejected"] += 1
        engine.rejections_by_reason["stop_disabled"] = (
            engine.rejections_by_reason.get("stop_disabled", 0) + 1
        )
        return False

    if stop_price is None:
        logger.warning("Stop order for %s missing stop price", symbol)
        engine.order_metrics["rejected"] += 1
        engine.rejections_by_reason["invalid_stop"] = (
            engine.rejections_by_reason.get("invalid_stop", 0) + 1
        )
        return False

    return True


__all__ = ["normalize_quantity", "validate_stop_order_requirements"]
