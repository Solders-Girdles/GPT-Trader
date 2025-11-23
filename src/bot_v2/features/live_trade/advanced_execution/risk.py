"""Risk validation helpers for the advanced execution engine."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Product, Quote

if TYPE_CHECKING:
    from .engine import AdvancedExecutionEngine
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_execution")


def run_risk_validation(
    engine: "AdvancedExecutionEngine",
    *,
    symbol: str,
    side: OrderSide,
    order_quantity: Decimal,
    limit_price: Decimal | None,
    order_type: OrderType,
    product: Product | None,
    quote: Quote | None,
) -> bool:
    """Execute pre-trade risk validation when a risk manager is configured."""

    if engine.risk_manager is None:
        return True
    try:
        validation_price = engine.position_sizer.determine_reference_price(
            symbol=symbol,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            quote=quote,
            product=product,
        )
        equity = engine.position_sizer.estimate_equity()
        current_positions_raw = engine.position_sizer.current_positions()
        current_positions = dict(current_positions_raw) if current_positions_raw else None
        engine.risk_manager.pre_trade_validate(
            symbol=symbol,
            side=side.value,
            quantity=order_quantity,
            price=validation_price,
            product=product,
            equity=equity,
            current_positions=current_positions,
        )
        return True
    except ValidationError as exc:
        logger.warning(f"Risk validation failed for {symbol}: {exc}")
        engine.order_metrics["rejected"] += 1
        engine.rejections_by_reason["risk"] = engine.rejections_by_reason.get("risk", 0) + 1
        return False


__all__ = ["run_risk_validation"]
