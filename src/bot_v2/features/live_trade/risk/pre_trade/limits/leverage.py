"""Leverage limit validation helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk_calculations import effective_symbol_leverage_cap

from ..exceptions import ValidationError
from ..utils import coalesce_quantity, logger


def validate_leverage(
    *,
    config: Any,
    symbol: str,
    quantity: Decimal | None,
    quantity_override: Decimal | None,
    price: Decimal | None,
    product: Product | None,
    equity: Decimal | None,
    now: Any,
    risk_info_provider: Callable[[str], dict[str, Any]] | None,
) -> None:
    """Validate that an order does not exceed leverage limits."""
    if price is None or product is None or equity is None:
        raise TypeError("price, product, and equity are required")

    order_qty = coalesce_quantity(quantity, quantity_override)

    if product.market_type != MarketType.PERPETUAL:
        return

    notional = order_qty * price
    target_leverage = Decimal("Infinity") if equity <= 0 else notional / equity

    symbol_cap = effective_symbol_leverage_cap(
        symbol,
        config,
        now=now,
        risk_info_provider=risk_info_provider,
        logger=logger,
    )

    symbol_cap_decimal = Decimal(str(symbol_cap))
    if target_leverage > symbol_cap_decimal:
        raise ValidationError(
            f"Leverage {float(target_leverage):.1f}x exceeds {symbol} cap of {symbol_cap_decimal}x "
            f"(notional: {notional}, equity: {equity})"
        )

    max_leverage_cap = Decimal(str(config.max_leverage))
    if target_leverage > max_leverage_cap:
        raise ValidationError(
            f"Leverage {float(target_leverage):.1f}x exceeds global cap of {max_leverage_cap}x"
        )


__all__ = ["validate_leverage"]
