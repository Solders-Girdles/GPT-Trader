"""Liquidation buffer validation helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk_calculations import effective_symbol_leverage_cap

from ..exceptions import ValidationError
from ..utils import coalesce_quantity, logger


def validate_liquidation_buffer(
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
    """Ensure adequate buffer from liquidation after trade."""
    if price is None or product is None or equity is None:
        raise TypeError("price, product, and equity are required")

    order_qty = coalesce_quantity(quantity, quantity_override)

    if product.market_type != MarketType.PERPETUAL:
        return

    notional = order_qty * price

    max_leverage = effective_symbol_leverage_cap(
        symbol,
        config,
        now=now,
        risk_info_provider=risk_info_provider,
        logger=logger,
    )
    margin_required = notional / max_leverage if max_leverage > 0 else notional

    remaining_equity = equity - margin_required
    buffer_pct = remaining_equity / equity if equity > 0 else Decimal("0")
    buffer_threshold = Decimal(str(config.min_liquidation_buffer_pct))

    if buffer_pct < buffer_threshold:
        raise ValidationError(
            f"Insufficient liquidation buffer for position size: {float(buffer_pct):.1%} < "
            f"{float(buffer_threshold):.1%} required "
            f"(margin needed: {margin_required}, equity: {equity})"
        )


__all__ = ["validate_liquidation_buffer"]
