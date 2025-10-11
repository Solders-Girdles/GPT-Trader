"""Common sizing and decision helpers for live-trade strategies."""

from __future__ import annotations

from collections.abc import MutableMapping
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.utilities.quantities import quantity_from


def create_entry_decision(
    *,
    symbol: str,
    action: Action,
    equity: Decimal,
    product: Product,
    position_fraction: float,
    target_leverage: int,
    max_trade_usd: Decimal | None,
    position_adds: MutableMapping[str, int],
    trailing_stops: MutableMapping[str, tuple[Decimal, Decimal]],
    reason: str,
) -> Decision:
    """Return a Decision for opening a position with standardized sizing."""
    fraction = Decimal(str(position_fraction or 0))
    if fraction <= Decimal("0"):
        fraction = Decimal("0.05")

    target_notional = equity * fraction

    if max_trade_usd is not None:
        try:
            cap = Decimal(str(max_trade_usd))
            target_notional = min(target_notional, cap)
        except Exception:
            pass

    leverage_value: int | None = None
    if product.market_type == MarketType.PERPETUAL:
        try:
            lv = Decimal(str(target_leverage))
            if lv > Decimal("1"):
                target_notional = target_notional * lv
        except Exception:
            lv = Decimal("1")
        try:
            leverage_value = int(target_leverage)
        except Exception:
            leverage_value = None

    position_adds[symbol] = 0
    trailing_stops.pop(symbol, None)

    return Decision(
        action=action,
        target_notional=target_notional,
        leverage=leverage_value,
        reason=reason,
    )


def create_close_decision(
    *,
    symbol: str,
    position_state: dict,
    position_adds: MutableMapping[str, int],
    trailing_stops: MutableMapping[str, tuple[Decimal, Decimal]],
    reason: str,
) -> Decision:
    """Return a reduce-only close decision and reset tracking state."""
    position_adds.pop(symbol, None)
    trailing_stops.pop(symbol, None)

    raw_quantity = quantity_from(position_state)
    close_quantity = abs(raw_quantity) if raw_quantity is not None else Decimal("0")

    return Decision(
        action=Action.CLOSE,
        quantity=close_quantity,
        reduce_only=True,
        reason=reason,
    )


__all__ = ["create_entry_decision", "create_close_decision"]
