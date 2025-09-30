"""Shared decision dataclasses for live trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from bot_v2.features.brokerages.core.interfaces import OrderType, TimeInForce


class Action(Enum):
    """Trading action decisions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass(init=False)
class Decision:
    """Strategy decision output."""

    action: Action
    target_notional: Decimal | None = None
    quantity: Decimal | None = None
    leverage: int | None = None
    stop_params: dict[str, Any] | None = None
    reduce_only: bool = False
    reason: str = ""
    order_type: OrderType | None = None
    stop_trigger: Decimal | None = None
    limit_price: Decimal | None = None
    time_in_force: TimeInForce | None = None
    post_only: bool | None = None
    filter_rejected: bool = False
    guard_rejected: bool = False
    rejection_type: str | None = None

    def __init__(
        self,
        *,
        action: Action,
        target_notional: Decimal | None = None,
        quantity: Decimal | None = None,
        leverage: int | None = None,
        stop_params: dict[str, Any] | None = None,
        reduce_only: bool = False,
        reason: str = "",
        order_type: OrderType | None = None,
        stop_trigger: Decimal | None = None,
        limit_price: Decimal | None = None,
        time_in_force: TimeInForce | None = None,
        post_only: bool | None = None,
        filter_rejected: bool = False,
        guard_rejected: bool = False,
        rejection_type: str | None = None,
    ) -> None:
        self.quantity = Decimal(str(quantity)) if quantity is not None else None

        self.action = action
        self.target_notional = (
            Decimal(str(target_notional)) if target_notional is not None else None
        )
        self.leverage = leverage
        self.stop_params = stop_params
        self.reduce_only = reduce_only
        self.reason = reason
        self.order_type = order_type
        self.stop_trigger = Decimal(str(stop_trigger)) if stop_trigger is not None else None
        self.limit_price = Decimal(str(limit_price)) if limit_price is not None else None
        self.time_in_force = time_in_force
        self.post_only = post_only
        self.filter_rejected = filter_rejected
        self.guard_rejected = guard_rejected
        self.rejection_type = rejection_type


__all__ = ["Action", "Decision"]
