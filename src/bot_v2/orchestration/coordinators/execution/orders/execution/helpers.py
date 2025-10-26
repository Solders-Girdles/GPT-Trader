"""Helper utilities for decision execution workflows."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.strategies.perps_baseline import Action


def get_execution_engine(mixin: "DecisionExecutionMixin") -> Any:
    runtime_state = mixin.context.runtime_state
    if runtime_state is None:
        return None
    return getattr(runtime_state, "exec_engine", None)


def extract_position_quantity(position_state: dict[str, Any] | None) -> Decimal:
    from bot_v2.utilities.quantities import quantity_from

    if not position_state:
        return Decimal("0")
    position_quantity_raw = quantity_from(position_state, default=Decimal("0"))
    if isinstance(position_quantity_raw, Decimal):
        return position_quantity_raw
    try:
        return Decimal(str(position_quantity_raw))
    except Exception:
        return Decimal("0")


def resolve_order_side(action: Action, position_state: dict[str, Any] | None):
    from bot_v2.features.brokerages.core.interfaces import OrderSide

    side = OrderSide.BUY if action == Action.BUY else OrderSide.SELL
    if action == Action.CLOSE:
        side = (
            OrderSide.SELL
            if position_state and position_state.get("side") == "long"
            else OrderSide.BUY
        )
    return side


def resolve_reduce_only(
    mixin: "DecisionExecutionMixin",
    decision: Any,
    position_state: dict[str, Any] | None,
) -> bool:
    reduce_only_global = False
    if getattr(mixin, "_config_controller", None) is not None:
        try:
            reduce_only_global = bool(
                mixin._config_controller.is_reduce_only_mode(mixin.context.risk_manager)
            )
        except Exception:
            reduce_only_global = False

    return (
        getattr(decision, "reduce_only", False)
        or reduce_only_global
        or decision.action == Action.CLOSE
    )


def resolve_order_details(decision: Any, config: Any) -> tuple[Any, Any, Any, Any]:
    from bot_v2.features.brokerages.core.interfaces import OrderType, TimeInForce

    order_type = getattr(decision, "order_type", OrderType.MARKET)
    limit_price = getattr(decision, "limit_price", None)
    stop_price = getattr(decision, "stop_trigger", None)
    tif = getattr(decision, "time_in_force", None)
    try:
        if isinstance(tif, str):
            tif = TimeInForce[tif.upper()]
        elif tif is None and isinstance(config.time_in_force, str):
            tif = TimeInForce[config.time_in_force.upper()]
    except Exception:
        tif = None

    if isinstance(order_type, OrderType):
        normalised_order_type = order_type
    else:
        normalised_order_type = (
            OrderType[order_type.upper()] if isinstance(order_type, str) else OrderType.MARKET
        )

    return normalised_order_type, limit_price, stop_price, tif


__all__ = [
    "get_execution_engine",
    "extract_position_quantity",
    "resolve_order_side",
    "resolve_reduce_only",
    "resolve_order_details",
]
