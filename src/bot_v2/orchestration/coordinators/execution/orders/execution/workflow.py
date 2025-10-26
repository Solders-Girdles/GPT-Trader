"""Decision execution orchestration logic."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.logging import (
    correlation_context,
    log_execution_error,
    log_order_event,
    symbol_context,
)

from ...logging_utils import json_logger, logger
from . import helpers, kwargs_builder


async def execute_decision(
    mixin: "DecisionExecutionMixin",
    symbol: str,
    decision: Any,
    mark: Decimal,
    product: Product,
    position_state: dict[str, Any] | None,
) -> None:
    ctx = mixin.context
    with correlation_context(operation="execute_decision"), symbol_context(symbol):
        runtime_state_obj = ctx.runtime_state
        if runtime_state_obj is None:
            logger.debug(
                "Runtime state missing; skipping decision execution",
                symbol=symbol,
                operation="execution_decision",
                stage="runtime_state",
            )
            json_logger.debug(
                "Runtime state missing; skipping decision execution",
                extra={
                    "symbol": symbol,
                    "operation": "execution_decision",
                    "stage": "runtime_state",
                },
            )
            return
        try:
            assert product is not None, "Missing product metadata"
            assert mark is not None and mark > 0, f"Invalid mark: {mark}"
            if position_state is not None and "quantity" not in position_state:
                raise AssertionError("Position state missing quantity")

            if ctx.config.dry_run:
                logger.info(
                    "DRY RUN: Would execute %s for %s",
                    decision.action.value,
                    symbol,
                    operation="execution_decision",
                    stage="dry_run",
                    symbol=symbol,
                    action=decision.action.value,
                )
                json_logger.info(
                    f"DRY RUN: Would execute {decision.action.value} for {symbol}",
                    extra={
                        "symbol": symbol,
                        "action": decision.action.value,
                        "operation": "execution_decision",
                        "stage": "dry_run",
                    },
                )
                return

            position_quantity = helpers.extract_position_quantity(position_state)

            if decision.action == Action.CLOSE:
                if not position_state or position_quantity == 0:
                    logger.warning(
                        "No position to close for %s",
                        symbol,
                        operation="execution_decision",
                        stage="close",
                    )
                    return
                order_quantity = abs(position_quantity)
            elif getattr(decision, "target_notional", None):
                order_quantity = Decimal(str(decision.target_notional)) / mark
            elif getattr(decision, "quantity", None) is not None:
                order_quantity = Decimal(str(decision.quantity))
            else:
                logger.warning(
                    "No quantity or notional in decision for %s",
                    symbol,
                    operation="execution_decision",
                    stage="quantity",
                )
                return

            side = helpers.resolve_order_side(decision.action, position_state)
            reduce_only = helpers.resolve_reduce_only(mixin, decision, position_state)
            order_type, limit_price, stop_price, tif = helpers.resolve_order_details(
                decision,
                ctx.config,
            )

            exec_engine = helpers.get_execution_engine(mixin)
            if exec_engine is None:
                logger.warning(
                    "Execution engine not initialized; cannot place order for %s",
                    symbol,
                    operation="execution_decision",
                    stage="engine_missing",
                    symbol=symbol,
                )
                return

            place_kwargs = kwargs_builder.build_order_kwargs(
                exec_engine,
                symbol,
                side,
                order_quantity,
                order_type,
                reduce_only,
                limit_price,
                stop_price,
                tif,
                getattr(decision, "leverage", None),
                product,
            )

            order = await mixin.place_order(exec_engine, **place_kwargs)
            if order:
                logger.info(
                    "Order placed successfully",
                    order_id=str(order.id),
                    symbol=symbol,
                    operation="execution_decision",
                    stage="placed",
                )
                log_order_event(
                    event_type="order_placed",
                    order_id=str(order.id),
                    symbol=symbol,
                    side=order.side.value,
                    quantity=float(place_kwargs.get("quantity", 0)),
                    price=float(limit_price) if limit_price else None,
                )
            else:
                logger.warning(
                    "Order rejected or failed",
                    symbol=symbol,
                    operation="execution_decision",
                    stage="placed",
                )
                json_logger.warning(
                    "Order rejected or failed",
                    extra={
                        "symbol": symbol,
                        "operation": "execution_decision",
                        "stage": "placed",
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Error processing %s: %s",
                symbol,
                exc,
                exc_info=True,
                operation="execution_decision",
                stage="exception",
            )
            log_execution_error(error=exc, operation="execute_decision", symbol=symbol)


__all__ = ["execute_decision"]
