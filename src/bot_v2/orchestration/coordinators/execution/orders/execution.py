"""Decision execution helpers."""

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

from ..logging_utils import json_logger, logger


class DecisionExecutionMixin:
    """Execute strategy decisions through the execution coordinator."""

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        with correlation_context(operation="execute_decision"), symbol_context(symbol):
            ctx = self.context
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

                position_quantity = self._extract_position_quantity(position_state)

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

                side = self._resolve_order_side(decision.action, position_state)
                reduce_only = self._resolve_reduce_only(decision, position_state)
                order_type, limit_price, stop_price, tif = self._resolve_order_details(
                    decision,
                    ctx.config,
                )

                exec_engine = self._get_execution_engine()
                if exec_engine is None:
                    logger.warning(
                        "Execution engine not initialized; cannot place order for %s",
                        symbol,
                        operation="execution_decision",
                        stage="engine_missing",
                        symbol=symbol,
                    )
                    return

                place_kwargs = self._build_order_kwargs(
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

                order = await self.place_order(exec_engine, **place_kwargs)
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
            except Exception as exc:
                logger.error(
                    "Error processing %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                    operation="execution_decision",
                    stage="exception",
                )
                log_execution_error(error=exc, operation="execute_decision", symbol=symbol)

    # ------------------------------------------------------------------
    def _get_execution_engine(self):
        runtime_state = self.context.runtime_state
        if runtime_state is None:
            return None
        return getattr(runtime_state, "exec_engine", None)

    def _extract_position_quantity(self, position_state: dict[str, Any] | None) -> Decimal:
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

    def _resolve_order_side(self, action: Action, position_state: dict[str, Any] | None):
        from bot_v2.features.brokerages.core.interfaces import OrderSide

        side = OrderSide.BUY if action == Action.BUY else OrderSide.SELL
        if action == Action.CLOSE:
            side = (
                OrderSide.SELL
                if position_state and position_state.get("side") == "long"
                else OrderSide.BUY
            )
        return side

    def _resolve_reduce_only(self, decision: Any, position_state: dict[str, Any] | None) -> bool:
        reduce_only_global = False
        if self._config_controller is not None:
            try:
                reduce_only_global = bool(
                    self._config_controller.is_reduce_only_mode(self.context.risk_manager)
                )
            except Exception:
                reduce_only_global = False

        return (
            getattr(decision, "reduce_only", False)
            or reduce_only_global
            or decision.action == Action.CLOSE
        )

    def _resolve_order_details(self, decision: Any, config: Any) -> tuple[Any, Any, Any, Any]:
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

    def _build_order_kwargs(
        self,
        exec_engine: Any,
        symbol: str,
        side: Any,
        quantity: Decimal,
        order_type: Any,
        reduce_only: bool,
        limit_price: Decimal | None,
        stop_price: Decimal | None,
        tif: Any,
        leverage: Any,
        product: Any,
    ) -> dict[str, Any]:
        from bot_v2.features.brokerages.core.interfaces import TimeInForce

        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "reduce_only": reduce_only,
            "leverage": leverage,
        }

        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        if isinstance(exec_engine, AdvancedExecutionEngine):
            kwargs.update(
                {
                    "limit_price": limit_price,
                    "stop_price": stop_price,
                    "time_in_force": tif or TimeInForce.GTC,
                }
            )
        else:
            kwargs.update(
                {
                    "product": product,
                    "price": limit_price,
                    "stop_price": stop_price,
                    "tif": tif or None,
                }
            )
        return kwargs


__all__ = ["DecisionExecutionMixin"]
