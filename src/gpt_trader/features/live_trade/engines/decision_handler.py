"""Strategy decision handling for the live TradingEngine.

Turns a strategy Decision (buy/sell/hold) into a sized, validated order through
the canonical guard stack, recording telemetry and decision traces. Extracted
from strategy.py following the engine's collaborator-function pattern.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.profiling import profile_span
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.engines.strategy import TradingEngine

logger = get_logger(__name__, component="trading_engine")


async def handle_decision(
    engine: TradingEngine,
    *,
    symbol: str,
    decision: Decision,
    price: Decimal,
    equity: Decimal,
    position_state: dict[str, Any] | None,
) -> None:
    if decision.action in (Action.BUY, Action.SELL):
        logger.info(
            "Executing order",
            symbol=symbol,
            action=decision.action.value,
            operation="order_placement",
            stage="start",
        )
        try:
            with profile_span(
                "order_placement", {"symbol": symbol, "action": decision.action.value}
            ):
                result = await engine._validate_and_place_order(
                    symbol=symbol,
                    decision=decision,
                    price=price,
                    equity=equity,
                )
            if result.blocked:
                logger.warning(
                    "Order blocked",
                    symbol=symbol,
                    action=decision.action.value,
                    reason=result.reason,
                    operation="order_placement",
                    stage="blocked",
                )
            elif result.failed:
                logger.error(
                    "Order submission failed",
                    symbol=symbol,
                    action=decision.action.value,
                    reason=result.reason,
                    error_message=result.error,
                    operation="order_placement",
                    stage="failed",
                )
                failure_detail = result.error or result.reason or "unknown"
                await engine._notify(
                    title="Order Submission Failed",
                    message=(
                        f"Failed to submit {decision.action.value} order for {symbol}: "
                        f"{failure_detail}"
                    ),
                    severity=AlertSeverity.ERROR,
                    context={
                        "symbol": symbol,
                        "action": decision.action.value,
                        "reason": result.reason,
                        "error": result.error,
                    },
                )
        except Exception as e:
            logger.error(
                "Order placement failed",
                symbol=symbol,
                action=decision.action.value,
                error_message=str(e),
                operation="order_placement",
                stage="failed",
            )
            await engine._notify(
                title="Order Placement Failed",
                message=f"Failed to execute {decision.action} for {symbol}: {e}",
                severity=AlertSeverity.ERROR,
                context={
                    "symbol": symbol,
                    "action": decision.action.value,
                    "error": str(e),
                },
            )
    elif decision.action == Action.CLOSE:
        if position_state is None:
            logger.info(
                "CLOSE signal ignored - no open position",
                symbol=symbol,
                action=decision.action.value,
                operation="order_placement",
                stage="skip",
            )
            return

        close_order = engine._resolve_close_order(position_state)
        if close_order is None:
            logger.warning(
                "CLOSE signal ignored - invalid position state",
                symbol=symbol,
                action=decision.action.value,
                position_state=position_state,
                operation="order_placement",
                stage="invalid_position_state",
            )
            return

        close_side, close_quantity = close_order
        logger.info(
            "Executing close order",
            symbol=symbol,
            action=decision.action.value,
            side=close_side.value,
            quantity=str(close_quantity),
            operation="order_placement",
            stage="start",
        )
        try:
            with profile_span(
                "order_placement",
                {"symbol": symbol, "action": decision.action.value, "side": close_side.value},
            ):
                result = await engine.submit_order(
                    symbol=symbol,
                    side=close_side,
                    price=price,
                    equity=equity,
                    quantity_override=close_quantity,
                    reduce_only=True,
                    reason=decision.reason,
                    confidence=decision.confidence,
                )
            if result.blocked:
                logger.warning(
                    "Close order blocked",
                    symbol=symbol,
                    action=decision.action.value,
                    side=close_side.value,
                    reason=result.reason,
                    operation="order_placement",
                    stage="blocked",
                )
            elif result.failed:
                logger.error(
                    "Close order submission failed",
                    symbol=symbol,
                    action=decision.action.value,
                    side=close_side.value,
                    reason=result.reason,
                    error_message=result.error,
                    operation="order_placement",
                    stage="failed",
                )
                failure_detail = result.error or result.reason or "unknown"
                await engine._notify(
                    title="Close Order Submission Failed",
                    message=f"Failed to close {symbol}: {failure_detail}",
                    severity=AlertSeverity.ERROR,
                    context={
                        "symbol": symbol,
                        "action": decision.action.value,
                        "side": close_side.value,
                        "reason": result.reason,
                        "error": result.error,
                    },
                )
        except Exception as e:
            logger.error(
                "Close order placement failed",
                symbol=symbol,
                action=decision.action.value,
                side=close_side.value,
                error_message=str(e),
                operation="order_placement",
                stage="failed",
            )
            await engine._notify(
                title="Close Order Placement Failed",
                message=f"Failed to close {symbol}: {e}",
                severity=AlertSeverity.ERROR,
                context={
                    "symbol": symbol,
                    "action": decision.action.value,
                    "side": close_side.value,
                    "error": str(e),
                },
            )
