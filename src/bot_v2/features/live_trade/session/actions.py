"""Order placement and cancellation helpers for the legacy facade."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import cast

from bot_v2.errors import ExecutionError, NetworkError, ValidationError, log_error
from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from bot_v2.validation import PositiveNumberValidator, SymbolValidator

from .account import get_account, get_positions
from .registry import (
    get_broker_client,
    get_connection,
    get_execution_engine,
)

logger = logging.getLogger(__name__)


def place_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal | int,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | float | None = None,
    stop_price: Decimal | float | None = None,
    time_in_force: TimeInForce = TimeInForce.GTC,
) -> Order | None:
    """
    Place an order (template interface).

    Raises:
        ValidationError: If inputs are invalid
        ExecutionError: If order placement fails
        NetworkError: If broker connection issues
    """
    connection = get_connection()
    if not connection or not connection.is_connected:
        raise NetworkError("Not connected to broker")

    execution_engine = get_execution_engine()
    if not execution_engine:
        raise ExecutionError("Execution engine not initialized")

    try:
        symbol_validator = SymbolValidator()
        symbol = symbol_validator.validate(symbol, "symbol")

        quantity_validator = PositiveNumberValidator(allow_zero=False)
        quantity_value = Decimal(str(quantity_validator.validate(quantity, "quantity")))

        account = get_account()
        if not account:
            raise ExecutionError("Unable to retrieve account information")

        limit_price_decimal = Decimal(str(limit_price)) if limit_price is not None else None
        stop_price_decimal = Decimal(str(stop_price)) if stop_price is not None else None

        order = execution_engine.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity_value,
            order_type=order_type,
            limit_price=limit_price_decimal,
            stop_price=stop_price_decimal,
            time_in_force=time_in_force,
        )

        if order:
            logger.info("Order placed successfully: %s", order.id)
            print(f"✅ Order placed: {order.id}")
            print(f"   {side.name} {quantity} {symbol} @ {order_type.name}")
            return order

        raise ExecutionError("Order placement returned None")

    except ValidationError as exc:
        log_error(exc)
        logger.error("Order placement failed: %s", exc.message)
        print(f"❌ Failed to place order: {exc.message}")
        raise
    except (ExecutionError, NetworkError) as exc:
        log_error(exc)
        logger.error("Order placement failed: %s", exc.message)
        print(f"❌ Failed to place order: {exc.message}")
        return None
    except Exception as exc:
        execution_error = ExecutionError(
            "Unexpected error during order placement",
            context={
                "symbol": symbol,
                "side": side,
                "quantity": quantity_value,
                "original_error": str(exc),
            },
        )
        log_error(execution_error)
        logger.error("Unexpected order placement error: %s", execution_error.message)
        print(f"❌ Failed to place order: {execution_error.message}")
        return None


def cancel_order(order_id: str) -> bool:
    """
    Cancel an order.

    Raises:
        ValidationError: If order ID is invalid
        NetworkError: If broker connection issues
    """
    try:
        if not order_id or not isinstance(order_id, str):
            raise ValidationError(
                "Order ID must be a non-empty string", field="order_id", value=order_id
            )

        broker = get_broker_client()
        if broker is None:
            raise NetworkError("Broker client not initialized")
        success = cast(IBrokerage, broker).cancel_order(order_id)

        if success:
            logger.info("Order cancelled successfully: %s", order_id)
            print(f"✅ Order {order_id} cancelled")
            return True

        logger.warning("Failed to cancel order: %s", order_id)
        print(f"❌ Failed to cancel order {order_id}")
        return False

    except (ValidationError, NetworkError) as exc:
        log_error(exc)
        logger.error("Order cancellation failed: %s", exc.message)
        print(f"❌ Failed to cancel order: {exc.message}")
        return False
    except Exception as exc:
        execution_error = ExecutionError(
            "Unexpected error during order cancellation",
            order_id=order_id,
            context={"original_error": str(exc)},
        )
        log_error(execution_error)
        logger.error("Unexpected cancellation error: %s", execution_error.message)
        print(f"❌ Failed to cancel order: {execution_error.message}")
        return False


def close_all_positions() -> bool:
    """
    Close all open positions.

    Returns:
        True if all positions closed successfully
    """
    try:
        positions = get_positions()

        if not positions:
            logger.info("No positions to close")
            print("No positions to close")
            return True

        logger.info("Closing %s positions", len(positions))
        success: bool = True
        failed_positions: list[str] = []

        for position in positions:
            try:
                close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

                order = place_order(
                    symbol=position.symbol,
                    side=close_side,
                    quantity=abs(position.quantity),
                    order_type=OrderType.MARKET,
                )

                if not order:
                    success = False
                else:
                    logger.info("Close order placed for %s: %s", position.symbol, order.id)
                    continue

                failed_positions.append(position.symbol)
                logger.error("Failed to close position: %s", position.symbol)
                print(f"❌ Failed to close {position.symbol}")

            except Exception as exc:
                success = False
                failed_positions.append(position.symbol)
                logger.error("Error closing %s: %s", position.symbol, exc)
                print(f"❌ Error closing {position.symbol}: {exc}")

        if failed_positions:
            execution_error = ExecutionError(
                f"Failed to close {len(failed_positions)} positions",
                context={"failed_positions": failed_positions},
            )
            log_error(execution_error)

        return success

    except Exception as exc:
        execution_error = ExecutionError(
            "Unexpected error closing all positions", context={"original_error": str(exc)}
        )
        log_error(execution_error)
        logger.error("Failed to close positions: %s", execution_error.message)
        print(f"❌ Failed to close positions: {execution_error.message}")
        return False


__all__ = ["place_order", "cancel_order", "close_all_positions"]
