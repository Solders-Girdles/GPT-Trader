"""
Simplified trading operations utilities.

This module extracts common trading patterns from legacy code to provide
clean, reusable components for trading operations.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from bot_v2.errors import ExecutionError, NetworkError, ValidationError, log_error
from bot_v2.errors.handler import RecoveryStrategy, get_error_handler
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from bot_v2.utilities.logging_patterns import get_logger, log_operation

logger = get_logger(__name__)


class TradingOperations:
    """Simplified trading operations with consistent error handling and logging."""

    def __init__(self, broker_client: Any, risk_manager: Any) -> None:
        """
        Initialize trading operations.

        Args:
            broker_client: Broker client instance
            risk_manager: Risk manager instance
        """
        self.broker = broker_client
        self.risk_manager = risk_manager
        self.error_handler = get_error_handler()

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal | int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | float | None = None,
        stop_price: Decimal | float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order | None:
        """
        Place an order with standardized error handling and logging.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force

        Returns:
            Order object or None if failed
        """
        with log_operation(
            "place_order", logger, symbol=symbol, side=side.value, quantity=quantity
        ):
            try:
                # Validate inputs
                self._validate_order_inputs(
                    symbol, side, quantity, order_type, limit_price, stop_price, time_in_force
                )

                # Place order with retry logic
                order = self._place_order_with_broker(
                    symbol, side, quantity, order_type, limit_price, stop_price, time_in_force
                )

                if order:
                    logger.info(f"Order placed successfully: {order.id}")
                    return order
                else:
                    raise ExecutionError("Broker returned None for order placement")

            except (ExecutionError, NetworkError) as e:
                log_error(e)
                logger.error(f"Order placement failed: {e.message}")
                return None
            except Exception as e:
                # Re-raise ValidationError as-is to maintain error context
                if isinstance(e, ValidationError):
                    raise
                execution_error = ExecutionError(
                    "Unexpected error during order placement",
                    context={
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": quantity,
                        "original_error": str(e),
                    },
                )
                log_error(execution_error)
                logger.error(f"Order placement failed: {execution_error.message}")
                return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with standardized error handling.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        with log_operation("cancel_order", logger, order_id=order_id):
            try:
                # Validate order ID
                if not order_id or not isinstance(order_id, str):
                    raise ValidationError(
                        "Order ID must be a non-empty string", field="order_id", value=order_id
                    )

                # Cancel order with retry logic
                success = self._cancel_order_with_broker(order_id)

                if success:
                    logger.info(f"Order cancelled successfully: {order_id}")
                else:
                    logger.warning(f"Failed to cancel order: {order_id}")

                return success

            except (ValidationError, NetworkError) as e:
                log_error(e)
                logger.error(f"Order cancellation failed: {e.message}")
                return False
            except Exception as e:
                execution_error = ExecutionError(
                    "Unexpected error during order cancellation",
                    order_id=order_id,
                    context={"original_error": str(e)},
                )
                log_error(execution_error)
                logger.error(f"Order cancellation failed: {execution_error.message}")
                return False

    def get_positions(self) -> list[Any]:
        """
        Get current positions with standardized error handling.

        Returns:
            List of positions or empty list if failed
        """
        with log_operation("get_positions", logger):
            try:
                positions = self._get_positions_with_retry()
                logger.info(f"Retrieved {len(positions)} positions")
                return positions

            except NetworkError as e:
                log_error(e)
                logger.error(f"Failed to get positions: {e.message}")
                return []
            except Exception as e:
                network_error = NetworkError(
                    "Unexpected error retrieving positions", context={"original_error": str(e)}
                )
                log_error(network_error)
                logger.error(f"Failed to get positions: {network_error.message}")
                return []

    def get_account(self) -> Any | None:
        """
        Get account information with standardized error handling.

        Returns:
            Account information or None if failed
        """
        with log_operation("get_account", logger):
            try:
                account = self._get_account_with_retry()
                if account:
                    logger.info(
                        f"Retrieved account info for {getattr(account, 'account_id', 'unknown')}"
                    )
                return account

            except NetworkError as e:
                log_error(e)
                logger.error(f"Failed to get account: {e.message}")
                return None
            except Exception as e:
                network_error = NetworkError(
                    "Unexpected error retrieving account", context={"original_error": str(e)}
                )
                log_error(network_error)
                logger.error(f"Failed to get account: {network_error.message}")
                return None

    def _place_order_with_broker(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal | int,
        order_type: OrderType,
        limit_price: Decimal | float | None,
        stop_price: Decimal | float | None,
        time_in_force: TimeInForce,
    ) -> Order:
        """Place order through broker with retry logic."""

        def _place_order() -> Order:
            return cast(
                Order,
                self.broker.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    time_in_force=time_in_force,
                ),
            )

        return cast(
            Order,
            self.error_handler.with_retry(_place_order, recovery_strategy=RecoveryStrategy.RETRY),
        )

    def _cancel_order_with_broker(self, order_id: str) -> bool:
        """Cancel order through broker with retry logic."""

        def _cancel_order() -> bool:
            return cast(bool, self.broker.cancel_order(order_id))

        return cast(
            bool,
            self.error_handler.with_retry(_cancel_order, recovery_strategy=RecoveryStrategy.RETRY),
        )

    def _get_positions_with_retry(self) -> list[Any]:
        """Get positions from broker with retry logic."""

        def _get_positions() -> list[Any]:
            return cast(list[Any], self.broker.get_positions())

        return cast(
            list[Any],
            self.error_handler.with_retry(_get_positions, recovery_strategy=RecoveryStrategy.RETRY),
        )

    def _get_account_with_retry(self) -> Any:
        """Get account from broker with retry logic."""

        def _get_account() -> Any:
            return self.broker.get_account()

        return self.error_handler.with_retry(_get_account, recovery_strategy=RecoveryStrategy.RETRY)

    def _validate_order_inputs(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal | int,
        order_type: OrderType,
        limit_price: Decimal | float | None,
        stop_price: Decimal | float | None,
        time_in_force: TimeInForce,
    ) -> None:
        """Validate order inputs using existing validators."""
        from bot_v2.validation import ChoiceValidator, PositiveNumberValidator, SymbolValidator

        # Validate symbol
        symbol_validator = SymbolValidator()
        symbol_validator.validate(symbol, "symbol")

        # Validate quantity
        quantity_value = float(quantity)
        quantity_validator = PositiveNumberValidator(allow_zero=False)
        quantity_validator.validate(quantity_value, "quantity")

        # Validate time in force
        tif_validator = ChoiceValidator(["day", "gtc", "ioc", "fok"])
        tif_value = (
            time_in_force.value.lower()
            if isinstance(time_in_force, TimeInForce)
            else str(time_in_force).lower()
        )
        tif_validator.validate(tif_value, "time_in_force")

        # Validate order type requirements
        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValidationError(
                "Limit order requires limit_price", field="limit_price", value=limit_price
            )

        if order_type == OrderType.STOP and stop_price is None:
            raise ValidationError(
                "Stop order requires stop_price", field="stop_price", value=stop_price
            )

        if order_type == OrderType.STOP_LIMIT and (limit_price is None or stop_price is None):
            raise ValidationError(
                "Stop-limit order requires both limit_price and stop_price",
                field="limit_price,stop_price",
                value={"limit_price": limit_price, "stop_price": stop_price},
            )

        # Validate prices when provided
        if limit_price is not None:
            PositiveNumberValidator(allow_zero=False).validate(limit_price, "limit_price")
        if stop_price is not None:
            PositiveNumberValidator(allow_zero=False).validate(stop_price, "stop_price")


class PositionManager:
    """Simplified position management operations."""

    def __init__(self, trading_ops: TradingOperations) -> None:
        """
        Initialize position manager.

        Args:
            trading_ops: Trading operations instance
        """
        self.trading_ops = trading_ops

    def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            True if all positions closed successfully
        """
        logger = get_logger(__name__)

        with log_operation("close_all_positions", logger):
            try:
                positions = self.trading_ops.get_positions()

                if not positions:
                    logger.info("No positions to close")
                    return True

                logger.info(f"Closing {len(positions)} positions")
                success = True
                failed_positions = []

                for position in positions:
                    try:
                        # Determine side for closing
                        close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

                        # Place market order to close
                        order = self.trading_ops.place_order(
                            symbol=position.symbol,
                            side=close_side,
                            quantity=abs(position.quantity),
                            order_type=OrderType.MARKET,
                        )

                        if not order:
                            success = False
                            failed_positions.append(position.symbol)
                            logger.error(f"Failed to close position: {position.symbol}")

                    except Exception as e:
                        success = False
                        failed_positions.append(position.symbol)
                        logger.error(f"Error closing {position.symbol}: {e}")

                if failed_positions:
                    execution_error = ExecutionError(
                        f"Failed to close {len(failed_positions)} positions",
                        context={"failed_positions": failed_positions},
                    )
                    log_error(execution_error)

                return success

            except Exception as e:
                execution_error = ExecutionError(
                    "Unexpected error closing all positions", context={"original_error": str(e)}
                )
                log_error(execution_error)
                logger.error(f"Failed to close positions: {execution_error.message}")
                return False


def create_trading_operations(broker_client: Any, risk_manager: Any) -> TradingOperations:
    """
    Factory function to create trading operations.

    Args:
        broker_client: Broker client instance
        risk_manager: Risk manager instance

    Returns:
        TradingOperations instance
    """
    return TradingOperations(broker_client, risk_manager)


def create_position_manager(trading_ops: TradingOperations) -> PositionManager:
    """
    Factory function to create position manager.

    Args:
        trading_ops: Trading operations instance

    Returns:
        PositionManager instance
    """
    return PositionManager(trading_ops)
