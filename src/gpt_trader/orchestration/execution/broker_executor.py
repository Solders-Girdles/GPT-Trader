"""
Broker order execution for live trading.

This module handles the actual communication with the broker API.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.core import OrderSide, OrderType
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol

logger = get_logger(__name__, component="broker_executor")


class BrokerExecutor:
    """Executes orders against the broker."""

    def __init__(
        self,
        broker: BrokerProtocol,
        *,
        integration_mode: bool = False,
    ) -> None:
        """
        Initialize broker executor.

        Args:
            broker: Brokerage adapter for order execution
            integration_mode: Enable integration test mode (reserved for future use)
        """
        self._broker = broker
        self._integration_mode = integration_mode

    def execute_order(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
    ) -> Any:
        """
        Execute order placement against the broker.

        Args:
            submit_id: Client order ID
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/etc.)
            quantity: Order quantity
            price: Limit price (None for market orders)
            stop_price: Stop price for stop orders
            tif: Time in force
            reduce_only: Whether order is reduce-only
            leverage: Leverage multiplier

        Returns:
            Order object from broker
        """
        order = self._broker.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif if tif is not None else None,
            reduce_only=reduce_only,
            leverage=leverage,
            client_id=submit_id,
        )
        return order
