"""
Broker order execution for live trading.

This module handles the actual communication with the broker API,
including async handling and integration test fallbacks.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from decimal import Decimal
from typing import Any

from gpt_trader.features.brokerages.coinbase.rest_service import CoinbaseRestService
from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="broker_executor")


class BrokerExecutor:
    """Executes orders against the broker with integration mode support."""

    def __init__(
        self,
        broker: CoinbaseRestService,
        *,
        integration_mode: bool = False,
    ) -> None:
        """
        Initialize broker executor.

        Args:
            broker: Brokerage adapter for order execution
            integration_mode: Enable integration test mode with fallback handling
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

        Raises:
            TypeError: If broker returns awaitable in non-integration mode
        """
        try:
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
        except TypeError as exc:
            # Handle integration test fallback for legacy signatures
            if not self._integration_mode or "unexpected keyword argument" not in str(exc):
                raise
            order = self._invoke_legacy_place_order(
                submit_id,
                symbol,
                side,
                order_type,
                quantity,
                price,
                stop_price,
                tif,
            )
        else:
            # Handle async results in integration mode
            if inspect.isawaitable(order):
                if not self._integration_mode:
                    raise TypeError("Broker place_order returned awaitable in non-integration mode")
                order = self._await_coroutine(order)

        return order

    def _invoke_legacy_place_order(
        self,
        submit_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: TimeInForce | None,
    ) -> Any:
        """
        Create Order object and call broker with legacy signature.

        Used in integration tests when broker doesn't support keyword arguments.
        """
        tif_value = tif if isinstance(tif, TimeInForce) else TimeInForce.GTC
        now = datetime.utcnow()
        order_obj = Order(
            id=submit_id,
            client_id=submit_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif_value,
            status=OrderStatus.PENDING,
            submitted_at=now,
            updated_at=now,
        )
        result = self._broker.place_order(order_obj)  # type: ignore[arg-type, call-arg]
        if inspect.isawaitable(result):
            return self._await_coroutine(result)
        return result

    @staticmethod
    def _await_coroutine(coro: Any) -> Any:
        """
        Synchronously await a coroutine.

        Used in integration tests when broker returns async results.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
