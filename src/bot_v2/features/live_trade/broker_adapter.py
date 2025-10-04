"""
Broker Adapter for Order Submission.

Handles broker-specific parameter mapping and API interaction,
enabling support for multiple exchange brokers with different signatures.
"""

from __future__ import annotations

import inspect
import logging
from decimal import Decimal
from typing import Any, cast

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class BrokerAdapter:
    """
    Adapter for broker-specific order submission.

    Handles parameter mapping for different broker API signatures,
    including TimeInForce conversion and parameter name variations.
    """

    def __init__(self, broker: Any) -> None:
        """
        Initialize broker adapter.

        Args:
            broker: Broker instance with place_order method
        """
        self.broker = broker
        logger.info("BrokerAdapter initialized")

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        client_id: str,
        reduce_only: bool,
        leverage: int | None,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order | None:
        """
        Submit order to broker with parameter mapping.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type
            quantity: Order quantity
            client_id: Client order ID
            reduce_only: Whether order can only reduce position
            leverage: Leverage multiplier (for derivatives)
            limit_price: Limit price for LIMIT/STOP_LIMIT orders
            stop_price: Stop price for STOP/STOP_LIMIT orders
            time_in_force: Time-in-force policy

        Returns:
            Order instance if successful, None if failed
        """
        try:
            # Get broker's place_order signature
            broker_place = getattr(self.broker, "place_order")
            params = inspect.signature(broker_place).parameters

            # Check if this is a Mock (has *args, **kwargs) or real broker
            is_mock = any(str(p).startswith("*") for p in params.values())

            # Build base kwargs
            kwargs: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "client_id": client_id,
                "reduce_only": reduce_only,
                "leverage": leverage,
            }

            # Map limit_price parameter (different brokers use different names)
            if limit_price is not None:
                if "limit_price" in params or is_mock:
                    kwargs["limit_price"] = limit_price
                elif "price" in params:
                    kwargs["price"] = limit_price

            # Map stop_price parameter
            if stop_price is not None:
                if "stop_price" in params or is_mock:
                    kwargs["stop_price"] = stop_price

            # Convert TimeInForce to appropriate format
            tif_value_enum, tif_value_str = self._convert_time_in_force(time_in_force)

            # Map TIF parameter (different brokers use different names/types)
            if "time_in_force" in params or is_mock:
                kwargs["time_in_force"] = tif_value_str
            if "tif" in params or is_mock:
                kwargs["tif"] = tif_value_enum

            # Submit order
            order = cast(Order | None, broker_place(**kwargs))

            if order:
                logger.info(
                    f"Order submitted via broker: {order.id} - "
                    f"{side.value} {quantity} {symbol} @ {limit_price or 'market'}"
                )

            return order

        except Exception as exc:
            logger.error(
                f"Broker order submission failed: {exc}",
                exc_info=True,
                extra={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                    "order_type": order_type.value,
                },
            )
            raise

    def _convert_time_in_force(self, time_in_force: TimeInForce | str) -> tuple[TimeInForce, str]:
        """
        Convert TimeInForce to both enum and string formats.

        Different brokers expect different formats (enum vs string),
        so we provide both.

        Args:
            time_in_force: TimeInForce enum or string

        Returns:
            Tuple of (TimeInForce enum, string value)
        """
        if isinstance(time_in_force, TimeInForce):
            tif_value_enum = time_in_force
            tif_value_str = time_in_force.value
        else:
            # Convert string to enum
            try:
                tif_value_enum = TimeInForce[str(time_in_force).upper()]
            except (KeyError, ValueError):
                logger.warning(f"Invalid TimeInForce '{time_in_force}', defaulting to GTC")
                tif_value_enum = TimeInForce.GTC
            tif_value_str = tif_value_enum.value

        return tif_value_enum, tif_value_str
