"""Internal protocols for Coinbase REST service composition.

These protocols define the contracts between composed services,
replacing the implicit contracts that were hidden in TYPE_CHECKING blocks.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol

from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    TimeInForce,
)


class OrderPayloadBuilder(Protocol):
    """Protocol for building order payloads.

    Implemented by CoinbaseRestServiceCore.
    Used by OrderService to build order payloads without
    knowing the implementation details.
    """

    def build_order_payload(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce | str = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        post_only: bool = False,
        include_client_id: bool = True,
    ) -> dict[str, Any]:
        """Build an order payload for the Coinbase API."""
        ...


class OrderPayloadExecutor(Protocol):
    """Protocol for executing order payloads.

    Implemented by CoinbaseRestServiceCore.
    Used by OrderService to execute order payloads without
    knowing the implementation details.
    """

    def execute_order_payload(
        self, symbol: str, payload: dict[str, Any], client_id: str | None = None
    ) -> Order:
        """Execute an order payload against the Coinbase API."""
        ...


class PositionProvider(Protocol):
    """Protocol for listing positions.

    Implemented by PortfolioService.
    Used by OrderService.close_position() to validate
    that a position exists before closing.
    """

    def list_positions(self) -> list[Position]:
        """List all open positions."""
        ...
