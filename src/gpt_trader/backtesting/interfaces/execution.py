"""Execution interface for backtesting and live trading."""

from typing import Protocol

from gpt_trader.features.brokerages.core.interfaces import Order, Position
from gpt_trader.types.trading import TradeFill as Fill


class IExecution(Protocol):
    """
    Execution interface for order management.

    This interface abstracts order placement and position tracking,
    allowing the same execution logic to work with both live exchange
    APIs and simulated order matching.
    """

    async def place(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        order_type: str,  # "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
        quantity: str,
        limit_price: str | None = None,
        stop_price: str | None = None,
        time_in_force: str | None = None,  # "GTC", "IOC", "FOK"
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a new order.

        Args:
            symbol: Trading pair (e.g., "BTC-PERP-USDC")
            side: "BUY" or "SELL"
            order_type: Order type
            quantity: Order size (in base currency)
            limit_price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)
            time_in_force: Time in force (default: "GTC")
            post_only: Only allow maker orders
            reduce_only: Only reduce existing position
            client_order_id: Client-assigned order ID

        Returns:
            Order object with initial status

        Raises:
            OrderRejectedError: If order is rejected by risk checks or exchange
        """
        ...

    async def cancel(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successfully cancelled, False otherwise

        Raises:
            OrderNotFoundError: If order does not exist
        """
        ...

    async def get_order(self, order_id: str) -> Order | None:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        ...

    async def list_orders(
        self,
        symbol: str | None = None,
        status: str | None = None,
    ) -> list[Order]:
        """
        List orders.

        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional, e.g., "OPEN", "FILLED", "CANCELLED")

        Returns:
            List of orders
        """
        ...

    async def fills(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[Fill]:
        """
        Fetch fill history.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of fills to return

        Returns:
            List of fills sorted by timestamp (descending)
        """
        ...

    async def positions(self) -> list[Position]:
        """
        Get current positions.

        Returns:
            List of open positions
        """
        ...

    async def position(self, symbol: str) -> Position | None:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position object or None if no position
        """
        ...
