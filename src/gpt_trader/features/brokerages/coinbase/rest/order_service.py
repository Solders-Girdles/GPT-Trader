"""Order management service for Coinbase REST API.

This service handles order operations with explicit dependencies
injected via constructor, replacing the OrderRestMixin.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.errors import (
    BrokerageError,
    NotFoundError,
    OrderCancellationError,
    OrderQueryError,
)
from gpt_trader.features.brokerages.coinbase.models import to_order
from gpt_trader.features.brokerages.coinbase.rest.protocols import (
    OrderPayloadBuilder,
    OrderPayloadExecutor,
    PositionProvider,
)
from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_orders")


class OrderService:
    """Handles order management operations.

    Dependencies:
        client: CoinbaseClient for API calls
        payload_builder: Builds order payloads (implements OrderPayloadBuilder)
        payload_executor: Executes order payloads (implements OrderPayloadExecutor)
        position_provider: Lists positions for close_position validation
    """

    def __init__(
        self,
        *,
        client: CoinbaseClient,
        payload_builder: OrderPayloadBuilder,
        payload_executor: OrderPayloadExecutor,
        position_provider: PositionProvider,
    ) -> None:
        self._client = client
        self._payload_builder = payload_builder
        self._payload_executor = payload_executor
        self._position_provider = position_provider

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        post_only: bool = False,
    ) -> Order:
        """Place a new order."""
        payload = self._payload_builder.build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=client_id,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
        )
        return self._payload_executor.execute_order_payload(symbol, payload, client_id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            True if cancellation succeeded.

        Raises:
            OrderCancellationError: If cancellation failed or order not found in response.
            BrokerageError: If API call failed (network, auth, etc).
        """
        try:
            response = self._client.cancel_orders(order_ids=[order_id])
            results = response.get("results", [])
            for res in results:
                if res.get("order_id") == order_id:
                    if res.get("success", False):
                        return True
                    # API explicitly reported failure
                    failure_reason = res.get("failure_reason", "unknown")
                    raise OrderCancellationError(
                        f"Cancellation rejected: {failure_reason}",
                        order_id=order_id,
                    )
            # Order not found in response
            raise OrderCancellationError(
                f"Order {order_id} not found in cancellation response",
                order_id=order_id,
            )
        except BrokerageError:
            # Re-raise brokerage errors (includes OrderCancellationError)
            raise
        except Exception as exc:
            logger.error(
                "Failed to cancel order",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="cancel_order",
                order_id=order_id,
            )
            raise OrderCancellationError(
                f"Unexpected error: {exc}",
                order_id=order_id,
            ) from exc

    def list_orders(
        self,
        product_id: str | None = None,
        status: list[str] | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """List orders with pagination.

        Args:
            product_id: Filter by product ID.
            status: Filter by order status.
            limit: Maximum orders per page.

        Returns:
            List of Order objects.

        Raises:
            OrderQueryError: If listing failed due to API or unexpected error.
        """
        orders: list[Order] = []
        cursor = None
        has_more = True

        while has_more:
            try:
                kwargs: dict[str, Any] = {"limit": limit}
                if product_id:
                    kwargs["product_id"] = product_id
                if status:
                    kwargs["order_status"] = status
                if cursor:
                    kwargs["cursor"] = cursor

                response = self._client.list_orders(**kwargs)

                page_orders = response.get("orders", [])
                for item in page_orders:
                    orders.append(to_order(item))

                cursor = response.get("cursor")
                if not cursor or not page_orders:
                    has_more = False
            except BrokerageError:
                # Re-raise brokerage errors
                raise
            except Exception as exc:
                logger.error(
                    "Failed to list orders from broker",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="list_orders",
                    product_id=product_id,
                )
                raise OrderQueryError(f"Failed to list orders: {exc}") from exc

        return orders

    def get_order(self, order_id: str) -> Order | None:
        """Get details of a single order.

        Args:
            order_id: The ID of the order to retrieve.

        Returns:
            Order object if found, None if order doesn't exist.

        Raises:
            OrderQueryError: If query failed due to API or unexpected error.
        """
        try:
            response = self._client.get_order_historical(order_id)
            order_data = response.get("order")
            if order_data:
                return to_order(order_data)
            return None
        except NotFoundError:
            # Order doesn't exist - this is expected, return None
            return None
        except BrokerageError:
            # Re-raise brokerage errors (auth, rate limit, etc)
            raise
        except Exception as exc:
            logger.error(
                "Failed to get order details",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_order",
                order_id=order_id,
            )
            raise OrderQueryError(f"Failed to get order {order_id}: {exc}") from exc

    def list_fills(
        self,
        product_id: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List fills with pagination.

        Args:
            product_id: Filter by product ID.
            order_id: Filter by order ID.
            limit: Maximum fills per page.

        Returns:
            List of fill dictionaries.

        Raises:
            OrderQueryError: If listing failed due to API or unexpected error.
        """
        fills: list[dict[str, Any]] = []
        cursor = None
        has_more = True

        while has_more:
            try:
                kwargs: dict[str, Any] = {"limit": limit}
                if product_id:
                    kwargs["product_id"] = product_id
                if order_id:
                    kwargs["order_id"] = order_id
                if cursor:
                    kwargs["cursor"] = cursor

                response = self._client.list_fills(**kwargs)

                page_fills = response.get("fills", [])
                fills.extend(page_fills)

                cursor = response.get("cursor")
                if not cursor or not page_fills:
                    has_more = False
            except BrokerageError:
                # Re-raise brokerage errors
                raise
            except Exception as exc:
                logger.error(
                    "Failed to list fills from broker",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="list_fills",
                    product_id=product_id,
                    order_id=order_id,
                )
                raise OrderQueryError(f"Failed to list fills: {exc}") from exc

        return fills

    def close_position(
        self,
        symbol: str,
        client_order_id: str | None = None,
        fallback: Callable[[], Order] | None = None,
    ) -> Order:
        """Close position for a symbol."""
        has_pos = False
        # Use injected position_provider instead of implicit self.list_positions()
        pos_list = self._position_provider.list_positions()
        for p in pos_list:
            if p.symbol == symbol and p.quantity > 0:
                has_pos = True
                break

        if not has_pos:
            raise ValidationError(f"No open position for {symbol}")

        try:
            payload = {"product_id": symbol}
            if client_order_id:
                payload["client_order_id"] = client_order_id

            response = self._client.close_position(payload)
            return to_order(response.get("order", {}))
        except Exception as e:
            if fallback:
                return fallback()
            raise e
