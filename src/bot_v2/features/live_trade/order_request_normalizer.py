"""Order request normalization for advanced execution engine.

This module handles the first phase of order placement: normalizing and preparing
order requests before validation and execution.
"""

from __future__ import annotations

import logging
import time
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from bot_v2.errors import ExecutionError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution_models.models import (
    NormalizedOrderRequest,
    OrderConfig,
)

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.features.brokerages.core.interfaces import IBrokerage

logger = logging.getLogger(__name__)


class OrderRequestNormalizer:
    """Normalizes order requests before validation and execution.

    Responsibilities:
    - Generate/validate client_id
    - Check for duplicate orders
    - Normalize quantity to Decimal
    - Fetch market data (product, quote)

    This class performs the initial preparation of order requests, ensuring
    all parameters are in the correct format and required market data is available.
    """

    def __init__(
        self,
        broker: IBrokerage,
        pending_orders: dict[str, Order],
        client_order_map: dict[str, str],
        config: OrderConfig,
    ) -> None:
        """Initialize order request normalizer.

        Args:
            broker: Broker adapter for fetching market data
            pending_orders: Map of order_id -> Order (shared with engine)
            client_order_map: Map of client_id -> order_id (shared with engine)
            config: Order configuration
        """
        self.broker = broker
        self.pending_orders = pending_orders
        self.client_order_map = client_order_map
        self.config = config

    def normalize(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal | int,
        order_type: OrderType,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        post_only: bool = False,
        client_id: str | None = None,
        leverage: int | None = None,
    ) -> NormalizedOrderRequest | None:
        """Normalize order request parameters.

        This method performs the following steps:
        1. Prepare/validate client_id
        2. Check for duplicate orders
        3. Normalize quantity to Decimal
        4. Fetch market data (product, quote if needed)

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity (will be normalized to Decimal)
            order_type: Order type
            limit_price: Limit price (for LIMIT/STOP_LIMIT orders)
            stop_price: Stop price (for STOP/STOP_LIMIT orders)
            time_in_force: Time-in-force policy
            reduce_only: Whether order can only reduce position
            post_only: Whether order should be post-only
            client_id: Optional client order ID
            leverage: Optional leverage multiplier

        Returns:
            NormalizedOrderRequest if successful, None if duplicate order found

        Raises:
            ExecutionError: If market data fetch fails for post-only orders
        """
        # 1. Prepare client_id
        prepared_client_id = self._prepare_client_id(client_id, symbol, side)

        # 2. Check duplicates
        if self._is_duplicate(prepared_client_id):
            logger.warning(f"Duplicate client_id {prepared_client_id}, skipping normalization")
            return None

        # 3. Normalize quantity
        normalized_quantity = self._normalize_quantity(quantity)

        # 4. Fetch market data
        product, quote = self._fetch_market_data(symbol, order_type, post_only)

        return NormalizedOrderRequest(
            client_id=prepared_client_id,
            symbol=symbol,
            side=side,
            quantity=normalized_quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            post_only=post_only,
            leverage=leverage,
            product=product,
            quote=quote,
        )

    def get_existing_order(self, client_id: str) -> Order | None:
        """Get existing order by client_id (for duplicate handling).

        Args:
            client_id: Client order identifier

        Returns:
            Existing order if found, None otherwise
        """
        if client_id in self.client_order_map:
            existing_id = self.client_order_map[client_id]
            return self.pending_orders.get(existing_id)
        return None

    def _prepare_client_id(self, client_id: str | None, symbol: str, side: OrderSide) -> str:
        """Prepare client order ID (generate if not provided).

        Args:
            client_id: Optional client-provided ID
            symbol: Trading symbol
            side: Order side

        Returns:
            Valid client order ID
        """
        if client_id:
            return client_id

        # Generate unique client_id
        timestamp_ms = int(time.time() * 1000)
        random_suffix = uuid.uuid4().hex[:8]
        return f"{symbol}_{side.value}_{timestamp_ms}_{random_suffix}"

    def _is_duplicate(self, client_id: str) -> bool:
        """Check if order with this client_id already exists.

        Args:
            client_id: Client order identifier

        Returns:
            True if duplicate, False otherwise
        """
        return client_id in self.client_order_map

    def _normalize_quantity(self, quantity: Decimal | int) -> Decimal:
        """Normalize quantity to Decimal type.

        Args:
            quantity: Order quantity (int or Decimal)

        Returns:
            Quantity as Decimal
        """
        if isinstance(quantity, Decimal):
            return quantity
        return Decimal(str(quantity))

    def _fetch_market_data(
        self, symbol: str, order_type: OrderType, post_only: bool
    ) -> tuple[Product | None, Quote | None]:
        """Fetch market data (product and optionally quote).

        Product is always fetched (or None if unavailable).
        Quote is only fetched for post-only LIMIT orders when reject_on_cross is enabled.

        Args:
            symbol: Trading symbol
            order_type: Order type
            post_only: Whether order is post-only

        Returns:
            Tuple of (product, quote) - either may be None

        Raises:
            ExecutionError: If quote fetch fails for post-only orders
        """
        # Always try to fetch product
        product: Product | None = None
        try:
            product = cast(Product | None, self.broker.get_product(symbol))
        except Exception as exc:
            logger.debug(f"Could not fetch product for {symbol}: {exc}")
            product = None

        # Fetch quote only for post-only LIMIT orders when needed
        quote: Quote | None = None
        if order_type == OrderType.LIMIT and post_only and self.config.reject_on_cross:
            try:
                quote = cast(Quote | None, self.broker.get_quote(symbol))
            except Exception as exc:
                logger.error(f"Failed to fetch quote for post-only validation on {symbol}: {exc}")
                raise ExecutionError(
                    f"Could not get quote for post-only validation on {symbol}"
                ) from exc

            if quote is None:
                raise ExecutionError(f"Could not get quote for post-only validation on {symbol}")

        return product, quote
