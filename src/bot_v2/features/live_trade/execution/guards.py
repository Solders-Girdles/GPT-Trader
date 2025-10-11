"""Order guard helpers (market data, post-only, spec validation)."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, cast

from bot_v2.errors import ExecutionError
from bot_v2.features.brokerages.coinbase.specs import validate_order as spec_validate_order
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Product, Quote
from bot_v2.utilities.quantization import quantize_price_side_aware

logger = logging.getLogger(__name__)


class OrderGuards:
    """Encapsulates market fetch, post-only, and spec validation routines."""

    def __init__(
        self,
        *,
        broker: Any,
        config: Any,
        order_metrics: dict[str, int],
        rejections_by_reason: dict[str, int],
    ) -> None:
        self._broker = broker
        self._config = config
        self._order_metrics = order_metrics
        self._rejections_by_reason = rejections_by_reason

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------
    def fetch_market_data(
        self, *, symbol: str, order_type: OrderType, post_only: bool
    ) -> tuple[Product | None, Quote | None]:
        """Fetch product/quote information needed for validation."""
        broker = self._broker
        try:
            product = cast(Product | None, broker.get_product(symbol))
        except Exception:
            product = None

        quote: Quote | None = None
        if (
            order_type == OrderType.LIMIT
            and post_only
            and getattr(self._config, "reject_on_cross", False)
        ):
            try:
                quote = cast(Quote | None, broker.get_quote(symbol))
            except Exception as exc:
                logger.error(
                    "Failed to fetch quote for post-only validation on %s: %s", symbol, exc
                )
                raise ExecutionError(
                    f"Could not get quote for post-only validation on {symbol}"
                ) from exc

            if quote is None:
                raise ExecutionError(f"Could not get quote for post-only validation on {symbol}")

        return product, quote

    # ------------------------------------------------------------------
    # Post-only guard
    # ------------------------------------------------------------------
    def validate_post_only(
        self,
        *,
        symbol: str,
        order_type: OrderType,
        post_only: bool,
        side: OrderSide,
        limit_price: Decimal | None,
        quote: Quote | None,
    ) -> bool:
        """Ensure post-only orders do not cross the book."""
        if not (
            order_type == OrderType.LIMIT
            and post_only
            and getattr(self._config, "reject_on_cross", False)
        ):
            return True

        if quote is None or limit_price is None:
            return True

        if side == OrderSide.BUY and limit_price >= quote.ask:
            logger.warning(f"Post-only buy would cross at {limit_price:.2f} >= {quote.ask:.2f}")
            self._order_metrics["post_only_rejected"] += 1
            return False

        if side == OrderSide.SELL and limit_price <= quote.bid:
            logger.warning(f"Post-only sell would cross at {limit_price:.2f} <= {quote.bid:.2f}")
            self._order_metrics["post_only_rejected"] += 1
            return False

        return True

    # ------------------------------------------------------------------
    # Spec validation & quantization
    # ------------------------------------------------------------------
    def apply_quantization_and_specs(
        self,
        *,
        symbol: str,
        product: Product | None,
        order_type: OrderType,
        side: OrderSide,
        limit_price: Decimal | None,
        stop_price: Decimal | None,
        order_quantity: Decimal,
    ) -> tuple[Decimal | None, Decimal | None, Decimal] | None:
        """Quantize prices and validate against product specs."""
        if product is not None and product.price_increment:
            increment = product.price_increment
            if order_type == OrderType.LIMIT and limit_price is not None:
                limit_price = quantize_price_side_aware(
                    Decimal(str(limit_price)), increment, side.value
                )
            if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is not None:
                stop_price = quantize_price_side_aware(
                    Decimal(str(stop_price)), increment, side.value
                )
            if order_type == OrderType.STOP_LIMIT and limit_price is not None:
                limit_price = quantize_price_side_aware(
                    Decimal(str(limit_price)), increment, side.value
                )

        if product is None:
            return limit_price, stop_price, order_quantity

        validation = spec_validate_order(
            product=product,
            side=side.value,
            quantity=Decimal(str(order_quantity)),
            order_type=order_type.value.lower(),
            price=(
                Decimal(str(limit_price))
                if (order_type == OrderType.LIMIT and limit_price is not None)
                else None
            ),
        )

        if not validation.ok:
            reason = validation.reason or "spec_violation"
            self._order_metrics["rejected"] += 1
            self._rejections_by_reason[reason] = self._rejections_by_reason.get(reason, 0) + 1
            logger.warning(f"Spec validation failed for {symbol}: {reason}")
            return None

        if validation.adjusted_quantity is not None:
            order_quantity = (
                validation.adjusted_quantity
                if isinstance(validation.adjusted_quantity, Decimal)
                else Decimal(str(validation.adjusted_quantity))
            )

        if validation.adjusted_price is not None and order_type == OrderType.LIMIT:
            limit_price = validation.adjusted_price

        return limit_price, stop_price, order_quantity
