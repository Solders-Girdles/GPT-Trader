"""Advanced execution engine with support for rich order workflows.

Note: Models extracted to bot_v2.features.live_trade.advanced_execution_models for better organization.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from decimal import Decimal
from typing import Any, cast

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.coinbase.specs import (
    validate_order as spec_validate_order,
)
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution_models.models import (
    OrderConfig,
    SizingMode,
    StopTrigger,
)
from bot_v2.features.live_trade.dynamic_sizing_helper import DynamicSizingHelper
from bot_v2.features.live_trade.risk import (
    LiveRiskManager,
    PositionSizingAdvice,
)
from bot_v2.features.live_trade.stop_trigger_manager import StopTriggerManager
from bot_v2.utilities.quantization import quantize_price_side_aware

__all__ = [
    "SizingMode",
    "OrderConfig",
    "StopTrigger",
    "AdvancedExecutionEngine",
]

logger = logging.getLogger(__name__)


class AdvancedExecutionEngine:
    """
    Enhanced execution engine with Week 3 features.

    Manages advanced order types, TIF mapping, and impact-aware sizing.
    """

    # TIF mapping for Coinbase Advanced Trade
    TIF_MAPPING = {
        TimeInForce.GTC: "GOOD_TILL_CANCELLED",
        TimeInForce.IOC: "IMMEDIATE_OR_CANCEL",
        TimeInForce.FOK: None,  # Gated - not supported yet
    }

    def __init__(
        self,
        broker: Any,
        risk_manager: LiveRiskManager | None = None,
        config: OrderConfig | None = None,
    ) -> None:
        """
        Initialize enhanced execution engine.

        Args:
            broker: Broker adapter instance
            risk_manager: Risk manager for validation
            config: Order configuration
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.config = config or OrderConfig()

        # Dedicated components
        self.stop_trigger_manager = StopTriggerManager(config=self.config)
        self.sizing_helper = DynamicSizingHelper(
            broker=broker,
            risk_manager=risk_manager,
            config=self.config,
        )

        # Order tracking
        self.pending_orders: dict[str, Order] = {}
        self.client_order_map: dict[str, str] = {}  # client_id -> order_id

        # Metrics
        self.order_metrics = {
            "placed": 0,
            "filled": 0,
            "cancelled": 0,
            "rejected": 0,
            "post_only_rejected": 0,
        }

        # Track rejection reasons
        self.rejections_by_reason: dict[str, int] = {}

        logger.info(f"AdvancedExecutionEngine initialized with config: {self.config}")

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        post_only: bool = False,
        client_id: str | None = None,
        leverage: int | None = None,
    ) -> Order | None:
        """
        Place an order with advanced features, adhering to IBrokerage.
        """
        client_id = self._prepare_order_request(client_id, symbol, side)

        duplicate_order = self._check_duplicate_order(client_id)
        if duplicate_order:
            return duplicate_order

        try:
            order_quantity = self._normalize_quantity(quantity)
            product, quote = self._fetch_market_data(
                symbol=symbol,
                order_type=order_type,
                post_only=post_only,
            )

            sizing_advice = self.sizing_helper.maybe_apply_position_sizing(
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                limit_price=limit_price,
                product=product,
                quote=quote,
                leverage=leverage,
            )

            if sizing_advice is not None:
                order_quantity = sizing_advice.target_quantity
                if order_quantity <= 0:
                    logger.info(
                        "Dynamic sizing prevented new order for %s (%s)",
                        symbol,
                        sizing_advice.reason or "no reason provided",
                    )
                    self.order_metrics["rejected"] += 1
                    self.rejections_by_reason["position_sizing"] = (
                        self.rejections_by_reason.get("position_sizing", 0) + 1
                    )
                    return None
                if sizing_advice.reason:
                    logger.debug(
                        "Dynamic sizing adjusted %s quantity to %s (%s)",
                        symbol,
                        order_quantity,
                        sizing_advice.reason,
                    )
                reduce_only = reduce_only or sizing_advice.reduce_only

            if not self._validate_post_only_constraints(
                symbol=symbol,
                order_type=order_type,
                post_only=post_only,
                side=side,
                limit_price=limit_price,
                quote=quote,
            ):
                return None

            adjustment = self._apply_quantization_and_specs(
                symbol=symbol,
                product=product,
                order_type=order_type,
                side=side,
                limit_price=limit_price,
                stop_price=stop_price,
                order_quantity=order_quantity,
            )
            if adjustment is None:
                return None
            limit_price, stop_price, order_quantity = adjustment

            if not self._run_risk_validation(
                symbol=symbol,
                side=side,
                order_quantity=order_quantity,
                limit_price=limit_price,
                order_type=order_type,
                product=product,
                quote=quote,
            ):
                return None

            is_valid, rejection_reason = self.stop_trigger_manager.validate_stop_order_requirements(
                symbol=symbol,
                order_type=order_type,
                stop_price=stop_price,
            )
            if not is_valid:
                self.order_metrics["rejected"] += 1
                if rejection_reason:
                    self.rejections_by_reason[rejection_reason] = (
                        self.rejections_by_reason.get(rejection_reason, 0) + 1
                    )
                return None

            self.stop_trigger_manager.register_stop_trigger(
                order_type=order_type,
                client_id=client_id,
                symbol=symbol,
                stop_price=stop_price,
                side=side,
                order_quantity=order_quantity,
                limit_price=limit_price,
            )

            return self._submit_order_to_broker(
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_id=client_id,
                reduce_only=reduce_only,
                leverage=leverage,
            )

        except Exception as exc:
            return self._handle_order_error(
                exc=exc,
                order_type=order_type,
                client_id=client_id,
            )

    def _prepare_order_request(self, client_id: str | None, symbol: str, side: OrderSide) -> str:
        return (
            client_id or f"{symbol}_{side.value}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        )

    def _check_duplicate_order(self, client_id: str) -> Order | None:
        if client_id in self.client_order_map:
            logger.warning(f"Duplicate client_id {client_id}, returning existing order")
            existing_id = self.client_order_map[client_id]
            return self.pending_orders.get(existing_id)
        return None

    def _normalize_quantity(self, quantity: Decimal | int) -> Decimal:
        return quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))

    def _run_risk_validation(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_quantity: Decimal,
        limit_price: Decimal | None,
        order_type: OrderType,
        product: Product | None,
        quote: Quote | None,
    ) -> bool:
        if self.risk_manager is None:
            return True
        try:
            validation_price = self.sizing_helper.determine_reference_price(
                symbol=symbol,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                quote=quote,
                product=product,
            )
            equity = self.sizing_helper.estimate_equity()
            current_positions = getattr(self.risk_manager, "positions", {})
            self.risk_manager.pre_trade_validate(
                symbol=symbol,
                side=side.value,
                quantity=order_quantity,
                price=validation_price,
                product=product,
                equity=equity,
                current_positions=current_positions,
            )
            return True
        except ValidationError as exc:
            logger.warning(f"Risk validation failed for {symbol}: {exc}")
            self.order_metrics["rejected"] += 1
            self.rejections_by_reason["risk"] = self.rejections_by_reason.get("risk", 0) + 1
            return False

    def _fetch_market_data(
        self, *, symbol: str, order_type: OrderType, post_only: bool
    ) -> tuple[Product | None, Quote | None]:
        try:
            product = cast(Product | None, self.broker.get_product(symbol))
        except Exception:
            product = None

        quote: Quote | None = None
        if order_type == OrderType.LIMIT and post_only and self.config.reject_on_cross:
            try:
                quote = cast(Quote | None, self.broker.get_quote(symbol))
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

    def _validate_post_only_constraints(
        self,
        *,
        symbol: str,
        order_type: OrderType,
        post_only: bool,
        side: OrderSide,
        limit_price: Decimal | None,
        quote: Quote | None,
    ) -> bool:
        if not (order_type == OrderType.LIMIT and post_only and self.config.reject_on_cross):
            return True

        if quote is None or limit_price is None:
            return True

        if side == OrderSide.BUY and limit_price >= quote.ask:
            logger.warning(f"Post-only buy would cross at {limit_price:.2f} >= {quote.ask:.2f}")
            self.order_metrics["post_only_rejected"] += 1
            return False

        if side == OrderSide.SELL and limit_price <= quote.bid:
            logger.warning(f"Post-only sell would cross at {limit_price:.2f} <= {quote.bid:.2f}")
            self.order_metrics["post_only_rejected"] += 1
            return False

        return True

    def _apply_quantization_and_specs(
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
            self.order_metrics["rejected"] += 1
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
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

    def _submit_order_to_broker(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        limit_price: Decimal | None,
        stop_price: Decimal | None,
        time_in_force: TimeInForce,
        client_id: str,
        reduce_only: bool,
        leverage: int | None,
    ) -> Order | None:
        broker_place = getattr(self.broker, "place_order")
        params = inspect.signature(broker_place).parameters

        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": order_quantity,
            "client_id": client_id,
            "reduce_only": reduce_only,
            "leverage": leverage,
        }

        if "limit_price" in params:
            kwargs["limit_price"] = limit_price
        elif "price" in params:
            kwargs["price"] = limit_price

        if "stop_price" in params:
            kwargs["stop_price"] = stop_price

        if isinstance(time_in_force, TimeInForce):
            tif_value_enum = time_in_force
            tif_value_str = time_in_force.value
        else:
            try:
                tif_value_enum = TimeInForce[str(time_in_force).upper()]
            except Exception:
                tif_value_enum = TimeInForce.GTC
            tif_value_str = tif_value_enum.value

        if "time_in_force" in params:
            kwargs["time_in_force"] = tif_value_str
        if "tif" in params:
            kwargs["tif"] = tif_value_enum

        order = cast(Order | None, broker_place(**kwargs))

        if order:
            self.pending_orders[order.id] = order
            self.client_order_map[client_id] = order.id
            self.order_metrics["placed"] += 1
            logger.info(f"Placed order {order.id}: {side.value} {order_quantity} {symbol}")

        return order

    def _handle_order_error(
        self,
        *,
        exc: Exception,
        order_type: OrderType,
        client_id: str,
    ) -> Order | None:
        logger.error(
            "Failed to place order via AdvancedExecutionEngine: %s",
            exc,
            exc_info=True,
        )
        self.order_metrics["rejected"] += 1
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            self.stop_trigger_manager.unregister_stop_trigger(client_id)
        return None

    def cancel_and_replace(
        self,
        order_id: str,
        new_price: Decimal | None = None,
        new_size: Decimal | None = None,
        max_retries: int = 3,
    ) -> Order | None:
        """
        Cancel and replace order atomically with retry logic.

        Args:
            order_id: Original order ID
            new_price: New limit/stop price
            new_size: New order size
            max_retries: Maximum retry attempts

        Returns:
            New order or None if failed
        """
        # Get original order
        original = self.pending_orders.get(order_id)
        if not original:
            logger.error(f"Order {order_id} not found for cancel/replace")
            return None

        # Generate new client ID for replacement
        replace_client_id = f"{original.client_id}_replace_{int(time.time() * 1000)}"

        # Attempt cancel with retries
        for attempt in range(max_retries):
            try:
                if bool(self.broker.cancel_order(order_id)):
                    self.order_metrics["cancelled"] += 1
                    del self.pending_orders[order_id]
                    break
            except Exception as e:
                logger.warning(f"Cancel attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.5 * (2**attempt))  # Exponential backoff

        # Place replacement order
        original_quantity = original.quantity
        replacement_side = OrderSide.SELL if original.side == OrderSide.BUY else OrderSide.BUY
        replacement_type = original.type
        replacement_tif = original.tif

        new_quantity = new_size if new_size is not None else original_quantity
        new_quantity = self._normalize_quantity(new_quantity)

        new_price_decimal = Decimal(str(new_price)) if new_price is not None else None

        replacement_limit = (
            new_price_decimal
            if replacement_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
            else original.price
        )
        replacement_stop = (
            new_price_decimal
            if replacement_type in (OrderType.STOP, OrderType.STOP_LIMIT)
            else original.stop_price
        )

        return self.place_order(
            symbol=original.symbol,
            side=replacement_side,
            quantity=new_quantity,
            order_type=replacement_type,
            limit_price=replacement_limit,
            stop_price=replacement_stop,
            time_in_force=replacement_tif,
            reduce_only=False,
            client_id=replace_client_id,
        )

    def calculate_impact_aware_size(
        self,
        symbol: str | None,
        target_notional: Decimal,
        market_snapshot: dict[str, Any],
        max_impact_bps: Decimal | None = None,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate position size that respects slippage constraints.

        Args:
            symbol: Trading symbol
            target_notional: Target position size in USD
            market_snapshot: Market depth and liquidity data
            max_impact_bps: Maximum acceptable impact (overrides config)

        Returns:
            (adjusted_notional, expected_impact_bps)
        """
        return self.sizing_helper.calculate_impact_aware_size(
            symbol=symbol,
            target_notional=target_notional,
            market_snapshot=market_snapshot,
            max_impact_bps=max_impact_bps,
        )

    def close_position(self, symbol: str, reduce_only: bool = True) -> Order | None:
        """
        Helper to close position with reduce-only market order.

        Args:
            symbol: Symbol to close
            reduce_only: Whether to use reduce-only flag

        Returns:
            Close order or None
        """
        # Get current position
        positions = cast(list[Position], self.broker.get_positions())
        position = next((p for p in positions if p.symbol == symbol), None)

        if position is None or position.quantity == 0:
            logger.warning(f"No position to close for {symbol}")
            return None

        # Determine side (opposite of position)
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        quantity = abs(position.quantity)

        logger.info(f"Closing position: {side} {quantity} {symbol}")

        return self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reduce_only=reduce_only,
            client_id=f"close_{symbol}_{int(time.time() * 1000)}",
        )

    def check_stop_triggers(self, current_prices: dict[str, Decimal]) -> list[str]:
        """
        Check if any stop orders should trigger.

        Args:
            current_prices: Current mark prices by symbol

        Returns:
            List of triggered order IDs
        """
        return self.stop_trigger_manager.check_stop_triggers(current_prices)

    @property
    def stop_triggers(self) -> dict[str, StopTrigger]:
        """Access to stop triggers for backward compatibility."""
        return self.stop_trigger_manager.stop_triggers

    @property
    def _last_sizing_advice(self) -> PositionSizingAdvice | None:
        """Access to last sizing advice for diagnostics."""
        return self.sizing_helper.last_sizing_advice

    def _validate_tif(self, tif: str) -> TimeInForce | None:
        """Validate and convert TIF string to enum."""

        tif_upper = tif.upper()

        if tif_upper == "GTC":
            return TimeInForce.GTC
        elif tif_upper == "IOC" and self.config.enable_ioc:
            return TimeInForce.IOC
        elif tif_upper == "FOK" and self.config.enable_fok:
            logger.warning("FOK order type is gated and not yet supported")
            return None
        else:
            logger.error(f"Unsupported or disabled TIF: {tif}")
            return None

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics."""
        stop_metrics = self.stop_trigger_manager.get_metrics()
        return {
            "orders": {**self.order_metrics.copy(), **stop_metrics},
            "pending_count": len(self.pending_orders),
            "stop_triggers": stop_metrics["stop_triggers"],
            "active_stops": stop_metrics["active_stops"],
        }
