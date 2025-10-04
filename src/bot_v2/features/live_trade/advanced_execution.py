"""Advanced execution engine with support for rich order workflows.

Note: Models extracted to bot_v2.features.live_trade.advanced_execution_models for better organization.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, cast

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution_models.models import (
    NormalizedOrderRequest,
    OrderConfig,
    SizingMode,
    StopTrigger,
)
from bot_v2.features.live_trade.broker_adapter import BrokerAdapter
from bot_v2.features.live_trade.dynamic_sizing_helper import DynamicSizingHelper
from bot_v2.features.live_trade.order_metrics_reporter import OrderMetricsReporter
from bot_v2.features.live_trade.order_request_normalizer import OrderRequestNormalizer
from bot_v2.features.live_trade.order_validation_pipeline import OrderValidationPipeline
from bot_v2.features.live_trade.risk import (
    LiveRiskManager,
    PositionSizingAdvice,
)
from bot_v2.features.live_trade.stop_trigger_manager import StopTriggerManager

__all__ = [
    "SizingMode",
    "OrderConfig",
    "StopTrigger",
    "NormalizedOrderRequest",
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

        # Order tracking (shared with normalizer)
        self.pending_orders: dict[str, Order] = {}
        self.client_order_map: dict[str, str] = {}  # client_id -> order_id

        # Dedicated components
        self.broker_adapter = BrokerAdapter(broker=broker)
        self.normalizer = OrderRequestNormalizer(
            broker=broker,
            pending_orders=self.pending_orders,
            client_order_map=self.client_order_map,
            config=self.config,
        )
        self.stop_trigger_manager = StopTriggerManager(config=self.config)
        self.sizing_helper = DynamicSizingHelper(
            broker=broker,
            risk_manager=risk_manager,
            config=self.config,
        )
        self.validation_pipeline = OrderValidationPipeline(
            config=self.config,
            sizing_helper=self.sizing_helper,
            stop_trigger_manager=self.stop_trigger_manager,
            risk_manager=risk_manager,
        )

        # Metrics
        self.metrics_reporter = OrderMetricsReporter()

        logger.info(f"AdvancedExecutionEngine initialized with config: {self.config}")

    @property
    def order_metrics(self) -> dict[str, int]:
        """Get order metrics dict (backward compatibility)."""
        return self.metrics_reporter.get_metrics_dict()

    @property
    def rejections_by_reason(self) -> dict[str, int]:
        """Get rejections by reason dict (backward compatibility)."""
        return self.metrics_reporter.rejections_by_reason

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
        request = None  # Track for cleanup in except block
        try:
            # Normalize request
            request = self.normalizer.normalize(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                reduce_only=reduce_only,
                post_only=post_only,
                client_id=client_id,
                leverage=leverage,
            )

            # Handle duplicate orders
            if request is None:
                existing = self.normalizer.get_existing_order(client_id or "")
                if existing:
                    logger.warning(f"Duplicate client_id, returning existing order {existing.id}")
                return existing
            validation = self.validation_pipeline.validate(request)
            if validation.failed:
                reason = validation.rejection_reason or "unknown"
                self._record_rejection(
                    reason,
                    post_only=validation.post_only_rejection,
                )
                return None

            self.stop_trigger_manager.register_stop_trigger(
                order_type=request.order_type,
                client_id=request.client_id,
                symbol=request.symbol,
                stop_price=request.stop_price,
                side=request.side,
                order_quantity=request.quantity,
                limit_price=request.limit_price,
            )

            return self._submit_order_to_broker(
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                order_quantity=request.quantity,
                limit_price=request.limit_price,
                stop_price=request.stop_price,
                time_in_force=request.time_in_force,
                client_id=request.client_id,
                reduce_only=request.reduce_only,
                leverage=request.leverage,
            )

        except Exception as exc:
            # Handle errors from normalization or execution
            logger.error(
                "Failed to place order via AdvancedExecutionEngine: %s",
                exc,
                exc_info=True,
            )
            self._record_rejection("exception")
            # Clean up stop trigger if this was a stop order
            if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                # Use client_id from request if available, otherwise use parameter
                cleanup_client_id = request.client_id if request else client_id
                if cleanup_client_id:
                    self.stop_trigger_manager.unregister_stop_trigger(cleanup_client_id)
            return None

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
        # Submit order via BrokerAdapter
        order = self.broker_adapter.submit_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            client_id=client_id,
            reduce_only=reduce_only,
            leverage=leverage,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

        if order:
            self.pending_orders[order.id] = order
            self.client_order_map[client_id] = order.id
            self.metrics_reporter.record_placement(order)
            logger.info(f"Placed order {order.id}: {side.value} {order_quantity} {symbol}")

        return order

    def _record_rejection(self, reason: str, *, post_only: bool = False) -> None:
        """Record order rejection metrics and categorize by reason."""
        self.metrics_reporter.record_rejection(reason, post_only=post_only)

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
                    # Get order before deleting for metrics
                    cancelled_order = self.pending_orders[order_id]
                    self.metrics_reporter.record_cancellation(cancelled_order)
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
        # Normalize to Decimal
        if not isinstance(new_quantity, Decimal):
            new_quantity = Decimal(str(new_quantity))

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
            "orders": {**self.metrics_reporter.get_metrics_dict(), **stop_metrics},
            "pending_count": len(self.pending_orders),
            "stop_triggers": stop_metrics["stop_triggers"],
            "active_stops": stop_metrics["active_stops"],
        }

    def export_metrics(self, collector: Any, prefix: str = "execution") -> None:
        """Export metrics to MetricsCollector for telemetry.

        Args:
            collector: MetricsCollector instance
            prefix: Metric name prefix (default: "execution")

        This method is called periodically by the bot's telemetry system to
        surface order execution metrics in monitoring dashboards.
        """
        # Export order metrics
        self.metrics_reporter.export_to_collector(collector, prefix=f"{prefix}.orders")

        # Export pending order count as gauge
        collector.record_gauge(f"{prefix}.pending_orders", float(len(self.pending_orders)))

        # Export stop trigger metrics
        stop_metrics = self.stop_trigger_manager.get_metrics()
        collector.record_gauge(
            f"{prefix}.stop_triggers", float(stop_metrics.get("stop_triggers", 0))
        )
        collector.record_gauge(f"{prefix}.active_stops", float(stop_metrics.get("active_stops", 0)))
