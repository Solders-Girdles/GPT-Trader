"""Advanced execution engine with support for rich order workflows."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ...errors import ExecutionError, ValidationError
from ..brokerages.coinbase.specs import (
    validate_order as spec_validate_order,
)
from ..brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)
from ..utils.quantization import quantize_price_side_aware
from .risk import LiveRiskManager

__all__ = [
    "SizingMode",
    "OrderConfig",
    "StopTrigger",
    "AdvancedExecutionEngine",
]

logger = logging.getLogger(__name__)


class SizingMode(Enum):
    """Position sizing strategy."""

    CONSERVATIVE = "conservative"  # Downsize to fit impact limit
    STRICT = "strict"  # Reject if can't fit
    AGGRESSIVE = "aggressive"  # Allow higher impact


@dataclass
class OrderConfig:
    """Configuration for advanced order types."""

    # Limit orders
    enable_limit_orders: bool = True
    limit_price_offset_bps: Decimal = Decimal("5")  # Offset from mid

    # Stop orders
    enable_stop_orders: bool = True
    stop_trigger_offset_pct: Decimal = Decimal("0.02")  # 2% from entry

    # Stop-limit orders
    enable_stop_limit: bool = True
    stop_limit_spread_bps: Decimal = Decimal("10")

    # Post-only protection
    enable_post_only: bool = True
    reject_on_cross: bool = True  # Reject if post-only would cross

    # TIF support
    enable_ioc: bool = True
    enable_fok: bool = False  # Gate until confirmed

    # Sizing
    sizing_mode: SizingMode = SizingMode.CONSERVATIVE
    max_impact_bps: Decimal = Decimal("15")


@dataclass
class StopTrigger:
    """Stop order trigger tracking."""

    order_id: str
    symbol: str
    trigger_price: Decimal
    side: OrderSide
    quantity: Decimal
    limit_price: Decimal | None = None
    created_at: datetime = field(default_factory=datetime.now)
    triggered: bool = False
    triggered_at: datetime | None = None


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
        broker,
        risk_manager: LiveRiskManager | None = None,
        config: OrderConfig | None = None,
    ):
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

        # Order tracking
        self.pending_orders: dict[str, Order] = {}
        self.stop_triggers: dict[str, StopTrigger] = {}
        self.client_order_map: dict[str, str] = {}  # client_id -> order_id

        # Metrics
        self.order_metrics = {
            "placed": 0,
            "filled": 0,
            "cancelled": 0,
            "rejected": 0,
            "post_only_rejected": 0,
            "stop_triggered": 0,
        }

        # Track rejection reasons
        self.rejections_by_reason: dict[str, int] = {}

        # Optional per-symbol slippage multipliers for impact-aware sizing
        self.slippage_multipliers: dict[str, Decimal] = {}
        try:
            import os

            env_val = os.getenv("SLIPPAGE_MULTIPLIERS", "")
            if env_val:
                for pair in env_val.split(","):
                    if ":" in pair:
                        sym, mult = pair.split(":", 1)
                        self.slippage_multipliers[sym.strip()] = Decimal(str(mult.strip()))
        except Exception:
            pass

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
        # Generate idempotent client ID if not provided
        client_id = (
            client_id or f"{symbol}_{side.value}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        )

        if client_id in self.client_order_map:
            logger.warning(f"Duplicate client_id {client_id}, returning existing order")
            return self.pending_orders.get(self.client_order_map[client_id])

        try:
            order_quantity = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))

            # Pre-trade risk validation should run before spec validation
            if self.risk_manager is not None:
                try:
                    self.risk_manager.pre_trade_validate(
                        symbol=symbol,
                        side=side,
                        quantity=order_quantity,
                        order_type=order_type,
                        price=limit_price,
                    )
                except ValidationError as e:
                    logger.warning(f"Risk validation failed for {symbol}: {e}")
                    self.order_metrics["rejected"] += 1
                    self.rejections_by_reason["risk"] = self.rejections_by_reason.get("risk", 0) + 1
                    return None

            # Fetch product for spec validation and quantization
            product: Product | None = None
            try:
                product = self.broker.get_product(symbol)
            except Exception:
                product = None

            quote: Quote | None = None

            # Post-only validation for limit orders
            if order_type == OrderType.LIMIT and post_only and self.config.reject_on_cross:
                try:
                    quote = self.broker.get_quote(symbol)
                except Exception as exc:
                    logger.error(
                        "Failed to fetch quote for post-only validation on %s: %s", symbol, exc
                    )
                    raise ExecutionError(
                        f"Could not get quote for post-only validation on {symbol}"
                    ) from exc

                if not quote:
                    raise ExecutionError(
                        f"Could not get quote for post-only validation on {symbol}"
                    )

                if side == OrderSide.BUY and limit_price and limit_price >= quote.ask:
                    logger.warning(
                        f"Post-only buy would cross at {limit_price:.2f} >= {quote.ask:.2f}"
                    )
                    self.order_metrics["post_only_rejected"] += 1
                    return None
                if side == OrderSide.SELL and limit_price and limit_price <= quote.bid:
                    logger.warning(
                        f"Post-only sell would cross at {limit_price:.2f} <= {quote.bid:.2f}"
                    )
                    self.order_metrics["post_only_rejected"] += 1
                    return None

            # Side-aware quantization for explicit limit/stop prices
            if product is not None and product.price_increment:
                if order_type == OrderType.LIMIT and limit_price is not None:
                    limit_price = quantize_price_side_aware(
                        Decimal(str(limit_price)), product.price_increment, side.value
                    )
                if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is not None:
                    stop_price = quantize_price_side_aware(
                        Decimal(str(stop_price)), product.price_increment, side.value
                    )
                if order_type == OrderType.STOP_LIMIT and limit_price is not None:
                    limit_price = quantize_price_side_aware(
                        Decimal(str(limit_price)), product.price_increment, side.value
                    )

            # Stop orders require a trigger price
            if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                if not self.config.enable_stop_orders:
                    logger.warning("Stop orders disabled by configuration; rejecting %s", symbol)
                    self.order_metrics["rejected"] += 1
                    self.rejections_by_reason["stop_disabled"] = (
                        self.rejections_by_reason.get("stop_disabled", 0) + 1
                    )
                    return None
                if stop_price is None:
                    logger.warning("Stop order for %s missing stop price", symbol)
                    self.order_metrics["rejected"] += 1
                    self.rejections_by_reason["invalid_stop"] = (
                        self.rejections_by_reason.get("invalid_stop", 0) + 1
                    )
                    return None

            # Exchange spec pre-flight validation (adopt adjusted values)
            if product is not None:
                vr = spec_validate_order(
                    product=product,
                    side=side.value,
                    qty=Decimal(str(order_quantity)),
                    order_type=order_type.value.lower(),
                    price=(
                        Decimal(str(limit_price))
                        if (order_type == OrderType.LIMIT and limit_price is not None)
                        else None
                    ),
                )
                if not vr.ok:
                    reason = vr.reason or "spec_violation"
                    self.order_metrics["rejected"] += 1
                    self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
                    logger.warning(f"Spec validation failed for {symbol}: {reason}")
                    return None
                if vr.adjusted_qty is not None:
                    order_quantity = (
                        vr.adjusted_qty
                        if isinstance(vr.adjusted_qty, Decimal)
                        else Decimal(str(vr.adjusted_qty))
                    )
                if vr.adjusted_price is not None and order_type == OrderType.LIMIT:
                    limit_price = vr.adjusted_price

            # Register stop trigger bookkeeping prior to placement so we can track even if broker call fails
            if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is not None:
                trigger = StopTrigger(
                    order_id=client_id,
                    symbol=symbol,
                    trigger_price=Decimal(str(stop_price)),
                    side=side,
                    quantity=order_quantity,
                    limit_price=(
                        Decimal(str(limit_price))
                        if order_type == OrderType.STOP_LIMIT and limit_price is not None
                        else None
                    ),
                )
                self.stop_triggers[client_id] = trigger

            # Directly call the broker's place_order method which follows IBrokerage
            order = self.broker.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=order_quantity,
                price=limit_price,
                stop_price=stop_price,
                tif=time_in_force,
                client_id=client_id,
                reduce_only=reduce_only,
                leverage=leverage,
            )

            if order:
                self.pending_orders[order.id] = order
                self.client_order_map[client_id] = order.id
                self.order_metrics["placed"] += 1
                logger.info(f"Placed order {order.id}: {side.value} {order_quantity} {symbol}")

            return order

        except Exception as e:
            logger.error(f"Failed to place order via AdvancedExecutionEngine: {e}", exc_info=True)
            self.order_metrics["rejected"] += 1
            if (
                order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
                and client_id in self.stop_triggers
            ):
                # Remove trigger when placement fails
                self.stop_triggers.pop(client_id, None)
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
                if self.broker.cancel_order(order_id):
                    self.order_metrics["cancelled"] += 1
                    del self.pending_orders[order_id]
                    break
            except Exception as e:
                logger.warning(f"Cancel attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.5 * (2**attempt))  # Exponential backoff

        # Place replacement order
        original_quantity = getattr(original, "quantity", original.quantity)
        return self.place_order(
            symbol=original.symbol,
            side="buy" if original.side == OrderSide.BUY else "sell",
            quantity=new_size or original_quantity,
            order_type="limit" if original.type == OrderType.LIMIT else "market",
            limit_price=new_price if original.type == OrderType.LIMIT else None,
            stop_price=(
                new_price if original.type in [OrderType.STOP, OrderType.STOP_LIMIT] else None
            ),
            time_in_force="GTC",
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
            target_notional: Target position size in USD
            market_snapshot: Market depth and liquidity data
            max_impact_bps: Maximum acceptable impact (overrides config)

        Returns:
            (adjusted_notional, expected_impact_bps)
        """
        max_impact = max_impact_bps or self.config.max_impact_bps

        l1_depth = Decimal(str(market_snapshot.get("depth_l1", 0)))
        l10_depth = Decimal(str(market_snapshot.get("depth_l10", 0)))

        if not l1_depth or not l10_depth:
            logger.warning("Insufficient depth data for impact calculation")
            return Decimal("0"), Decimal("0")

        # Binary search for max size within impact limit
        low, high = Decimal("0"), min(target_notional, l10_depth)
        best_size = Decimal("0")
        best_impact = Decimal("0")
        # Add per-symbol slippage multiplier as extra expected impact (bps)
        extra_bps = Decimal("0")
        if symbol and symbol in self.slippage_multipliers:
            try:
                extra_bps = Decimal("10000") * Decimal(str(self.slippage_multipliers[symbol]))
            except Exception:
                extra_bps = Decimal("0")

        while high - low > Decimal("1"):  # $1 precision
            mid = (low + high) / 2
            impact = self._estimate_impact(mid, l1_depth, l10_depth) + extra_bps

            if impact <= max_impact:
                best_size = mid
                best_impact = impact
                low = mid
            else:
                high = mid

        # Apply sizing mode
        if self.config.sizing_mode == SizingMode.STRICT and best_size < target_notional:
            logger.warning(
                f"Strict mode: Cannot fit {target_notional} within {max_impact} bps impact"
            )
            return Decimal("0"), Decimal("0")
        elif self.config.sizing_mode == SizingMode.AGGRESSIVE:
            # Allow up to 2x the impact limit in aggressive mode
            if target_notional <= l10_depth:
                return (
                    target_notional,
                    self._estimate_impact(target_notional, l1_depth, l10_depth) + extra_bps,
                )

        # Conservative mode (default): use best size found
        if best_size < target_notional:
            logger.info(
                f"SIZED_DOWN: Original=${target_notional:.0f} → Adjusted=${best_size:.0f} "
                f"(Impact: {best_impact:.1f}bps, Limit: {max_impact}bps)"
            )

        return best_size, best_impact

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
        positions = self.broker.get_positions()
        position = next((p for p in positions if p.symbol == symbol), None)

        position_quantity = getattr(position, "quantity", position.quantity)
        if not position or position_quantity == 0:
            logger.warning(f"No position to close for {symbol}")
            return None

        # Determine side (opposite of position)
        side = "sell" if position_quantity > 0 else "buy"
        quantity = abs(position_quantity)

        logger.info(f"Closing position: {side} {quantity} {symbol}")

        return self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="market",
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
        triggered = []

        for trigger_id, trigger in self.stop_triggers.items():
            if trigger.triggered:
                continue

            price = current_prices.get(trigger.symbol)
            if not price:
                continue

            # Check trigger condition
            should_trigger = False
            if trigger.side == OrderSide.BUY and price >= trigger.trigger_price:
                should_trigger = True
            elif trigger.side == OrderSide.SELL and price <= trigger.trigger_price:
                should_trigger = True

            if should_trigger:
                trigger.triggered = True
                trigger.triggered_at = datetime.now()
                triggered.append(trigger_id)
                self.order_metrics["stop_triggered"] += 1

                logger.info(
                    f"Stop trigger activated: {trigger.symbol} {trigger.side} "
                    f"@ {trigger.trigger_price} (current: {price})"
                )

        return triggered

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

    def _estimate_impact(
        self, order_size: Decimal, l1_depth: Decimal, l10_depth: Decimal
    ) -> Decimal:
        """
        Estimate market impact in basis points.

        Uses square root model for realistic large order impact.
        """
        if order_size <= l1_depth:
            # Linear impact within L1
            return (order_size / l1_depth) * Decimal("5")
        elif order_size <= l10_depth:
            # Square root impact beyond L1
            l1_impact = Decimal("5")
            excess = order_size - l1_depth
            excess_depth = l10_depth - l1_depth if l10_depth > l1_depth else l1_depth
            excess_ratio = min(excess / excess_depth, Decimal("1"))
            additional_impact = excess_ratio ** Decimal("0.5") * Decimal("20")
            return l1_impact + additional_impact
        else:
            # Order exceeds L10 - very high impact
            return Decimal("100")  # 100 bps = 1%

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics."""
        return {
            "orders": self.order_metrics.copy(),
            "pending_count": len(self.pending_orders),
            "stop_triggers": len(self.stop_triggers),
            "active_stops": sum(1 for t in self.stop_triggers.values() if not t.triggered),
        }
