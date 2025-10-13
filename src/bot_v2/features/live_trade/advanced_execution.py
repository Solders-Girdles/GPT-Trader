"""Advanced execution engine with support for rich order workflows."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, cast

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.execution import (
    OrderGuards,
    OrderRouter,
    PositionSizer,
    StopManager,
)
from bot_v2.features.live_trade.risk import LiveRiskManager, PositionSizingAdvice
from bot_v2.utilities.config import load_slippage_multipliers

__all__ = [
    "SizingMode",
    "OrderConfig",
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
        slippage_multipliers: Mapping[str, Decimal] | None = None,
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
        self.position_sizer = PositionSizer(broker, risk_manager)
        self.config = config or OrderConfig()

        # Order tracking
        self.pending_orders: dict[str, Order] = {}
        self.stop_manager = StopManager()
        # Backwards compatibility: historical tests access stop_triggers dict directly
        self.stop_triggers = self.stop_manager
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

        # Router handles broker submissions and lifecycle
        self.order_router = OrderRouter(
            broker=self.broker,
            pending_orders=self.pending_orders,
            client_order_map=self.client_order_map,
            order_metrics=self.order_metrics,
        )

        # Track rejection reasons
        self.rejections_by_reason: dict[str, int] = {}
        # Guard helpers for market data + spec enforcement
        self.order_guards = OrderGuards(
            broker=self.broker,
            config=self.config,
            order_metrics=self.order_metrics,
            rejections_by_reason=self.rejections_by_reason,
        )

        # Last sizing advice for diagnostics/tests
        self._last_sizing_advice: PositionSizingAdvice | None = None

        # Optional per-symbol slippage multipliers for impact-aware sizing
        resolved_multipliers = (
            slippage_multipliers
            if slippage_multipliers is not None
            else load_slippage_multipliers()
        )
        self.slippage_multipliers = {
            symbol: Decimal(str(mult)) for symbol, mult in resolved_multipliers.items()
        }

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
        client_id = self.order_router.prepare_client_id(client_id, symbol, side)

        duplicate_order = self.order_router.check_duplicate(client_id)
        if duplicate_order:
            return duplicate_order

        try:
            order_quantity = self._normalize_quantity(quantity)
            product, quote = self.order_guards.fetch_market_data(
                symbol=symbol,
                order_type=order_type,
                post_only=post_only,
            )

            sizing_advice = self.position_sizer.maybe_apply_position_sizing(
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                limit_price=limit_price,
                product=product,
                quote=quote,
                leverage=leverage,
            )
            self._last_sizing_advice = sizing_advice

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

            if not self.order_guards.validate_post_only(
                symbol=symbol,
                order_type=order_type,
                post_only=post_only,
                side=side,
                limit_price=limit_price,
                quote=quote,
            ):
                return None

            adjustment = self.order_guards.apply_quantization_and_specs(
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

            if not self._validate_stop_order_requirements(
                symbol=symbol,
                order_type=order_type,
                stop_price=stop_price,
            ):
                return None

            self.stop_manager.register(
                order_type=order_type,
                client_id=client_id,
                symbol=symbol,
                stop_price=stop_price,
                side=side,
                order_quantity=order_quantity,
                limit_price=limit_price,
            )

            return self.order_router.submit(
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
                client_id=client_id or "",
            )

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
            validation_price = self.position_sizer.determine_reference_price(
                symbol=symbol,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                quote=quote,
                product=product,
            )
            equity = self.position_sizer.estimate_equity()
            current_positions_raw = self.position_sizer.current_positions()
            current_positions = dict(current_positions_raw) if current_positions_raw else None
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

    def _validate_stop_order_requirements(
        self,
        *,
        symbol: str,
        order_type: OrderType,
        stop_price: Decimal | None,
    ) -> bool:
        if order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
            return True

        if not self.config.enable_stop_orders:
            logger.warning("Stop orders disabled by configuration; rejecting %s", symbol)
            self.order_metrics["rejected"] += 1
            self.rejections_by_reason["stop_disabled"] = (
                self.rejections_by_reason.get("stop_disabled", 0) + 1
            )
            return False

        if stop_price is None:
            logger.warning("Stop order for %s missing stop price", symbol)
            self.order_metrics["rejected"] += 1
            self.rejections_by_reason["invalid_stop"] = (
                self.rejections_by_reason.get("invalid_stop", 0) + 1
            )
            return False

        return True

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
            self.stop_manager.remove(client_id)
        return None

    def cancel_and_replace(
        self,
        order_id: str,
        new_price: Decimal | None = None,
        new_size: Decimal | None = None,
        max_retries: int = 3,
    ) -> Order | None:
        """Cancel and replace order atomically with retry logic."""

        return self.order_router.cancel_and_replace(
            order_id=order_id,
            new_price=new_price,
            new_size=new_size,
            max_retries=max_retries,
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
        triggered_pairs = self.stop_manager.evaluate(current_prices)
        for _trigger_id, _trigger in triggered_pairs:
            self.order_metrics["stop_triggered"] += 1
        return [trigger_id for trigger_id, _ in triggered_pairs]

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
            "stop_triggers": self.stop_manager.total(),
            "active_stops": self.stop_manager.active_count(),
        }
