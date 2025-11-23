"""Core advanced execution engine implementation."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Mapping, cast

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
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
from bot_v2.utilities.logging_patterns import get_logger

from . import impact, order_utils, risk
from .config import OrderConfig

logger = get_logger(__name__, component="live_trade_execution")


class AdvancedExecutionEngine:
    """Enhanced execution engine with advanced order handling."""

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
        self.broker = broker
        self.risk_manager = risk_manager
        self.position_sizer = PositionSizer(broker, risk_manager)
        self.config = config or OrderConfig()

        self.pending_orders: dict[str, Order] = {}
        self.stop_manager = StopManager()
        self.stop_triggers = self.stop_manager  # Backwards compatibility
        self.client_order_map: dict[str, str] = {}

        self.order_metrics = {
            "placed": 0,
            "filled": 0,
            "cancelled": 0,
            "rejected": 0,
            "post_only_rejected": 0,
            "stop_triggered": 0,
        }

        self.order_router = OrderRouter(
            broker=self.broker,
            pending_orders=self.pending_orders,
            client_order_map=self.client_order_map,
            order_metrics=self.order_metrics,
        )

        self.rejections_by_reason: dict[str, int] = {}
        self.order_guards = OrderGuards(
            broker=self.broker,
            config=self.config,
            order_metrics=self.order_metrics,
            rejections_by_reason=self.rejections_by_reason,
        )

        self._last_sizing_advice: PositionSizingAdvice | None = None

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
        client_id = self.order_router.prepare_client_id(client_id, symbol, side)

        duplicate_order = self.order_router.check_duplicate(client_id)
        if duplicate_order:
            return duplicate_order

        try:
            order_quantity = order_utils.normalize_quantity(quantity)
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

            if not risk.run_risk_validation(
                self,
                symbol=symbol,
                side=side,
                order_quantity=order_quantity,
                limit_price=limit_price,
                order_type=order_type,
                product=product,
                quote=quote,
            ):
                return None

            if not order_utils.validate_stop_order_requirements(
                self,
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
        return impact.calculate_impact_aware_size(
            self,
            symbol=symbol,
            target_notional=target_notional,
            market_snapshot=market_snapshot,
            max_impact_bps=max_impact_bps,
        )

    def close_position(self, symbol: str, reduce_only: bool = True) -> Order | None:
        positions = cast(list[Position], self.broker.get_positions())
        position = next((p for p in positions if p.symbol == symbol), None)

        if position is None or position.quantity == 0:
            logger.warning(f"No position to close for {symbol}")
            return None

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
        triggered_pairs = self.stop_manager.evaluate(current_prices)
        for _trigger_id, _trigger in triggered_pairs:
            self.order_metrics["stop_triggered"] += 1
        return [trigger_id for trigger_id, _ in triggered_pairs]

    def _validate_tif(self, tif: str) -> TimeInForce | None:
        tif_upper = tif.upper()

        if tif_upper == "GTC":
            return TimeInForce.GTC
        if tif_upper == "IOC" and self.config.enable_ioc:
            return TimeInForce.IOC
        if tif_upper == "FOK" and self.config.enable_fok:
            logger.warning("FOK order type is gated and not yet supported")
            return None
        logger.error(f"Unsupported or disabled TIF: {tif}")
        return None

    def _estimate_impact(
        self,
        order_size: Decimal,
        l1_depth: Decimal,
        l10_depth: Decimal,
    ) -> Decimal:
        return impact.estimate_impact(order_size, l1_depth, l10_depth)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "orders": self.order_metrics.copy(),
            "pending_count": len(self.pending_orders),
            "stop_triggers": self.stop_manager.total(),
            "active_stops": self.stop_manager.active_count(),
        }


__all__ = ["AdvancedExecutionEngine"]
