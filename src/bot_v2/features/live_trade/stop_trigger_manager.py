"""Stop-trigger handling for advanced execution orders."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.features.live_trade.advanced_execution_models.models import (
    OrderConfig,
    StopTrigger,
)

__all__ = ["StopTriggerManager"]

logger = logging.getLogger(__name__)


class StopTriggerManager:
    """Manages stop order triggers and monitoring."""

    def __init__(self, config: OrderConfig | None = None) -> None:
        """
        Initialize stop trigger manager.

        Args:
            config: Order configuration
        """
        self.config = config or OrderConfig()
        self.stop_triggers: dict[str, StopTrigger] = {}

        # Metrics
        self._trigger_count = 0

    def validate_stop_order_requirements(
        self,
        *,
        symbol: str,
        order_type: OrderType,
        stop_price: Decimal | None,
    ) -> tuple[bool, str | None]:
        """
        Validate stop order requirements.

        Args:
            symbol: Trading symbol
            order_type: Type of order
            stop_price: Stop price

        Returns:
            (is_valid, rejection_reason)
        """
        if order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
            return True, None

        if not self.config.enable_stop_orders:
            logger.warning("Stop orders disabled by configuration; rejecting %s", symbol)
            return False, "stop_disabled"

        if stop_price is None:
            logger.warning("Stop order for %s missing stop price", symbol)
            return False, "invalid_stop"

        return True, None

    def register_stop_trigger(
        self,
        *,
        order_type: OrderType,
        client_id: str,
        symbol: str,
        stop_price: Decimal | None,
        side: OrderSide,
        order_quantity: Decimal,
        limit_price: Decimal | None,
    ) -> None:
        """
        Register a stop trigger for monitoring.

        Args:
            order_type: Type of order
            client_id: Client order ID
            symbol: Trading symbol
            stop_price: Stop trigger price
            side: Order side
            order_quantity: Order quantity
            limit_price: Limit price (for stop-limit orders)
        """
        if order_type not in (OrderType.STOP, OrderType.STOP_LIMIT) or stop_price is None:
            return

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

    def unregister_stop_trigger(self, client_id: str) -> None:
        """
        Remove a stop trigger.

        Args:
            client_id: Client order ID to remove
        """
        self.stop_triggers.pop(client_id, None)

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
                self._trigger_count += 1

                logger.info(
                    f"Stop trigger activated: {trigger.symbol} {trigger.side} "
                    f"@ {trigger.trigger_price} (current: {price})"
                )

        return triggered

    def get_metrics(self) -> dict[str, int]:
        """Get stop trigger metrics."""
        return {
            "stop_triggers": len(self.stop_triggers),
            "active_stops": sum(1 for t in self.stop_triggers.values() if not t.triggered),
            "stop_triggered": self._trigger_count,
        }
