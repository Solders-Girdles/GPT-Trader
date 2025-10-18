"""Stop trigger management for live trade execution."""

from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, Mapping, MutableMapping, ValuesView
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_execution")


@dataclass
class StopTrigger:
    """Metadata describing a stop/stop-limit trigger."""

    order_id: str
    symbol: str
    trigger_price: Decimal
    side: OrderSide
    quantity: Decimal
    limit_price: Decimal | None = None
    created_at: datetime = field(default_factory=datetime.now)
    triggered: bool = False
    triggered_at: datetime | None = None


class StopManager:
    """Manage registration and evaluation of stop triggers."""

    def __init__(self) -> None:
        self._triggers: MutableMapping[str, StopTrigger] = {}

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def register(
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
        """Register a stop trigger when submitting a stop/stop-limit order."""
        if order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
            return
        if stop_price is None:
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
        self._triggers[client_id] = trigger

    def remove(self, client_id: str) -> None:
        """Remove a trigger, typically after an error."""
        self._triggers.pop(client_id, None)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, current_prices: Mapping[str, Decimal]) -> list[tuple[str, StopTrigger]]:
        """
        Evaluate triggers against current prices.

        Returns list of (client_id, trigger) pairs that fired.
        """
        triggered: list[tuple[str, StopTrigger]] = []
        for client_id, trigger in self._triggers.items():
            if trigger.triggered:
                continue

            price = current_prices.get(trigger.symbol)
            if price is None:
                continue

            should_trigger = False
            if trigger.side == OrderSide.BUY and price >= trigger.trigger_price:
                should_trigger = True
            elif trigger.side == OrderSide.SELL and price <= trigger.trigger_price:
                should_trigger = True

            if not should_trigger:
                continue

            trigger.triggered = True
            trigger.triggered_at = datetime.now()
            triggered.append((client_id, trigger))

            logger.info(
                "Stop trigger activated: %s %s @ %s (current: %s)",
                trigger.symbol,
                trigger.side,
                trigger.trigger_price,
                price,
            )

        return triggered

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def total(self) -> int:
        """Total registered triggers."""
        return len(self._triggers)

    def active_count(self) -> int:
        """Number of triggers that have not fired yet."""
        return sum(1 for trigger in self._triggers.values() if not trigger.triggered)

    def snapshot(self) -> Iterable[tuple[str, StopTrigger]]:
        """Iterate over all tracked triggers."""
        return list(self._triggers.items())

    # ------------------------------------------------------------------
    # Mapping-style helpers (backwards compatibility)
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._triggers)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - trivial
        return iter(self._triggers)

    def __getitem__(self, key: str) -> StopTrigger:  # pragma: no cover - trivial
        return self._triggers[key]

    def values(self) -> ValuesView[StopTrigger]:  # pragma: no cover - trivial
        return self._triggers.values()

    def items(self) -> ItemsView[str, StopTrigger]:  # pragma: no cover - trivial
        return self._triggers.items()
