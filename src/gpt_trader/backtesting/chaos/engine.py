"""Chaos engine for testing strategy robustness."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.backtesting.types import ChaosScenario
from gpt_trader.features.brokerages.core.interfaces import Candle, Order, OrderStatus

if TYPE_CHECKING:
    from gpt_trader.backtesting.simulation.broker import SimulatedBroker


@dataclass
class ChaosEvent:
    """Record of a chaos injection event."""

    timestamp: datetime
    scenario_name: str
    event_type: str
    symbol: str | None
    details: dict[str, Any] = field(default_factory=dict)


class ChaosEngine:
    """
    Injects controlled chaos into backtesting simulation.

    Tests strategy robustness against:
    - Missing candles / data gaps
    - Stale market data
    - Wide spreads / illiquidity
    - Order errors / rejections
    - Partial fills
    - Network latency

    Usage:
        engine = ChaosEngine(broker)
        engine.add_scenario(ChaosScenario(
            name="volatile_market",
            spread_multiplier=Decimal("3.0"),
            partial_fill_probability=Decimal("0.3"),
        ))
        engine.enable()

        # During simulation, chaos events are automatically injected
    """

    def __init__(
        self,
        broker: SimulatedBroker,
        seed: int | None = None,
    ):
        """
        Initialize chaos engine.

        Args:
            broker: SimulatedBroker to inject chaos into
            seed: Random seed for reproducibility
        """
        self.broker = broker
        self._scenarios: list[ChaosScenario] = []
        self._active_scenarios: list[ChaosScenario] = []
        self._events: list[ChaosEvent] = []
        self._enabled = False

        self._seed: int | None = seed
        if seed is not None:
            random.seed(seed)

    def add_scenario(self, scenario: ChaosScenario) -> None:
        """
        Add a chaos scenario to the engine.

        Args:
            scenario: Scenario configuration
        """
        self._scenarios.append(scenario)
        if scenario.enabled:
            self._active_scenarios.append(scenario)

    def enable(self) -> None:
        """Enable chaos injection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable chaos injection."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if chaos is enabled."""
        return self._enabled and len(self._active_scenarios) > 0

    def process_candle(
        self,
        symbol: str,
        candle: Candle,
        timestamp: datetime,
    ) -> Candle | None:
        """
        Process a candle through chaos scenarios.

        May drop, delay, or modify the candle.

        Args:
            symbol: Trading symbol
            candle: Original candle
            timestamp: Current simulation time

        Returns:
            Modified candle or None if dropped
        """
        if not self.is_enabled():
            return candle

        result_candle = candle

        for scenario in self._active_scenarios:
            # Missing candles
            if scenario.missing_candles_probability > 0:
                if random.random() < float(scenario.missing_candles_probability):
                    self._record_event(
                        timestamp=timestamp,
                        scenario_name=scenario.name,
                        event_type="candle_dropped",
                        symbol=symbol,
                        details={"original_ts": str(candle.ts)},
                    )
                    return None

            # Wide spreads (modify candle high/low)
            if scenario.spread_multiplier > Decimal("1"):
                mid = (candle.high + candle.low) / Decimal("2")
                half_spread = (candle.high - candle.low) / Decimal("2")
                new_half_spread = half_spread * scenario.spread_multiplier

                result_candle = Candle(
                    ts=candle.ts,
                    open=candle.open,
                    high=mid + new_half_spread,
                    low=mid - new_half_spread,
                    close=candle.close,
                    volume=candle.volume,
                )

                self._record_event(
                    timestamp=timestamp,
                    scenario_name=scenario.name,
                    event_type="spread_widened",
                    symbol=symbol,
                    details={
                        "multiplier": str(scenario.spread_multiplier),
                        "new_high": str(result_candle.high),
                        "new_low": str(result_candle.low),
                    },
                )

            # Stale marks (delay candle timestamp)
            if scenario.stale_marks_delay_seconds > 0:
                delayed_ts = candle.ts - timedelta(seconds=scenario.stale_marks_delay_seconds)
                result_candle = Candle(
                    ts=delayed_ts,
                    open=result_candle.open,
                    high=result_candle.high,
                    low=result_candle.low,
                    close=result_candle.close,
                    volume=result_candle.volume,
                )

                self._record_event(
                    timestamp=timestamp,
                    scenario_name=scenario.name,
                    event_type="mark_stale",
                    symbol=symbol,
                    details={"delay_seconds": scenario.stale_marks_delay_seconds},
                )

        return result_candle

    def process_order(
        self,
        order: Order,
        timestamp: datetime,
    ) -> Order:
        """
        Process an order through chaos scenarios.

        May reject or partially fill the order.

        Args:
            order: Order to process
            timestamp: Current simulation time

        Returns:
            Modified order
        """
        if not self.is_enabled():
            return order

        for scenario in self._active_scenarios:
            # Order errors
            if scenario.order_error_probability > 0:
                if random.random() < float(scenario.order_error_probability):
                    order.status = OrderStatus.REJECTED
                    self._record_event(
                        timestamp=timestamp,
                        scenario_name=scenario.name,
                        event_type="order_rejected",
                        symbol=order.symbol,
                        details={"order_id": order.id, "reason": "chaos_injection"},
                    )
                    return order

            # Partial fills
            if scenario.partial_fill_probability > 0:
                if random.random() < float(scenario.partial_fill_probability):
                    fill_pct = scenario.partial_fill_pct / Decimal("100")
                    order.filled_quantity = order.quantity * fill_pct
                    order.status = OrderStatus.PARTIALLY_FILLED

                    self._record_event(
                        timestamp=timestamp,
                        scenario_name=scenario.name,
                        event_type="partial_fill",
                        symbol=order.symbol,
                        details={
                            "order_id": order.id,
                            "fill_pct": str(scenario.partial_fill_pct),
                            "filled_quantity": str(order.filled_quantity),
                        },
                    )
                    return order

        return order

    def apply_latency(self, timestamp: datetime) -> datetime:
        """
        Apply network latency to a timestamp.

        Args:
            timestamp: Original timestamp

        Returns:
            Delayed timestamp
        """
        if not self.is_enabled():
            return timestamp

        total_latency_ms = 0
        for scenario in self._active_scenarios:
            if scenario.network_latency_ms > 0:
                total_latency_ms += scenario.network_latency_ms

        if total_latency_ms > 0:
            return timestamp + timedelta(milliseconds=total_latency_ms)

        return timestamp

    def _record_event(
        self,
        timestamp: datetime,
        scenario_name: str,
        event_type: str,
        symbol: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a chaos event."""
        self._events.append(
            ChaosEvent(
                timestamp=timestamp,
                scenario_name=scenario_name,
                event_type=event_type,
                symbol=symbol,
                details=details or {},
            )
        )

    def get_events(
        self,
        scenario_name: str | None = None,
        event_type: str | None = None,
        symbol: str | None = None,
    ) -> list[ChaosEvent]:
        """
        Get recorded chaos events with optional filtering.

        Args:
            scenario_name: Filter by scenario name
            event_type: Filter by event type
            symbol: Filter by symbol

        Returns:
            List of matching events
        """
        events = self._events

        if scenario_name:
            events = [e for e in events if e.scenario_name == scenario_name]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if symbol:
            events = [e for e in events if e.symbol == symbol]

        return events

    def get_statistics(self) -> dict[str, Any]:
        """Get chaos injection statistics."""
        event_counts: dict[str, int] = {}
        for event in self._events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        return {
            "total_events": len(self._events),
            "scenarios_active": len(self._active_scenarios),
            "events_by_type": event_counts,
            "seed": self._seed,
        }

    def reset(self) -> None:
        """Reset chaos engine state."""
        self._events = []
        if self._seed is not None:
            random.seed(self._seed)
