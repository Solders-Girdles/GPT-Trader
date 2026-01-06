"""
Scenario builder utilities for orchestration coordinator tests.

Provides factories for creating test scenarios including strategy signals,
guard responses, telemetry payloads, and common test data patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

from gpt_trader.core import (
    Balance,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    TimeInForce,
)
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision

# Constants for consistent test data
TEST_PRODUCT_IDS = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
TEST_TIMESTAMPS = [1640995200, 1641081600, 1641168000]  # UTC timestamps
TEST_RISK_FLAGS = ["high_volatility", "low_liquidity", "circuit_breaker"]


@dataclass
class StrategySignal:
    """Represents a strategy decision signal."""

    symbol: str
    action: Action
    quantity: Decimal | None = None
    target_notional: Decimal | None = None
    reason: str = "test_signal"
    leverage: Decimal | None = None
    reduce_only: bool = False
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_trigger: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC


@dataclass
class GuardResponse:
    """Represents a guard manager response."""

    is_blocking: bool
    active_guards: list[str]
    reason: str | None = None


@dataclass
class TelemetryPayload:
    """Represents a telemetry event payload."""

    event_type: str
    symbol: str | None = None
    mark: Decimal | None = None
    quantity: Decimal | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class ScenarioBuilder:
    """Factory for creating test scenarios and data."""

    @staticmethod
    def create_strategy_signal(
        symbol: str = "BTC-PERP",
        action: Action = Action.BUY,
        quantity: Decimal | None = Decimal("0.1"),
        reason: str = "test_buy_signal",
    ) -> StrategySignal:
        """Create a strategy signal for testing."""
        return StrategySignal(symbol=symbol, action=action, quantity=quantity, reason=reason)

    @staticmethod
    def create_guard_response(
        is_blocking: bool = False, active_guards: list[str] | None = None, reason: str | None = None
    ) -> GuardResponse:
        """Create a guard response for testing."""
        return GuardResponse(
            is_blocking=is_blocking, active_guards=active_guards or [], reason=reason
        )

    @staticmethod
    def create_telemetry_payload(
        event_type: str = "mark_update",
        symbol: str = "BTC-PERP",
        mark: Decimal = Decimal("50000"),
        metadata: dict[str, Any] | None = None,
    ) -> TelemetryPayload:
        """Create a telemetry payload for testing."""
        return TelemetryPayload(event_type=event_type, symbol=symbol, mark=mark, metadata=metadata)

    @staticmethod
    def create_balance(asset: str = "USDC", total: Decimal = Decimal("10000")) -> Balance:
        """Create a mock balance."""
        balance = Mock(spec=Balance)
        balance.asset = asset
        balance.total = total
        return balance

    @staticmethod
    def create_position(
        symbol: str = "BTC-PERP",
        quantity: Decimal = Decimal("0.5"),
        side: str = "long",
        entry_price: Decimal = Decimal("50000"),
    ) -> Position:
        """Create a mock position."""
        position = Mock(spec=Position)
        position.symbol = symbol
        position.quantity = quantity
        position.side = side
        position.entry_price = entry_price
        return position

    @staticmethod
    def create_product(
        symbol: str = "BTC-PERP",
        base_asset: str = "BTC",
        quote_asset: str = "USD",
        market_type: MarketType = MarketType.PERPETUAL,
        min_size: Decimal = Decimal("0.001"),
        price_increment: Decimal = Decimal("0.01"),
        leverage_max: int = 5,
    ) -> Product:
        """Create a mock product."""
        product = Mock(spec=Product)
        product.symbol = symbol
        product.base_asset = base_asset
        product.quote_asset = quote_asset
        product.market_type = market_type
        product.min_size = min_size
        product.price_increment = price_increment
        product.leverage_max = leverage_max
        return product

    @staticmethod
    def create_order(
        id: str = "test-order-123",
        symbol: str = "BTC-PERP",
        side: OrderSide = OrderSide.BUY,
        quantity: Decimal = Decimal("0.1"),
        price: Decimal | None = None,
        status: OrderStatus = OrderStatus.FILLED,
    ) -> Order:
        """Create a mock order."""
        from datetime import datetime, timezone

        order = Mock(spec=Order)
        order.id = id
        order.symbol = symbol
        order.side = side
        order.quantity = quantity
        order.price = price
        order.status = status
        order.submitted_at = datetime.now(timezone.utc)
        order.updated_at = datetime.now(timezone.utc)
        return order

    @staticmethod
    def create_decision(
        action: Action = Action.BUY,
        quantity: Decimal | None = Decimal("0.1"),
        reason: str = "test_decision",
    ) -> Decision:
        """Create a strategy decision."""
        return Decision(action=action, quantity=quantity, reason=reason)


class TestScenarios:
    """Predefined test scenarios for common coordinator flows."""

    @staticmethod
    def happy_path_buy() -> dict[str, Any]:
        """Happy path buy scenario."""
        return {
            "signal": ScenarioBuilder.create_strategy_signal(
                action=Action.BUY, quantity=Decimal("0.1")
            ),
            "guard_response": ScenarioBuilder.create_guard_response(is_blocking=False),
            "expected_action": "execute_buy",
            "expected_telemetry": ["order_placed", "position_updated"],
        }

    @staticmethod
    def guard_trigger_block() -> dict[str, Any]:
        """Guard trigger blocking scenario."""
        return {
            "signal": ScenarioBuilder.create_strategy_signal(action=Action.SELL),
            "guard_response": ScenarioBuilder.create_guard_response(
                is_blocking=True, active_guards=["circuit_breaker"], reason="volatility_spike"
            ),
            "expected_action": "skip_execution",
            "expected_telemetry": ["execution_blocked", "guard_triggered"],
        }

    @staticmethod
    def execution_failure_retry() -> dict[str, Any]:
        """Execution failure with retry scenario."""
        return {
            "signal": ScenarioBuilder.create_strategy_signal(action=Action.CLOSE),
            "guard_response": ScenarioBuilder.create_guard_response(is_blocking=False),
            "execution_error": Exception("network_timeout"),
            "expected_action": "retry_execution",
            "expected_telemetry": ["execution_failed", "retry_scheduled"],
        }

    @staticmethod
    def lifecycle_tick() -> dict[str, Any]:
        """Lifecycle tick scenario."""
        return {
            "marks_updated": True,
            "strategy_fetched": True,
            "backoff_respected": True,
            "expected_action": "tick_complete",
            "expected_telemetry": ["tick_processed", "strategy_evaluated"],
        }

    @staticmethod
    def telemetry_batch_flush() -> dict[str, Any]:
        """Telemetry batching and flush scenario."""
        return {
            "events": [
                ScenarioBuilder.create_telemetry_payload(
                    "mark_update", "BTC-PERP", Decimal("50000")
                ),
                ScenarioBuilder.create_telemetry_payload(
                    "mark_update", "ETH-PERP", Decimal("3000")
                ),
            ],
            "batch_threshold": 5,
            "flush_triggered": True,
            "expected_action": "batch_flushed",
            "expected_telemetry": ["batch_processed", "metrics_emitted"],
        }

    @staticmethod
    def runtime_state_transition() -> dict[str, Any]:
        """Runtime state transition scenario."""
        return {
            "initial_state": "active",
            "transition_to": "suspend",
            "reason": "maintenance_window",
            "propagated_to_dependents": True,
            "expected_action": "state_changed",
            "expected_telemetry": ["state_transition", "dependents_notified"],
        }
