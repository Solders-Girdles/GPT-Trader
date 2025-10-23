"""
Integration test fixtures and utilities for comprehensive end-to-end testing.

This module provides the integration test infrastructure that stitches together
Live Risk Manager, Execution Coordinator, and Live Execution Engine for
realistic trading system validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.config.types import Profile
from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    Order,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide as Side,
)
from bot_v2.features.live_trade.risk.manager import LiveRiskManager
from bot_v2.orchestration.configuration.core import BotConfig
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.persistence.event_store import EventStore


class MockIntegrationBroker(IBrokerage):
    """Mock broker for integration testing with realistic failure modes."""

    def __init__(self, failure_mode: str | None = None):
        self.failure_mode = failure_mode
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.connection_dropped = False
        self.api_rate_limited = False
        self.maintenance_mode = False

    async def place_order(self, order: Order) -> Order:
        """Mock order placement with configurable failure modes."""
        if self.connection_dropped:
            raise ConnectionError("WebSocket connection dropped")

        if self.api_rate_limited:
            raise Exception("API rate limit exceeded")

        if self.maintenance_mode:
            raise Exception("Broker under maintenance")

        # Simulate realistic order processing
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = f"broker_{order.id}"

        self.orders[order.id] = order

        # Simulate async order update
        if self.failure_mode == "order_failure":
            order.status = OrderStatus.REJECTED
            order.error_message = "Insufficient liquidity"
        elif self.failure_mode == "partial_fill":
            order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_quantity = order.quantity * 0.5
        else:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_positions(self) -> list[Position]:
        """Mock position retrieval."""
        return list(self.positions.values())

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Mock order status retrieval."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.UNKNOWN

    def drop_connection(self):
        """Simulate connection drop."""
        self.connection_dropped = True

    def restore_connection(self):
        """Restore connection."""
        self.connection_dropped = False

    def enable_rate_limiting(self):
        """Enable API rate limiting."""
        self.api_rate_limited = True

    def disable_rate_limiting(self):
        """Disable API rate limiting."""
        self.api_rate_limited = False

    def enable_maintenance_mode(self):
        """Enable maintenance mode."""
        self.maintenance_mode = True

    def disable_maintenance_mode(self):
        """Disable maintenance mode."""
        self.maintenance_mode = False


class MockIntegrationEventStore(EventStore):
    """Mock event store for integration testing."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []

    def store_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Store an integration event."""
        self.events.append(
            {
                "type": event_type,
                "data": data,
                "timestamp": datetime.utcnow(),
            }
        )

    def store_metric(self, metric_name: str, value: float, tags: dict[str, str] = None) -> None:
        """Store a performance metric."""
        self.metrics.append(
            {
                "name": metric_name,
                "value": value,
                "tags": tags or {},
                "timestamp": datetime.utcnow(),
            }
        )

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Get events by type."""
        return [event for event in self.events if event["type"] == event_type]

    def get_metrics_by_name(self, metric_name: str) -> list[dict[str, Any]]:
        """Get metrics by name."""
        return [metric for metric in self.metrics if metric["name"] == metric_name]


class IntegrationTestScenarios:
    """Test scenario utilities for integration testing."""

    @staticmethod
    def create_test_order(
        order_id: str = "test_order_001",
        symbol: str = "BTC-USD",
        side: Side = Side.BUY,
        order_type: OrderType = OrderType.MARKET,
        quantity: float = 1.0,
        price: float | None = None,
    ) -> Order:
        """Create a test order with realistic parameters."""
        from decimal import Decimal

        now = datetime.utcnow()

        order = Order(
            id=order_id,
            client_id=None,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)) if price is not None else None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            submitted_at=now,
            updated_at=now,
        )
        return order

    @staticmethod
    def create_test_position(
        symbol: str = "BTC-USD", side: str = "long", size: float = 1.0, entry_price: float = 50000.0
    ) -> Position:
        """Create a test position with realistic parameters."""
        from decimal import Decimal

        position = Position(
            symbol=symbol,
            side=side,
            quantity=Decimal(str(size)),
            entry_price=Decimal(str(entry_price)),
            mark_price=Decimal(str(entry_price)),
            unrealized_pnl=Decimal("0.0"),
            realized_pnl=Decimal("0.0"),
        )
        return position

    @staticmethod
    def create_market_scenario(scenario_type: str = "normal") -> dict[str, Any]:
        """Create market condition scenarios for testing."""
        scenarios = {
            "normal": {
                "volatility": 0.02,
                "liquidity": "high",
                "trend": "sideways",
                "spread_bps": 5,
            },
            "high_volatility": {
                "volatility": 0.15,
                "liquidity": "medium",
                "trend": "volatile",
                "spread_bps": 25,
            },
            "low_liquidity": {
                "volatility": 0.03,
                "liquidity": "low",
                "trend": "down",
                "spread_bps": 50,
            },
            "flash_crash": {
                "volatility": 0.50,
                "liquidity": "very_low",
                "trend": "crash",
                "spread_bps": 200,
            },
        }
        return scenarios.get(scenario_type, scenarios["normal"])


@pytest.fixture
def integration_runtime_settings(tmp_path):
    """Runtime settings configured for integration testing."""
    runtime_root = tmp_path / "runtime"
    event_store_root = runtime_root / "events"
    risk_config_path = runtime_root / "risk_config.json"
    event_store_root.mkdir(parents=True, exist_ok=True)
    risk_config_path.parent.mkdir(parents=True, exist_ok=True)
    risk_config_path.write_text("{}", encoding="utf-8")

    env_map = {
        "GPT_TRADER_RUNTIME_ROOT": str(runtime_root),
        "EVENT_STORE_ROOT": str(event_store_root),
        "COINBASE_DEFAULT_QUOTE": "usd",
        "COINBASE_ENABLE_DERIVATIVES": "true",
        "PERPS_ENABLE_STREAMING": "true",
        "PERPS_STREAM_LEVEL": "2",
        "PERPS_PAPER": "false",
        "PERPS_FORCE_MOCK": "true",
        "PERPS_SKIP_RECONCILE": "true",
        "PERPS_POSITION_FRACTION": "0.25",
        "ORDER_PREVIEW_ENABLED": "true",
        "SPOT_FORCE_LIVE": "false",
        "BROKER": "mock",
        "COINBASE_SANDBOX": "true",
        "COINBASE_API_MODE": "sandbox",
        "COINBASE_INTX_PORTFOLIO_UUID": "integration-test",
        "RISK_CONFIG_PATH": str(risk_config_path),
        "LOG_LEVEL": "DEBUG",
        "INTEGRATION_TEST_MODE": "true",
    }

    return load_runtime_settings(env_map)


@pytest.fixture(scope="session")
def integration_risk_config():
    """Risk configuration for integration testing."""
    return RiskConfig(
        kill_switch_enabled=False,
        enable_pre_trade_liq_projection=True,
        max_leverage=3,  # Conservative leverage
        leverage_max_per_symbol={"BTC-USD": 2, "ETH-USD": 2},
        min_liquidation_buffer_pct=0.15,  # 15% buffer
        max_market_impact_bps=5,  # 5 bps max impact
        enable_market_impact_guard=True,
        default_maintenance_margin_rate=0.01,  # 1% MMR
    )


@pytest.fixture
def mock_integration_broker():
    """Mock broker instance for integration testing."""
    return MockIntegrationBroker()


@pytest.fixture
def mock_integration_event_store():
    """Mock event store for integration testing."""
    return MockIntegrationEventStore()


@pytest.fixture
def integration_live_risk_manager(integration_risk_config, mock_integration_event_store):
    """Live Risk Manager configured for integration testing."""
    return LiveRiskManager(config=integration_risk_config, event_store=mock_integration_event_store)


@pytest.fixture
def integration_execution_coordinator(
    integration_runtime_settings,
    mock_integration_broker,
    mock_integration_event_store,
    integration_live_risk_manager,
):
    """Execution Coordinator configured for integration testing."""

    symbols = ("BTC-USD", "ETH-USD")
    bot_config = BotConfig(
        profile=Profile.DEMO,
        dry_run=True,
        symbols=list(symbols),
        mock_broker=True,
        derivatives_enabled=True,
        perps_force_mock=True,
        perps_enable_streaming=True,
    )

    service_registry = ServiceRegistry(
        config=bot_config,
        event_store=mock_integration_event_store,
        broker=mock_integration_broker,
        risk_manager=integration_live_risk_manager,
        runtime_settings=integration_runtime_settings,
    )

    runtime_state = PerpsBotRuntimeState(symbols)

    context = CoordinatorContext(
        config=bot_config,
        registry=service_registry,
        event_store=mock_integration_event_store,
        broker=mock_integration_broker,
        risk_manager=integration_live_risk_manager,
        symbols=symbols,
        bot_id="integration_test_bot",
        runtime_state=runtime_state,
    )

    coordinator = ExecutionCoordinator(context=context)

    # Add convenience attribute for broker access
    coordinator.broker = context.broker

    return coordinator


@pytest.fixture
def integration_live_execution_engine(
    integration_runtime_settings,
    mock_integration_broker,
    mock_integration_event_store,
    integration_live_risk_manager,
):
    """Live Execution Engine configured for integration testing."""
    return LiveExecutionEngine(
        broker=mock_integration_broker,
        risk_manager=integration_live_risk_manager,
        event_store=mock_integration_event_store,
        settings=integration_runtime_settings,
        bot_id="integration_test_engine",
    )


@pytest.fixture
def integrated_trading_system(
    integration_live_risk_manager,
    integration_execution_coordinator,
    integration_live_execution_engine,
    mock_integration_event_store,
):
    """Fully integrated trading system for end-to-end testing."""
    return {
        "risk_manager": integration_live_risk_manager,
        "execution_coordinator": integration_execution_coordinator,
        "execution_engine": integration_live_execution_engine,
        "event_store": mock_integration_event_store,
    }


@pytest.fixture(scope="session")
def integration_test_scenarios():
    """Test scenario utilities."""
    return IntegrationTestScenarios()


@pytest.fixture
def async_integrated_system(integrated_trading_system):
    """Return the integrated system for async tests.

    Individual tests are responsible for awaiting any async methods on the returned components.
    """
    return integrated_trading_system


@pytest.fixture(scope="session")
def circuit_breaker_test_scenarios():
    """Circuit breaker test scenarios."""
    return {
        "daily_loss_breach": {
            "current_loss": 600.0,
            "daily_limit": 500.0,
            "expected_action": "stop_trading",
        },
        "liquidation_buffer_breach": {
            "buffer_ratio": 0.08,
            "min_buffer": 0.15,
            "expected_action": "reduce_positions",
        },
        "volatility_spike": {
            "current_volatility": 0.25,
            "volatility_threshold": 0.10,
            "expected_action": "reduce_size",
        },
        "correlation_risk": {
            "correlation": 0.95,
            "correlation_limit": 0.8,
            "expected_action": "halt_new_positions",
        },
    }


@pytest.fixture(scope="session")
def broker_error_scenarios():
    """Broker error scenarios for testing."""
    return {
        "connection_drop": {
            "error_type": "ConnectionError",
            "recovery_action": "reconnect",
            "expected_orders": "retry",
        },
        "rate_limit": {
            "error_type": "RateLimitError",
            "recovery_action": "backoff",
            "expected_orders": "queue",
        },
        "maintenance": {
            "error_type": "MaintenanceError",
            "recovery_action": "wait",
            "expected_orders": "reject",
        },
        "insufficient_liquidity": {
            "error_type": "LiquidityError",
            "recovery_action": "reduce_size",
            "expected_orders": "modify",
        },
    }
