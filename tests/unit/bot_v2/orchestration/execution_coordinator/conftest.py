"""
Fixtures for ExecutionCoordinator tests.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
)
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def fake_perps_bot():
    """Create fake PerpsBot with async broker methods."""
    bot = Mock()
    bot.config = BotConfig(profile=Profile.PROD, dry_run=False)
    bot.config.enable_order_preview = False
    bot.config.time_in_force = "GTC"

    # Async broker methods
    bot.broker = Mock()
    bot.broker.list_balances = AsyncMock()
    bot.broker.list_positions = AsyncMock()
    bot.broker.get_candles = AsyncMock()
    bot.broker.get_order = AsyncMock()
    bot.broker.list_orders = AsyncMock()

    # Risk manager with kill switch
    bot.risk_manager = Mock()
    bot.risk_manager.config = Mock()
    bot.risk_manager.config.kill_switch_enabled = False
    bot.risk_manager.config.enable_dynamic_position_sizing = False
    bot.risk_manager.config.enable_market_impact_guard = False
    bot.risk_manager.set_impact_estimator = Mock()

    # Runtime state
    state = PerpsBotRuntimeState([])
    bot.runtime_state = state

    # Strategy and execution
    bot.execute_decision = AsyncMock()
    bot.get_product = Mock()

    return bot


@pytest.fixture
def fake_balance():
    """Create test balance."""
    balance = Mock(spec=Balance)
    balance.asset = "USDC"
    balance.total = Decimal("10000")
    return balance


@pytest.fixture
def fake_position():
    """Create test position."""
    position = Mock(spec=Position)
    position.symbol = "BTC-PERP"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("50000")
    return position


@pytest.fixture
def fake_product():
    """Create test product."""
    product = Mock(spec=Product)
    product.symbol = "BTC-PERP"
    product.base_asset = "BTC"
    product.quote_asset = "USD"
    product.market_type = MarketType.PERPETUAL
    product.min_size = Decimal("0.001")
    product.step_size = Decimal("0.001")
    product.min_notional = Decimal("1")
    product.price_increment = Decimal("0.01")
    product.leverage_max = 5
    return product


@pytest.fixture
def fake_order():
    """Create test order."""
    order = Mock(spec=Order)
    order.id = "test-order-123"
    order.symbol = "BTC-PERP"
    order.side = OrderSide.BUY
    order.type = OrderType.MARKET
    order.quantity = Decimal("0.1")
    order.price = None
    order.status = OrderStatus.SUBMITTED
    order.filled_quantity = Decimal("0")
    order.avg_fill_price = None
    order.submitted_at = Mock()
    order.updated_at = Mock()
    return order


@pytest.fixture
def execution_context(fake_perps_bot):
    """Create ExecutionCoordinator context."""
    from bot_v2.orchestration.coordinators.base import CoordinatorContext

    config = BotConfig(profile=Profile.PROD, dry_run=False)
    runtime_state = PerpsBotRuntimeState([])
    broker = fake_perps_bot.broker
    risk_manager = fake_perps_bot.risk_manager

    registry = ServiceRegistry(config=config, broker=broker, risk_manager=risk_manager)
    context = CoordinatorContext(
        config=config,
        registry=registry,
        runtime_state=runtime_state,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="test-bot",
    )
    return context


@pytest.fixture
def execution_coordinator(execution_context):
    """Create ExecutionCoordinator instance."""
    return ExecutionCoordinator(execution_context)
