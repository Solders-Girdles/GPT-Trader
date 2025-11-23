from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.engines.base import CoordinatorContext
from bot_v2.orchestration.engines.execution import ExecutionEngine
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "coinbase_trader"


@pytest.fixture
def base_context() -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], dry_run=False)
    config.time_in_force = "GTC"
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

    broker = Mock()
    risk_manager = Mock()
    risk_manager.config = Mock(
        enable_dynamic_position_sizing=False, enable_market_impact_guard=False
    )
    risk_manager.set_impact_estimator = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=event_store,
        orders_store=orders_store,
    )

    controller = Mock()
    controller.is_reduce_only_mode.return_value = False
    controller.sync_with_risk_manager = Mock()

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id=BOT_ID,
        runtime_state=runtime_state,
        config_controller=controller,
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def coordinator(base_context: CoordinatorContext) -> ExecutionEngine:
    return ExecutionEngine(base_context)


@pytest.fixture
def test_product() -> Product:
    product = Mock(spec=Product)
    product.symbol = "BTC-PERP"
    product.market_type = MarketType.PERPETUAL
    product.base_increment = Decimal("0.0001")
    product.quote_increment = Decimal("0.01")
    return product


@pytest.fixture
def test_order() -> Order:
    now = datetime.now(timezone.utc)
    return Order(
        id="order-1",
        client_id="client-1",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        price=None,
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        avg_fill_price=Decimal("50000"),
        submitted_at=now,
        updated_at=now,
    )
