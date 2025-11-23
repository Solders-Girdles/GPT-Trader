"""Shared fixtures for PerpsBot end-to-end tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# Import helpers for reusable test data
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    IBrokerage,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
)
from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.execution.state_collection import StateCollector
from bot_v2.orchestration.perps_bot import PerpsBot, _CallableSymbolProcessor
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.runtime_settings import RuntimeSettings
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore
from bot_v2.security.order_validator import OrderValidator


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_event_store(temp_dir):
    """Mock event store for testing."""
    event_store = MagicMock(spec=EventStore)
    event_store.root_path = temp_dir
    event_store.append_metric = MagicMock()
    event_store.append_event = MagicMock()
    return event_store


@pytest.fixture
def mock_orders_store():
    """Mock orders store for testing."""
    return MagicMock(spec=OrdersStore)


@pytest.fixture
def mock_session_guard():
    """Mock trading session guard."""
    guard = MagicMock()
    guard.is_trading_window = MagicMock(return_value=True)
    guard.should_continue_trading = True
    return guard


@pytest.fixture
def mock_baseline_snapshot():
    """Mock baseline snapshot for configuration drift detection."""
    from datetime import UTC, datetime

    from bot_v2.monitoring.configuration_guardian import BaselineSnapshot

    return BaselineSnapshot(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        config_dict={"symbols": ["BTC-PERP"], "mock_broker": True},
        config_hash="test_hash",
        env_keys=set(),
        critical_env_values={},
        active_symbols=["BTC-PERP"],
        open_positions={},
        account_equity=None,
        total_exposure=Decimal("0"),
        profile="dev",
        broker_type="mock",
        risk_limits={},
    )


@pytest.fixture
def mock_configuration_guardian():
    """Mock configuration guardian."""
    guardian = MagicMock(spec=ConfigurationGuardian)
    guardian.reset_baseline = MagicMock()
    guardian.check_for_drift = MagicMock(return_value=None)
    return guardian


@pytest.fixture
def minimal_bot_config():
    """Minimal bot configuration for testing."""
    return BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        update_interval=1,
        mock_broker=True,
    )


@pytest.fixture
def full_bot_config():
    """Full-featured bot configuration for testing."""
    return BotConfig(
        profile=Profile.LIVE,
        symbols=["BTC-PERP", "ETH-PERP"],
        update_interval=5,
        mock_broker=False,
        derivatives_enabled=True,
        trading_window_start="10:00",
        trading_window_end="16:00",
        trading_days=["monday", "tuesday", "wednesday", "thursday", "friday"],
    )


@pytest.fixture
def mock_runtime_settings():
    """Mock runtime settings."""
    settings = MagicMock(spec=RuntimeSettings)
    settings.raw_env = {
        "PERPS_COLLATERAL_ASSETS": "USD,USDC,ETH",
        "COINBASE_DEFAULT_QUOTE": "USD",
    }
    settings.coinbase_default_quote = "USD"
    return settings


@pytest.fixture
def mock_brokerage():
    """Mock brokerage adapter."""
    broker = MagicMock(spec=IBrokerage)

    # Mock balance operations
    broker.list_balances.return_value = [
        Balance(
            asset="USD", available=Decimal("10000.0"), total=Decimal("10000.0"), hold=Decimal("0")
        ),
        Balance(
            asset="USDC", available=Decimal("5000.0"), total=Decimal("5000.0"), hold=Decimal("0")
        ),
    ]

    # Mock position operations
    broker.list_positions.return_value = []

    # Mock product operations
    mock_product = MagicMock(spec=Product)
    mock_product.symbol = "BTC-PERP"
    mock_product.base_asset = "BTC"
    mock_product.quote_asset = "USD"
    mock_product.market_type = "perpetual"
    mock_product.bid_price = Decimal("50000.0")
    mock_product.ask_price = Decimal("50010.0")
    mock_product.price = Decimal("50005.0")
    mock_product.step_size = Decimal("0.00000001")
    mock_product.min_size = Decimal("0.001")
    mock_product.price_increment = Decimal("0.01")
    mock_product.min_notional = Decimal("10")
    broker.get_product.return_value = mock_product

    # Mock order operations
    def place_order(**kwargs):
        order = MagicMock(spec=Order)
        order.id = "test-order-123"
        order.client_id = kwargs.get("client_id", "test-client-123")
        order.symbol = kwargs.get("symbol", "BTC-PERP")
        order.side = kwargs.get("side", OrderSide.BUY)
        order.type = kwargs.get("order_type", OrderType.MARKET)
        order.quantity = kwargs.get("quantity", Decimal("0.01"))
        order.price = kwargs.get("price")
        order.status = OrderStatus.FILLED
        order.filled_quantity = kwargs.get("quantity", Decimal("0.01"))
        order.avg_fill_price = Decimal("50000.0")
        order.submitted_at = datetime.now(UTC)
        order.updated_at = datetime.now(UTC)
        return order

    broker.place_order = place_order
    broker.get_order = place_order

    # Mock streaming operations
    broker.stream_orderbook = MagicMock()
    broker.stream_trades = MagicMock()

    return broker


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    risk_manager = MagicMock()
    risk_manager.set_reduce_only_mode = MagicMock()
    risk_manager.is_reduce_only_mode = MagicMock(return_value=False)
    risk_manager.validate_order = MagicMock(return_value=True)
    return risk_manager


@pytest.fixture
def mock_state_collector():
    """Mock state collector."""
    collector = MagicMock(spec=StateCollector)
    collector.collect_account_state = MagicMock(
        return_value=(
            [],  # balances
            Decimal("10000.0"),  # equity
            [],  # collateral_balances
            Decimal("10000.0"),  # total_balance
            [],  # positions
        )
    )
    return collector


@pytest.fixture
def mock_order_validator():
    """Mock order validator."""
    validator = MagicMock(spec=OrderValidator)
    validator.validate_exchange_rules = MagicMock(return_value=(MagicMock(), None))
    validator.ensure_mark_is_fresh = MagicMock(return_value=None)
    validator.enforce_slippage_guard = MagicMock(return_value=None)
    validator.run_pre_trade_validation = MagicMock(return_value=None)
    validator.maybe_preview_order = MagicMock(return_value=(MagicMock(), None))
    validator.finalize_reduce_only_flag = MagicMock(return_value=MagicMock())
    return validator


@pytest.fixture
def service_registry(minimal_bot_config, mock_brokerage, mock_risk_manager, mock_runtime_settings):
    """Service registry with mocked dependencies."""
    return ServiceRegistry(
        config=minimal_bot_config,
        broker=mock_brokerage,
        risk_manager=mock_risk_manager,
        runtime_settings=mock_runtime_settings,
    )


@pytest.fixture
def mock_config_controller(minimal_bot_config):
    """Mock config controller."""
    controller = MagicMock(spec=ConfigController)
    controller.current = minimal_bot_config
    controller.sync_with_risk_manager = MagicMock()

    # Mock config change creation
    def mock_apply_change(diff):
        new_config = minimal_bot_config.model_copy()
        for key, value in diff.items():
            setattr(new_config, key, value)

        from bot_v2.orchestration.config_controller import ConfigChange

        change = MagicMock(spec=ConfigChange)
        change.diff = diff
        change.updated = new_config
        return change

    controller.apply_change = mock_apply_change
    return controller


@pytest.fixture
def perps_bot_state():
    """PerpsBot runtime state for testing."""
    return PerpsBotRuntimeState(["BTC-PERP"])


@pytest.fixture
def perps_bot_instance(
    mock_config_controller,
    service_registry,
    mock_event_store,
    mock_orders_store,
    mock_session_guard,
    mock_baseline_snapshot,
    mock_configuration_guardian,
):
    """Create a real PerpsBot instance for testing."""
    # Patch streaming background to avoid actual network calls
    original_init = PerpsBot.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Override streaming methods to prevent background tasks
        self._start_streaming_background = lambda: None
        self._stop_streaming_background = lambda: None

    with pytest.MonkeyPatch().context() as m:
        m.setattr(PerpsBot, "__init__", patched_init)
        bot = PerpsBot(
            config_controller=mock_config_controller,
            registry=service_registry,
            event_store=mock_event_store,
            orders_store=mock_orders_store,
            session_guard=mock_session_guard,
            baseline_snapshot=mock_baseline_snapshot,
            configuration_guardian=mock_configuration_guardian,
        )

    return bot


@pytest.fixture
def callable_symbol_processor():
    """Sample callable symbol processor for testing."""

    def processor(symbol: str, balances=None, position_map=None):
        return {
            "symbol": symbol,
            "decision": "BUY",
            "quantity": Decimal("0.01"),
            "reason": "test_signal",
        }

    return processor


@pytest.fixture
def callable_symbol_processor_no_context():
    """Sample callable symbol processor that doesn't need context."""

    def processor(symbol: str):
        return {
            "symbol": symbol,
            "decision": "SELL",
            "quantity": Decimal("0.02"),
            "reason": "simple_signal",
        }

    return processor


@pytest.fixture
def wrapped_symbol_processor(callable_symbol_processor):
    """Wrapped symbol processor using _CallableSymbolProcessor adapter."""
    return _CallableSymbolProcessor(
        func=callable_symbol_processor,
        requires_context=True,
    )


@pytest.fixture
def sample_positions():
    """Sample position data for testing."""
    return [
        ScenarioBuilder.create_position(
            symbol="BTC-PERP", quantity=Decimal("0.5"), side="long", entry_price=Decimal("45000.0")
        ),
        ScenarioBuilder.create_position(
            symbol="ETH-PERP", quantity=Decimal("-2.0"), side="short", entry_price=Decimal("3000.0")
        ),
    ]


@pytest.fixture
def sample_balances():
    """Sample balance data for testing."""
    return [
        ScenarioBuilder.create_balance(asset="USD", total=Decimal("15000.0")),
        ScenarioBuilder.create_balance(asset="USDC", total=Decimal("25000.0")),
        ScenarioBuilder.create_balance(asset="ETH", total=Decimal("2.0")),
        ScenarioBuilder.create_balance(asset="BTC", total=Decimal("0.1")),
    ]


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
