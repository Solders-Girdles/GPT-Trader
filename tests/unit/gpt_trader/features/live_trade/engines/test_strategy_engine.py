"""
Unit tests for Strategy Engine dynamic sizing and state tracking.
"""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.app.config import BotConfig, BotRiskConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.core import (
    Balance,
    OrderSide,
    OrderType,
    Position,
    Product,
)
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_ticker.return_value = {"price": "50000"}
    broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("1000")),
        Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
    ]
    broker.list_positions.return_value = [
        Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    ]
    return broker


@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.decide.return_value = Decision(Action.HOLD, "test")
    # Add config attribute
    config = MagicMock()
    config.position_fraction = None
    strategy.config = config
    return strategy


@pytest.fixture
def context(mock_broker):
    risk = BotRiskConfig(position_fraction=Decimal("0.1"))
    config = BotConfig(symbols=["BTC-USD"], interval=1, risk=risk)
    risk_manager = MagicMock()
    # Ensure start_of_day_equity is a valid number for comparisons
    risk_manager._start_of_day_equity = Decimal("1000.0")
    # Default check_mark_staleness to return False (fresh mark price)
    risk_manager.check_mark_staleness.return_value = False
    # Mock config with degradation settings for graceful degradation tests
    risk_manager.config = MagicMock()
    risk_manager.config.broker_outage_max_failures = 3
    risk_manager.config.broker_outage_cooldown_seconds = 120
    risk_manager.config.mark_staleness_cooldown_seconds = 120
    risk_manager.config.mark_staleness_allow_reduce_only = True
    risk_manager.config.slippage_failure_pause_after = 3
    risk_manager.config.slippage_pause_seconds = 60
    risk_manager.config.validation_failure_cooldown_seconds = 180
    risk_manager.config.preview_failure_disable_after = 5
    risk_manager.config.api_health_cooldown_seconds = 300
    return CoordinatorContext(config=config, broker=mock_broker, risk_manager=risk_manager)


@pytest.fixture
def application_container(context):
    """Set up application container for TradingEngine tests."""
    container = ApplicationContainer(context.config)
    set_application_container(container)
    yield container
    clear_application_container()


@pytest.fixture
def engine(context, mock_strategy, application_container):
    with patch(
        "gpt_trader.features.live_trade.engines.strategy.create_strategy",
        return_value=mock_strategy,
    ):
        engine = TradingEngine(context)
        # Mock strategy explicitly in case init created a new one
        engine.strategy = mock_strategy

        # Mock StateCollector methods for tests that don't override them
        engine._state_collector = MagicMock()
        engine._state_collector.require_product.return_value = MagicMock()
        engine._state_collector.resolve_effective_price.return_value = Decimal("50000")
        engine._state_collector.build_positions_dict.return_value = {}

        # Mock OrderValidator to pass by default (tests can override)
        engine._order_validator = MagicMock()
        engine._order_validator.validate_exchange_rules.return_value = (
            Decimal("0.02"),
            None,
        )
        engine._order_validator.enforce_slippage_guard.return_value = None
        engine._order_validator.ensure_mark_is_fresh.return_value = None
        engine._order_validator.run_pre_trade_validation.return_value = None
        engine._order_validator.maybe_preview_order.return_value = None
        engine._order_validator.finalize_reduce_only_flag.return_value = False

        # Mock OrderSubmitter
        engine._order_submitter = MagicMock()

        return engine


def test_reset_daily_tracking_recomputes_equity(engine):
    """reset_daily_tracking recomputes equity and invalidates guard cache."""
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("1000"))
    ]
    engine._state_collector.calculate_equity_from_balances.return_value = (
        Decimal("1000"),
        [],
        Decimal("1000"),
    )
    engine.context.risk_manager.reset_daily_tracking = MagicMock()
    engine._guard_manager = MagicMock()

    engine.reset_daily_tracking()

    engine.context.broker.list_balances.assert_called_once()
    engine._state_collector.calculate_equity_from_balances.assert_called_once()
    engine.context.risk_manager.reset_daily_tracking.assert_called_once()
    engine._guard_manager.invalidate_cache.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_total_equity_success(engine):
    """Test successful equity fetch including non-USD assets valued in USD."""
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("800")),
        Balance(asset="USDC", total=Decimal("500"), available=Decimal("200")),
        Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
    ]
    # No positions in this test case
    positions = {}

    equity = await engine._fetch_total_equity(positions)
    assert equity == Decimal("51000")  # 800 + 200 + (1 BTC @ 50,000)


@pytest.mark.asyncio
async def test_fetch_total_equity_includes_unrealized_pnl(engine):
    """Test equity fetch includes unrealized PnL from positions."""
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("1000")),
    ]
    positions = {
        "BTC-USD": Position(
            symbol="BTC-USD",
            quantity=Decimal("1"),
            entry_price=Decimal("0"),
            mark_price=Decimal("0"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            side="long",
        ),
        "ETH-USD": Position(
            symbol="ETH-USD",
            quantity=Decimal("10"),
            entry_price=Decimal("0"),
            mark_price=Decimal("0"),
            unrealized_pnl=Decimal("-200"),
            realized_pnl=Decimal("0"),
            side="short",
        ),
    }

    equity = await engine._fetch_total_equity(positions)
    # 1000 (collateral) + 500 (BTC PnL) - 200 (ETH PnL) = 1300
    assert equity == Decimal("1300")


@pytest.mark.asyncio
async def test_fetch_total_equity_failure(engine):
    """Test equity fetch handles broker errors gracefully."""
    engine.context.broker.list_balances.side_effect = Exception("API Error")
    equity = await engine._fetch_total_equity({})
    assert equity is None


@pytest.mark.asyncio
async def test_fetch_positions_success(engine):
    """Test successful position fetch converting to dict."""
    positions = await engine._fetch_positions()
    assert "BTC-USD" in positions
    assert positions["BTC-USD"].quantity == Decimal("0.5")


@pytest.mark.asyncio
async def test_fetch_positions_failure(engine):
    """Test position fetch handles broker errors gracefully."""
    engine.context.broker.list_positions.side_effect = Exception("API Error")
    positions = await engine._fetch_positions()
    assert positions == {}


@pytest.mark.asyncio
async def test_cycle_skips_on_equity_failure(engine):
    """Test cycle aborts if equity cannot be fetched."""
    engine.context.broker.list_balances.side_effect = Exception("API Error")

    # Spy on strategy.decide to ensure it's NOT called
    await engine._cycle()
    engine.strategy.decide.assert_not_called()


def test_build_position_state(engine):
    """Test position state formatting."""
    positions = {
        "BTC-USD": Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    }

    state = engine._build_position_state("BTC-USD", positions)
    assert state["quantity"] == Decimal("0.5")
    assert state["entry_price"] == Decimal("40000")
    assert state["side"] == "long"

    state_none = engine._build_position_state("ETH-USD", positions)
    assert state_none is None


def test_positions_to_risk_format(engine):
    """Test conversion of positions to risk manager dict format."""
    positions = {
        "BTC-USD": Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    }

    risk_format = engine._positions_to_risk_format(positions)
    assert "BTC-USD" in risk_format
    assert risk_format["BTC-USD"]["quantity"] == Decimal("0.5")
    assert risk_format["BTC-USD"]["mark"] == Decimal("50000")
    # Ensure no Position object leakage
    assert not isinstance(risk_format["BTC-USD"], Position)


@pytest.mark.asyncio
async def test_risk_manager_receives_dict_format(engine):
    """Test that risk manager receives correctly formatted dicts."""
    # Setup risk manager
    mock_risk_manager = MagicMock()
    mock_risk_manager._start_of_day_equity = Decimal("1000.0")
    mock_risk_manager.pre_trade_validate.return_value = MagicMock(is_valid=True)
    engine.context.risk_manager = mock_risk_manager

    # Setup security validator via patch
    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        # Setup engine state to force a trade
        engine.strategy.decide.return_value = Decision(Action.BUY, "test")
        engine.strategy.config.position_fraction = Decimal("0.1")

        # Setup mock broker responses
        engine.context.broker.list_positions.return_value = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("40000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]
        engine.context.broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
        ]

        # Run cycle
        await engine._cycle()

    # Verify pre_trade_validate call
    mock_risk_manager.pre_trade_validate.assert_called_once()
    call_args = mock_risk_manager.pre_trade_validate.call_args
    current_positions = call_args.kwargs["current_positions"]

    assert "BTC-USD" in current_positions
    assert isinstance(current_positions["BTC-USD"], dict)
    assert current_positions["BTC-USD"]["quantity"] == Decimal("1.0")


def test_calculate_order_quantity_with_strategy_config(engine):
    """Test quantity calculation uses strategy config if set."""
    engine.strategy.config.position_fraction = Decimal("0.5")
    equity = Decimal("10000")
    price = Decimal("50000")

    # Target notional = 10000 * 0.5 = 5000
    # Quantity = 5000 / 50000 = 0.1
    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, None)
    assert quantity == Decimal("0.1")


def test_calculate_order_quantity_fallback_to_bot_config(engine):
    """Test quantity calculation falls back to bot config."""
    engine.strategy.config.position_fraction = None
    # Production code looks for perps_position_fraction on the config, not risk.position_fraction
    engine.context.config.perps_position_fraction = Decimal("0.2")

    equity = Decimal("10000")
    price = Decimal("50000")

    # Target notional = 10000 * 0.2 = 2000
    # Quantity = 2000 / 50000 = 0.04
    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, None)
    assert quantity == Decimal("0.04")


def test_calculate_order_quantity_min_size(engine):
    """Test quantity respects product min size."""
    engine.strategy.config.position_fraction = Decimal("0.1")
    equity = Decimal("100")  # Small equity
    price = Decimal("50000")  # High price

    # Target notional = 100 * 0.1 = 10
    # Quantity = 10 / 50000 = 0.0002

    product = MagicMock(spec=Product)
    product.min_size = Decimal("0.001")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, product)
    assert quantity == Decimal("0")  # Should reject as too small


@pytest.mark.asyncio
async def test_order_placed_with_dynamic_quantity(engine):
    """Test full flow from decision to order placement with calculated size."""
    # Setup
    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    # Setup risk manager
    engine.context.risk_manager.pre_trade_validate.return_value.is_valid = True

    # Setup security validator via patch
    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        # Execute
        await engine._cycle()

    # Verify
    # Target notional = 10000 * 0.1 = 1000
    # Quantity = 1000 / 50000 = 0.02
    engine._order_submitter.submit_order.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order.call_args[1]
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["side"] == OrderSide.BUY
    assert call_kwargs["order_type"] == OrderType.MARKET
    assert call_kwargs["order_quantity"] == Decimal("0.02")


@pytest.mark.asyncio
async def test_position_state_passed_to_strategy(engine):
    """Verify strategy receives correct position state."""
    engine.context.broker.list_positions.return_value = [
        Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    ]
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]

    await engine._cycle()

    # Extract call args
    call_args = engine.strategy.decide.call_args
    position_state = call_args.kwargs["position_state"]

    assert position_state is not None
    assert position_state["quantity"] == Decimal("1.0")
    assert position_state["entry_price"] == Decimal("45000")
    assert position_state["side"] == "long"


@pytest.mark.asyncio
async def test_reduce_only_clamps_quantity_to_prevent_position_flip(engine):
    """Test that reduce-only mode clamps order quantity to prevent position flip."""
    # Setup: 1 BTC long position, strategy wants to sell 2 BTC (would flip to short)
    engine.strategy.decide.return_value = Decision(Action.SELL, "test")
    engine.strategy.config.position_fraction = Decimal("0.2")  # Would want to sell 2 BTC
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("500000"), available=Decimal("500000"))
    ]
    engine.context.broker.list_positions.return_value = [
        Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),  # Only 1 BTC to sell
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("10000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    ]

    # Enable reduce-only mode
    engine.context.risk_manager._reduce_only_mode = True
    engine.context.risk_manager._daily_pnl_triggered = False
    engine.context.risk_manager.check_order.return_value = True
    engine.context.risk_manager.pre_trade_validate.return_value.is_valid = True

    # Make validate_exchange_rules return the input quantity (preserves clamping)
    engine._order_validator.validate_exchange_rules.side_effect = lambda **kw: (
        kw.get("order_quantity"),
        None,
    )

    # Setup security validator via patch
    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        await engine._cycle()

    # Verify order was clamped to position size (1.0), not the calculated 2.0
    engine._order_submitter.submit_order.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order.call_args[1]
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["side"] == OrderSide.SELL
    assert call_kwargs["order_type"] == OrderType.MARKET
    assert call_kwargs["order_quantity"] == Decimal("1.0")


@pytest.mark.asyncio
async def test_reduce_only_blocks_new_position_on_empty_symbol(engine):
    """Test that reduce-only mode blocks orders for symbols with no position."""
    # Setup: no position, strategy wants to buy
    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []  # No positions

    # Enable reduce-only mode
    engine.context.risk_manager._reduce_only_mode = True
    engine.context.risk_manager._daily_pnl_triggered = False
    engine.context.risk_manager.check_order.return_value = False  # Block non-reduce orders
    engine.context.risk_manager.pre_trade_validate.return_value.is_valid = True

    # Setup security validator via patch
    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        await engine._cycle()

    # Verify order was NOT placed (blocked by reduce-only)
    engine.context.broker.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_mark_staleness_seeded_from_rest_fetch(engine):
    """Test that REST price fetch seeds mark staleness timestamp."""
    # Replace last_mark_update with a real dict to test seeding
    engine.context.risk_manager.last_mark_update = {}

    # Initial state: no mark update recorded
    assert "BTC-USD" not in engine.context.risk_manager.last_mark_update

    engine.strategy.decide.return_value = Decision(Action.HOLD, "test")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    await engine._cycle()

    # After cycle, mark update should be seeded
    assert "BTC-USD" in engine.context.risk_manager.last_mark_update
    assert engine.context.risk_manager.last_mark_update["BTC-USD"] > 0


@pytest.mark.asyncio
async def test_exchange_rules_blocks_small_order(engine):
    """Test that exchange rules guard blocks orders below min size."""
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.001")  # Very small
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("100"), available=Decimal("100"))
    ]
    engine.context.broker.list_positions.return_value = []

    # Mock the order validator to raise ValidationError for small orders
    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.side_effect = ValidationError(
        "Order size 0.00002 below minimum 0.0001"
    )

    # Mock state collector
    engine._state_collector = MagicMock()
    from gpt_trader.core import MarketType

    engine._state_collector.require_product.return_value = Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.0001"),
        step_size=Decimal("0.00001"),
        min_notional=Decimal("1"),
        price_increment=Decimal("0.01"),
        leverage_max=None,
    )

    # Mock submitter
    engine._order_submitter = MagicMock()

    engine.context.risk_manager.pre_trade_validate.return_value.is_valid = True

    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        await engine._cycle()

    # Order should NOT be placed
    engine.context.broker.place_order.assert_not_called()

    # Rejection should be recorded
    engine._order_submitter.record_rejection.assert_called_once()


@pytest.mark.asyncio
async def test_slippage_guard_blocks_order(engine):
    """Test that slippage guard blocks orders with excessive expected slippage."""
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    # Mock the order validator - exchange rules pass, slippage fails
    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.return_value = (
        Decimal("0.02"),
        None,
    )
    engine._order_validator.enforce_slippage_guard.side_effect = ValidationError(
        "Expected slippage 150 bps exceeds guard 50"
    )

    # Mock state collector
    engine._state_collector = MagicMock()
    from gpt_trader.core import MarketType

    engine._state_collector.require_product.return_value = Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.0001"),
        step_size=Decimal("0.00001"),
        min_notional=Decimal("1"),
        price_increment=Decimal("0.01"),
        leverage_max=None,
    )

    # Mock submitter
    engine._order_submitter = MagicMock()

    engine.context.risk_manager.pre_trade_validate.return_value.is_valid = True

    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        await engine._cycle()

    # Order should NOT be placed
    engine.context.broker.place_order.assert_not_called()

    # Rejection should be recorded
    engine._order_submitter.record_rejection.assert_called_once()


# =============================================================================
# Runtime Guard Sweep Tests
# =============================================================================


@pytest.mark.asyncio
async def test_runtime_guard_sweep_calls_guard_manager(engine, monkeypatch):
    """Test that runtime guard sweep calls GuardManager.run_runtime_guards."""
    engine._guard_manager = MagicMock()
    engine.running = True

    # Make asyncio.sleep raise CancelledError to exit after 1 iteration
    async def _sleep(_):
        engine.running = False  # Stop the loop after first iteration
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()

    engine._guard_manager.run_runtime_guards.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_guard_sweep_skips_without_guard_manager(engine, monkeypatch):
    """Test that runtime guard sweep handles missing guard manager gracefully."""
    engine._guard_manager = None
    engine.running = True

    # Make asyncio.sleep raise CancelledError to exit
    async def _sleep(_):
        engine.running = False
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    # Should not raise any exception besides CancelledError
    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()


@pytest.mark.asyncio
async def test_runtime_guard_sweep_uses_config_interval(engine, monkeypatch):
    """Test that runtime guard sweep uses configured interval."""
    engine._guard_manager = MagicMock()
    engine.running = True
    engine.context.config.runtime_guard_interval = 30  # Custom interval

    sleep_intervals = []

    async def _sleep(interval):
        sleep_intervals.append(interval)
        engine.running = False
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()

    assert sleep_intervals == [30]


@pytest.mark.asyncio
async def test_runtime_guard_sweep_handles_exceptions(engine, monkeypatch):
    """Test that runtime guard sweep continues after generic exceptions."""
    engine._guard_manager = MagicMock()
    engine._guard_manager.run_runtime_guards.side_effect = RuntimeError("test error")
    engine.running = True

    call_count = 0

    async def _sleep(_):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            engine.running = False
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    # Should not raise - exception is caught and logged
    with pytest.raises(asyncio.CancelledError):
        await engine._runtime_guard_sweep()

    # Guard manager was called twice before cancellation
    assert engine._guard_manager.run_runtime_guards.call_count == 2


# --- Health Check Runner Integration Tests ---


def test_health_check_runner_initialized(engine):
    """Test that health check runner is initialized with engine."""
    from gpt_trader.monitoring.health_checks import HealthCheckRunner

    assert hasattr(engine, "_health_check_runner")
    assert isinstance(engine._health_check_runner, HealthCheckRunner)
    # Verify dependencies were wired
    assert engine._health_check_runner._broker is engine.context.broker
    assert engine._health_check_runner._degradation_state is engine._degradation
    assert engine._health_check_runner._risk_manager is engine.context.risk_manager


@pytest.mark.asyncio
async def test_health_check_runner_started_and_stopped(engine, monkeypatch):
    """Test that health check runner starts/stops with engine lifecycle."""
    from unittest.mock import AsyncMock

    # Track start/stop calls
    start_mock = AsyncMock()
    stop_mock = AsyncMock()

    monkeypatch.setattr(engine._health_check_runner, "start", start_mock)
    monkeypatch.setattr(engine._health_check_runner, "stop", stop_mock)

    # Mock all other services to avoid running real loops
    engine._heartbeat.start = AsyncMock(return_value=None)
    engine._status_reporter.start = AsyncMock(return_value=None)
    engine._system_maintenance.start_prune_loop = AsyncMock(
        return_value=asyncio.create_task(asyncio.sleep(0))
    )
    engine._heartbeat.stop = AsyncMock()
    engine._status_reporter.stop = AsyncMock()
    engine._system_maintenance.stop = AsyncMock()

    # Prevent actual trading loop from running
    engine.running = False

    # Mock _run_loop to return immediately
    async def mock_run_loop():
        pass

    monkeypatch.setattr(engine, "_run_loop", mock_run_loop)

    # Mock WS health monitoring
    async def mock_monitor_ws_health():
        pass

    monkeypatch.setattr(engine, "_monitor_ws_health", mock_monitor_ws_health)

    # Disable streaming
    monkeypatch.setattr(engine, "_should_enable_streaming", lambda: False)

    # Start background tasks
    await engine.start_background_tasks()
    start_mock.assert_called_once()

    # Shutdown
    await engine.shutdown()
    stop_mock.assert_called_once()
