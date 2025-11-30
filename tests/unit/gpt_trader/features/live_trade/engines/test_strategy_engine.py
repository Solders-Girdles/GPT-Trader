"""
Unit tests for Strategy Engine dynamic sizing and state tracking.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.brokerages.core.interfaces import (
    Balance,
    OrderSide,
    OrderType,
    Position,
    Product,
)
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.orchestration.configuration import BotConfig, BotRiskConfig


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
    return CoordinatorContext(config=config, broker=mock_broker, risk_manager=risk_manager)


@pytest.fixture
def engine(context, mock_strategy):
    with patch(
        "gpt_trader.features.live_trade.engines.strategy.create_strategy",
        return_value=mock_strategy,
    ):
        engine = TradingEngine(context)
        # Mock strategy explicitly in case init created a new one
        engine.strategy = mock_strategy
        return engine


@pytest.mark.asyncio
async def test_fetch_total_equity_success(engine):
    """Test successful equity fetch summing USD and USDC collateral."""
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("800")),
        Balance(asset="USDC", total=Decimal("500"), available=Decimal("200")),
        Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
    ]
    # No positions in this test case
    positions = {}

    equity = await engine._fetch_total_equity(positions)
    assert equity == Decimal("1000")  # 800 + 200


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
    engine.context.broker.place_order.assert_called_with(
        "BTC-USD", OrderSide.BUY, OrderType.MARKET, Decimal("0.02")
    )


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
