"""
Fixtures for strategy_orchestrator tests.

This module contains common fixtures used across all strategy_orchestrator test modules.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, MarketType, Position, Product
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.spot_profile_service import SpotProfileService
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator


@pytest.fixture
def fake_perps_bot():
    """Create fake PerpsBot with async broker methods."""
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Profile.PROD
    bot.config.derivatives_enabled = True
    bot.config.short_ma = 10
    bot.config.long_ma = 30
    bot.config.target_leverage = 2
    bot.config.trailing_stop_pct = Decimal("0.02")
    bot.config.enable_shorts = True
    bot.config.perps_position_fraction = None
    bot.config.symbols = ["BTC-PERP", "ETH-PERP"]

    # Synchronous broker methods (for asyncio.to_thread compatibility)
    bot.broker = Mock()
    bot.broker.list_balances = Mock(return_value=[])
    bot.broker.list_positions = Mock(return_value=[])
    bot.broker.get_candles = Mock(return_value=[])

    # Risk manager with kill switch
    bot.risk_manager = Mock()
    bot.risk_manager.config = Mock()
    bot.risk_manager.config.kill_switch_enabled = False
    bot.risk_manager.check_volatility_circuit_breaker = Mock()
    bot.risk_manager.check_mark_staleness = Mock()

    # Runtime state
    state = PerpsBotRuntimeState(bot.config.symbols or [])
    bot.runtime_state = state
    bot.mark_windows = state.mark_windows
    bot.last_decisions = state.last_decisions

    # Strategy and execution
    bot.execute_decision = Mock()
    bot.get_product = Mock()

    return bot


@pytest.fixture
def fake_spot_profile_service():
    """Create fake SpotProfileService."""
    service = Mock(spec=SpotProfileService)
    service.load = Mock(return_value={})
    service.get = Mock(return_value=None)
    return service


@pytest.fixture
def async_orchestrator(fake_perps_bot, fake_spot_profile_service):
    """Create StrategyOrchestrator with async fakes."""
    return StrategyOrchestrator(bot=fake_perps_bot, spot_profile_service=fake_spot_profile_service)


@pytest.fixture
def test_balance():
    """Create test balance."""
    balance = Mock(spec=Balance)
    balance.asset = "USDC"
    balance.total = Decimal("10000")
    return balance


@pytest.fixture
def test_position():
    """Create test position."""
    position = Mock(spec=Position)
    position.symbol = "BTC-PERP"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("50000")
    return position


@pytest.fixture
def test_product():
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


# Additional fixtures for main test file refactoring


@pytest.fixture
def mock_bot():
    """Create mock PerpsBot instance (for main file tests)."""
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Profile.PROD  # Non-SPOT profile for perps
    bot.config.derivatives_enabled = True
    bot.config.short_ma = 10
    bot.config.long_ma = 30
    bot.config.target_leverage = 2
    bot.config.trailing_stop_pct = Decimal("0.02")
    bot.config.enable_shorts = True
    bot.config.perps_position_fraction = None
    bot.config.symbols = ["BTC-PERP", "ETH-PERP"]

    bot.broker = Mock()
    bot.risk_manager = Mock()
    bot.risk_manager.config = Mock()
    bot.risk_manager.config.kill_switch_enabled = False

    state = PerpsBotRuntimeState(bot.config.symbols or [])
    bot.runtime_state = state
    bot.mark_windows = state.mark_windows
    bot.last_decisions = state.last_decisions
    bot._symbol_strategies = state.symbol_strategies
    bot.execute_decision = Mock()
    bot.get_product = Mock()

    return bot


@pytest.fixture
def mock_spot_profile_service():
    """Create mock SpotProfileService (for main file tests)."""
    service = Mock()
    service.load = Mock(return_value={})
    service.get = Mock(return_value=None)
    return service


@pytest.fixture
def orchestrator(mock_bot, mock_spot_profile_service):
    """Create StrategyOrchestrator instance (for main file tests)."""
    return StrategyOrchestrator(bot=mock_bot, spot_profile_service=mock_spot_profile_service)
