from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpt_trader.app.config import BotConfig, BotRiskConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.core import Balance, Position
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
    config = MagicMock()
    config.position_fraction = None
    strategy.config = config
    return strategy


@pytest.fixture
def context(mock_broker):
    risk = BotRiskConfig(position_fraction=Decimal("0.1"))
    config = BotConfig(symbols=["BTC-USD"], interval=1, risk=risk)
    risk_manager = MagicMock()
    risk_manager._start_of_day_equity = Decimal("1000.0")
    risk_manager.check_mark_staleness.return_value = False
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
        engine.strategy = mock_strategy

        engine._state_collector = MagicMock()
        engine._state_collector.require_product.return_value = MagicMock()
        engine._state_collector.resolve_effective_price.return_value = Decimal("50000")
        engine._state_collector.build_positions_dict.return_value = {}

        engine._order_validator = MagicMock()
        engine._order_validator.validate_exchange_rules.return_value = (
            Decimal("0.02"),
            None,
        )
        engine._order_validator.enforce_slippage_guard.return_value = None
        engine._order_validator.ensure_mark_is_fresh.return_value = None
        engine._order_validator.run_pre_trade_validation.return_value = None
        engine._order_validator.maybe_preview_order_async = AsyncMock(return_value=None)
        engine._order_validator.maybe_preview_order.return_value = None
        engine._order_validator.finalize_reduce_only_flag.return_value = False

        engine._order_submitter = MagicMock()

        return engine
