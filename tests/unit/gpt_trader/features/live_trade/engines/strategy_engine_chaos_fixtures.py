"""Shared fixtures for TradingEngine chaos tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

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


def make_position(symbol: str = "BTC-USD", qty: str = "1.0", side: str = "long") -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        entry_price=Decimal("40000"),
        mark_price=Decimal("50000"),
        unrealized_pnl=Decimal("10000"),
        realized_pnl=Decimal("0"),
        side=side,
    )


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_ticker.return_value = {"price": "50000"}
    broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    broker.list_positions.return_value = [make_position(qty="0.5")]
    broker.get_resilience_status.return_value = None
    broker.get_market_snapshot.return_value = {"spread_bps": 10, "depth_l1": 10000}
    broker.place_order.return_value = "order-123"
    return broker


@pytest.fixture
def mock_risk_config():
    c = MagicMock()
    c.broker_outage_max_failures, c.broker_outage_cooldown_seconds = 3, 120
    c.mark_staleness_cooldown_seconds, c.mark_staleness_allow_reduce_only = 60, True
    c.slippage_failure_pause_after, c.slippage_pause_seconds = 3, 60
    c.validation_failure_cooldown_seconds, c.preview_failure_disable_after = 180, 3
    c.api_health_cooldown_seconds, c.api_error_rate_threshold, c.api_rate_limit_usage_threshold = (
        300,
        0.2,
        0.9,
    )
    # WS health config
    c.ws_health_interval_seconds = 1  # Fast for tests
    c.ws_message_stale_seconds = 15
    c.ws_heartbeat_stale_seconds = 30
    c.ws_reconnect_pause_seconds = 10
    c.kill_switch_enabled = False
    return c


@pytest.fixture
def context(mock_broker, mock_risk_config):
    risk = BotRiskConfig(position_fraction=Decimal("0.1"))
    config = BotConfig(symbols=["BTC-USD"], interval=1, risk=risk)
    rm = MagicMock()
    rm._start_of_day_equity, rm.check_mark_staleness.return_value = Decimal("10000.0"), False
    rm.is_reduce_only_mode.return_value, rm.config = False, mock_risk_config
    return CoordinatorContext(config=config, broker=mock_broker, risk_manager=rm)


@pytest.fixture
def application_container(context):
    """Set up application container for TradingEngine chaos tests."""
    container = ApplicationContainer(context.config)
    set_application_container(container)
    yield container
    clear_application_container()


@pytest.fixture
def mock_security_validator():
    v, r = MagicMock(), MagicMock()
    r.is_valid, r.errors = True, []
    v.validate_order_request.return_value = r
    with patch("gpt_trader.security.security_validator.get_validator", return_value=v):
        yield v


@pytest.fixture
def engine(context, mock_security_validator, application_container):
    strategy = MagicMock()
    strategy.decide.return_value, strategy.config.position_fraction = (
        Decision(Action.HOLD, "test"),
        Decimal("0.1"),
    )
    with patch(
        "gpt_trader.features.live_trade.engines.strategy.create_strategy", return_value=strategy
    ):
        eng = TradingEngine(context)
        eng._state_collector = MagicMock()
        eng._state_collector.require_product.return_value = MagicMock()
        eng._state_collector.resolve_effective_price.return_value = Decimal("50000")
        eng._state_collector.build_positions_dict.return_value = {}
        eng._order_validator = MagicMock()
        eng._order_validator.validate_exchange_rules.return_value = (Decimal("0.02"), None)
        for attr in [
            "enforce_slippage_guard",
            "ensure_mark_is_fresh",
            "run_pre_trade_validation",
            "maybe_preview_order",
        ]:
            setattr(eng._order_validator, attr, MagicMock(return_value=None))
        from unittest.mock import AsyncMock

        eng._order_validator.maybe_preview_order_async = AsyncMock(return_value=None)
        eng._order_validator.finalize_reduce_only_flag.return_value = False
        eng._order_validator.enable_order_preview = True
        eng._order_submitter = MagicMock()
        yield eng
