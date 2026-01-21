"""Tests for TradingEngine order flow, guards, and quantity calculations."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.security.security_validator as security_validator_module
from gpt_trader.core import Balance, OrderSide, OrderType, Product
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.mark.asyncio
async def test_order_placed_with_dynamic_quantity(engine, monkeypatch: pytest.MonkeyPatch):
    """Test full flow from decision to order placement with calculated size."""
    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validator_module, "get_validator", lambda: mock_validator)

    await engine._cycle()

    engine._order_submitter.submit_order.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order.call_args[1]
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["side"] == OrderSide.BUY
    assert call_kwargs["order_type"] == OrderType.MARKET
    assert call_kwargs["order_quantity"] == Decimal("0.02")


@pytest.mark.asyncio
async def test_mark_staleness_seeded_from_rest_fetch(engine):
    """Test that REST price fetch seeds mark staleness timestamp."""
    engine.context.risk_manager.last_mark_update = {}

    assert "BTC-USD" not in engine.context.risk_manager.last_mark_update

    engine.strategy.decide.return_value = Decision(Action.HOLD, "test")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    await engine._cycle()

    assert "BTC-USD" in engine.context.risk_manager.last_mark_update
    assert engine.context.risk_manager.last_mark_update["BTC-USD"] > 0


@pytest.mark.asyncio
async def test_exchange_rules_blocks_small_order(engine, monkeypatch: pytest.MonkeyPatch):
    """Test that exchange rules guard blocks orders below min size."""
    from gpt_trader.core import MarketType
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.001")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("100"), available=Decimal("100"))
    ]
    engine.context.broker.list_positions.return_value = []

    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.side_effect = ValidationError(
        "Order size 0.00002 below minimum 0.0001"
    )

    engine._state_collector = MagicMock()
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

    engine._order_submitter = MagicMock()

    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validator_module, "get_validator", lambda: mock_validator)

    await engine._cycle()

    engine.context.broker.place_order.assert_not_called()
    engine._order_submitter.record_rejection.assert_called_once()


@pytest.mark.asyncio
async def test_slippage_guard_blocks_order(engine, monkeypatch: pytest.MonkeyPatch):
    """Test that slippage guard blocks orders with excessive expected slippage."""
    from gpt_trader.core import MarketType
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.return_value = (
        Decimal("0.02"),
        None,
    )
    engine._order_validator.enforce_slippage_guard.side_effect = ValidationError(
        "Expected slippage 150 bps exceeds guard 50"
    )

    engine._state_collector = MagicMock()
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

    engine._order_submitter = MagicMock()

    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validator_module, "get_validator", lambda: mock_validator)

    await engine._cycle()

    engine.context.broker.place_order.assert_not_called()
    engine._order_submitter.record_rejection.assert_called_once()


def test_calculate_order_quantity_with_strategy_config(engine):
    """Test quantity calculation uses strategy config if set."""
    engine.strategy.config.position_fraction = Decimal("0.5")
    equity = Decimal("10000")
    price = Decimal("50000")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, None)
    assert quantity == Decimal("0.1")


def test_calculate_order_quantity_fallback_to_bot_config(engine):
    """Test quantity calculation falls back to bot config."""
    engine.strategy.config.position_fraction = None
    engine.context.config.perps_position_fraction = Decimal("0.2")

    equity = Decimal("10000")
    price = Decimal("50000")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, None)
    assert quantity == Decimal("0.04")


def test_calculate_order_quantity_min_size(engine):
    """Test quantity respects product min size."""
    engine.strategy.config.position_fraction = Decimal("0.1")
    equity = Decimal("100")
    price = Decimal("50000")

    product = MagicMock(spec=Product)
    product.min_size = Decimal("0.001")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, product)
    assert quantity == Decimal("0")
