"""Tests for TradingEngine order guards (exchange rules, slippage, etc)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.security.security_validator as security_validator_module
from gpt_trader.core import Balance, Product
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


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
