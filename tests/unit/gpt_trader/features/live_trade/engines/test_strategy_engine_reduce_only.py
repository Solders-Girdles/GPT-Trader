"""Tests for TradingEngine reduce-only behavior."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import Balance, OrderSide, OrderType, Position
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.mark.asyncio
async def test_reduce_only_clamps_quantity_to_prevent_position_flip(engine):
    """Test that reduce-only mode clamps order quantity to prevent position flip."""
    engine.strategy.decide.return_value = Decision(Action.SELL, "test")
    engine.strategy.config.position_fraction = Decimal("0.2")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("500000"), available=Decimal("500000"))
    ]
    engine.context.broker.list_positions.return_value = [
        Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("10000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    ]

    engine.context.risk_manager._reduce_only_mode = True
    engine.context.risk_manager._daily_pnl_triggered = False
    engine.context.risk_manager.check_order.return_value = True

    engine._order_validator.validate_exchange_rules.side_effect = lambda **kw: (
        kw.get("order_quantity"),
        None,
    )

    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        await engine._cycle()

    engine._order_submitter.submit_order.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order.call_args[1]
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["side"] == OrderSide.SELL
    assert call_kwargs["order_type"] == OrderType.MARKET
    assert call_kwargs["order_quantity"] == Decimal("1.0")


@pytest.mark.asyncio
async def test_reduce_only_blocks_new_position_on_empty_symbol(engine):
    """Test that reduce-only mode blocks orders for symbols with no position."""
    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    engine.context.risk_manager._reduce_only_mode = True
    engine.context.risk_manager._daily_pnl_triggered = False
    engine.context.risk_manager.check_order.return_value = False

    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        await engine._cycle()

    engine.context.broker.place_order.assert_not_called()
