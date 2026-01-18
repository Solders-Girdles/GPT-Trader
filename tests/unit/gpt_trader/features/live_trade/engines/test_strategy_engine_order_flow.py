"""Tests for TradingEngine decision -> order flow and mark seeding."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import Balance, OrderSide, OrderType
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.mark.asyncio
async def test_order_placed_with_dynamic_quantity(engine):
    """Test full flow from decision to order placement with calculated size."""
    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

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
