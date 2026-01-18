"""Tests for TradingEngine equity and positions fetch helpers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import Balance, Position


@pytest.mark.asyncio
async def test_fetch_total_equity_success(engine):
    """Test successful equity fetch including non-USD assets valued in USD."""
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("800")),
        Balance(asset="USDC", total=Decimal("500"), available=Decimal("200")),
        Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
    ]
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

    await engine._cycle()
    engine.strategy.decide.assert_not_called()
