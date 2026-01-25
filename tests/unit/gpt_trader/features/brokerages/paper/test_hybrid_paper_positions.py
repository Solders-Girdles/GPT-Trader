"""Tests for HybridPaperBroker spot positions/balances."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import Balance, OrderSide, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


@pytest.fixture
def broker(broker_factory) -> HybridPaperBroker:
    """Create broker fixture with deterministic math (no slippage/commission)."""
    return broker_factory(
        initial_equity=Decimal("10000"),
        slippage_bps=0,
        commission_bps=Decimal("0"),
    )


class TestHybridPaperBrokerPositionsBalances:
    def test_list_positions_empty(self, broker: HybridPaperBroker) -> None:
        assert broker.list_positions() == []

    def test_list_positions_refreshes_mark_and_clears_unrealized(
        self, broker: HybridPaperBroker
    ) -> None:
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("0"),
            unrealized_pnl=Decimal("123"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("51000")

        positions = broker.list_positions()

        assert positions[0].mark_price == Decimal("51000")
        assert positions[0].unrealized_pnl == Decimal("0")

    def test_list_balances_initially_usd_only(self, broker: HybridPaperBroker) -> None:
        balances = broker.list_balances()
        assert len(balances) == 1
        assert balances[0].asset == "USD"
        assert balances[0].total == Decimal("10000")

    def test_get_equity_cash_only(self, broker: HybridPaperBroker) -> None:
        assert broker.get_equity() == Decimal("10000")

    def test_get_equity_with_spot_holdings(self, broker: HybridPaperBroker) -> None:
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("5000"), available=Decimal("5000")
        )
        broker._balances["BTC"] = Balance(
            asset="BTC", total=Decimal("0.1"), available=Decimal("0.1")
        )
        broker._last_prices["BTC-USD"] = Decimal("51000")

        assert broker.get_equity() == Decimal("10100")


class TestHybridPaperBrokerLifecycle:
    def test_buy_sell_round_trip_returns_flat(self, broker: HybridPaperBroker) -> None:
        broker._last_prices["BTC-USD"] = Decimal("50000")

        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type="market",
            quantity=Decimal("0.1"),
        )

        assert broker._balances["USD"].total == Decimal("5000")
        assert broker._balances["BTC"].total == Decimal("0.1")
        assert broker._positions["BTC-USD"].quantity == Decimal("0.1")

        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.SELL,
            order_type="market",
            quantity=Decimal("0.1"),
        )

        assert broker._balances["BTC"].total == Decimal("0")
        assert broker._balances["USD"].total == Decimal("10000")
        assert broker.list_positions() == []
