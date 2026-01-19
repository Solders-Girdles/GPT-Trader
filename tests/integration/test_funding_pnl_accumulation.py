"""Integration tests for funding accumulation/settlement behavior."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.engine.bar_runner import FundingProcessor
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.core import OrderSide, OrderType
from tests.integration.funding_pnl_test_base import SYMBOL, FundingPnLTestBase

pytestmark = pytest.mark.integration


class TestFundingPnLAccumulation(FundingPnLTestBase):
    """Tests for funding accumulation across settlement windows."""

    def test_funding_accumulates_over_time(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """Funding accumulates correctly over multiple intervals."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time
        self.seed_market_data(broker, start_time)

        broker.place_order(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        funding_values = self.run_funding_hours_collect(
            broker, funding_processor, start_time, hours=18
        )

        final_funding = funding_values[-1]
        assert final_funding > Decimal("0"), "Long should pay funding over time"

        # First real settlement at hour 9, second at hour 17
        assert funding_values[8] > Decimal("0"), "Settlement at hour 9 should show funding"
        assert final_funding > funding_values[8], "Funding should continue accumulating"

    def test_no_position_no_funding(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """No funding is charged when there's no position."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time

        self.run_funding_hours(broker, funding_processor, start_time, hours=8)

        stats = broker.get_statistics()
        assert stats["funding_pnl"] == Decimal("0")
