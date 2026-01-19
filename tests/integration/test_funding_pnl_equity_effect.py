"""Integration tests for funding impact on equity."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.engine.bar_runner import FundingProcessor
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.core import OrderSide, OrderType
from tests.integration.funding_pnl_test_base import SYMBOL, FundingPnLTestBase

pytestmark = pytest.mark.integration


class TestFundingPnLEquityImpact(FundingPnLTestBase):
    """Tests that funding payments are reflected in equity."""

    def test_funding_affects_final_equity(
        self,
        broker: SimulatedBroker,
        funding_processor: FundingProcessor,
    ) -> None:
        """Funding payments are reflected in final equity."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        broker._simulation_time = start_time
        self.seed_market_data(broker, start_time)

        _initial_equity = broker.get_equity()  # Stored for potential future assertions

        broker.place_order(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        equity_after_entry = broker.get_equity()

        # Run funding for 24 hours (3 full 8-hour cycles)
        self.run_funding_hours(broker, funding_processor, start_time, hours=24)

        final_equity = broker.get_equity()
        stats = broker.get_statistics()
        funding_pnl = stats["funding_pnl"]

        # funding_pnl is positive (tracking amount paid), so final_equity should be less
        # than equity_after_entry by approximately the funding amount
        actual_equity_change = final_equity - equity_after_entry

        assert funding_pnl > Decimal("0"), "Long should pay funding with positive rate"
        assert actual_equity_change < Decimal("0"), "Equity should decrease from funding costs"
