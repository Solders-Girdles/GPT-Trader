"""Tests for OrderFillModel spread impact behavior."""

from datetime import datetime, timezone
from decimal import Decimal

from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle, OrderSide


class TestSpreadImpact:
    """Test spread impact on fill prices."""

    def test_zero_spread_impact(self) -> None:
        """Test no spread impact when spread_impact_pct is 0."""
        model = OrderFillModel(spread_impact_pct=Decimal("0"))
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),
        )
        order = make_order(side=OrderSide.BUY)

        result = model.fill_market_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("102"),  # 2 spread
            next_bar=None,
        )

        # With 0 spread impact, only slippage applies
        assert result.filled is True

    def test_full_spread_impact(self) -> None:
        """Test full spread impact when spread_impact_pct is 1."""
        model = OrderFillModel(spread_impact_pct=Decimal("1.0"), slippage_bps={})
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        )
        order = make_order(side=OrderSide.BUY, symbol="UNKNOWN")

        # With full spread impact, buy should pay half the spread more
        result = model.fill_market_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("99"),
            best_ask=Decimal("101"),  # 2 spread
            next_bar=None,
        )

        assert result.filled is True
        # Fill price should include spread impact + slippage
        assert result.fill_price > Decimal("100")
