"""Tests for OrderFillModel market order fill simulation and price touch logic."""

from decimal import Decimal

from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle, OrderSide, OrderType


class TestMarketOrderFill:
    """Test market order fill simulation."""

    def test_buy_market_order_fill(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test buy market order fills at next bar open with slippage."""
        order = make_order(side=OrderSide.BUY, order_type=OrderType.MARKET)
        best_bid = Decimal("50100")
        best_ask = Decimal("50150")

        result = fill_model.fill_market_order(
            order=order,
            current_bar=current_bar,
            best_bid=best_bid,
            best_ask=best_ask,
            next_bar=next_bar,
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("1")
        assert result.is_maker is False  # Market orders are always taker
        assert result.fill_time == next_bar.ts
        # Fill price should be higher than next bar open (spread + slippage)
        assert result.fill_price > next_bar.open

    def test_sell_market_order_fill(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test sell market order fills with slippage working against seller."""
        order = make_order(side=OrderSide.SELL, order_type=OrderType.MARKET)
        best_bid = Decimal("50100")
        best_ask = Decimal("50150")

        result = fill_model.fill_market_order(
            order=order,
            current_bar=current_bar,
            best_bid=best_bid,
            best_ask=best_ask,
            next_bar=next_bar,
        )

        assert result.filled is True
        assert result.is_maker is False
        # Sell fill price should be lower than next bar open
        assert result.fill_price < next_bar.open

    def test_market_order_without_next_bar(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test market order uses current bar close when no next bar."""
        order = make_order(side=OrderSide.BUY, order_type=OrderType.MARKET)
        best_bid = Decimal("50100")
        best_ask = Decimal("50150")

        result = fill_model.fill_market_order(
            order=order,
            current_bar=current_bar,
            best_bid=best_bid,
            best_ask=best_ask,
            next_bar=None,
        )

        assert result.filled is True
        assert result.fill_time == current_bar.ts
        # Should be based on current_bar.close
        assert result.fill_price > current_bar.close

    def test_market_order_slippage_recorded(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test that slippage is recorded in basis points."""
        order = make_order(side=OrderSide.BUY, order_type=OrderType.MARKET)
        best_bid = Decimal("50100")
        best_ask = Decimal("50150")

        result = fill_model.fill_market_order(
            order=order,
            current_bar=current_bar,
            best_bid=best_bid,
            best_ask=best_ask,
            next_bar=next_bar,
        )

        assert result.slippage_bps is not None
        assert result.slippage_bps >= Decimal("0")


class TestPriceTouchLogic:
    """Test price touch detection for limit orders."""

    def test_buy_limit_touched_at_exact_low(self) -> None:
        """Test buy limit touched when limit equals bar low."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("100"),
            is_buy=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_buy_limit_touched_above_low(self) -> None:
        """Test buy limit touched when limit is above bar low."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("105"),
            is_buy=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_buy_limit_not_touched_below_low(self) -> None:
        """Test buy limit not touched when limit is below bar low."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("99"),
            is_buy=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is False

    def test_sell_limit_touched_at_exact_high(self) -> None:
        """Test sell limit touched when limit equals bar high."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("110"),
            is_buy=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_sell_limit_touched_below_high(self) -> None:
        """Test sell limit touched when limit is below bar high."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("105"),
            is_buy=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_sell_limit_not_touched_above_high(self) -> None:
        """Test sell limit not touched when limit is above bar high."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("115"),
            is_buy=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is False
