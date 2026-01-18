"""Tests for OrderFillModel price touch detection for limit orders."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel


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
