"""Tests for OrderFillModel stop trigger detection."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel


class TestStopTriggerLogic:
    """Test stop trigger detection."""

    def test_buy_stop_triggered_at_exact_high(self) -> None:
        """Test buy stop triggers when stop equals bar high."""
        model = OrderFillModel()
        triggered = model._is_stop_triggered(
            stop_price=Decimal("110"),
            is_buy_stop=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert triggered is True

    def test_buy_stop_triggered_below_high(self) -> None:
        """Test buy stop triggers when stop is below bar high."""
        model = OrderFillModel()
        triggered = model._is_stop_triggered(
            stop_price=Decimal("105"),
            is_buy_stop=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert triggered is True

    def test_buy_stop_not_triggered_above_high(self) -> None:
        """Test buy stop not triggered when stop is above bar high."""
        model = OrderFillModel()
        triggered = model._is_stop_triggered(
            stop_price=Decimal("115"),
            is_buy_stop=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert triggered is False

    def test_sell_stop_triggered_at_exact_low(self) -> None:
        """Test sell stop triggers when stop equals bar low."""
        model = OrderFillModel()
        triggered = model._is_stop_triggered(
            stop_price=Decimal("100"),
            is_buy_stop=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert triggered is True

    def test_sell_stop_triggered_above_low(self) -> None:
        """Test sell stop triggers when stop is above bar low."""
        model = OrderFillModel()
        triggered = model._is_stop_triggered(
            stop_price=Decimal("105"),
            is_buy_stop=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert triggered is True

    def test_sell_stop_not_triggered_below_low(self) -> None:
        """Test sell stop not triggered when stop is below bar low."""
        model = OrderFillModel()
        triggered = model._is_stop_triggered(
            stop_price=Decimal("95"),
            is_buy_stop=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert triggered is False
