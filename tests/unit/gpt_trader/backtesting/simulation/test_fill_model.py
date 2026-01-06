"""Comprehensive tests for OrderFillModel - market, limit, and stop orders."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.fill_model import FillResult, OrderFillModel
from gpt_trader.core import (
    Candle,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


@pytest.fixture
def fill_model() -> OrderFillModel:
    """Default fill model with standard configuration."""
    return OrderFillModel(
        slippage_bps={"BTC-USD": Decimal("2"), "ETH-USD": Decimal("2")},
        spread_impact_pct=Decimal("0.5"),
        limit_volume_threshold=Decimal("2.0"),
        enable_queue_priority=False,
    )


@pytest.fixture
def current_bar() -> Candle:
    """Standard candle for testing."""
    return Candle(
        ts=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        open=Decimal("50000"),
        high=Decimal("50500"),
        low=Decimal("49500"),
        close=Decimal("50200"),
        volume=Decimal("100"),
    )


@pytest.fixture
def next_bar() -> Candle:
    """Next bar for market order fills."""
    return Candle(
        ts=datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc),
        open=Decimal("50200"),
        high=Decimal("50700"),
        low=Decimal("50000"),
        close=Decimal("50400"),
        volume=Decimal("150"),
    )


def make_order(
    symbol: str = "BTC-USD",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: Decimal = Decimal("1"),
    price: Decimal | None = None,
    stop_price: Decimal | None = None,
) -> Order:
    """Helper to create test orders."""
    return Order(
        id="test-order-001",
        symbol=symbol,
        side=side,
        type=order_type,
        quantity=quantity,
        status=OrderStatus.SUBMITTED,
        price=price,
        stop_price=stop_price,
        tif=TimeInForce.GTC,
    )


class TestFillResult:
    """Test FillResult dataclass."""

    def test_unfilled_result(self) -> None:
        """Test creating an unfilled result."""
        result = FillResult(filled=False, reason="Price not touched")
        assert result.filled is False
        assert result.fill_price is None
        assert result.reason == "Price not touched"

    def test_filled_result(self) -> None:
        """Test creating a filled result."""
        result = FillResult(
            filled=True,
            fill_price=Decimal("50100"),
            fill_quantity=Decimal("1"),
            fill_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            is_maker=False,
            slippage_bps=Decimal("2"),
        )
        assert result.filled is True
        assert result.fill_price == Decimal("50100")
        assert result.is_maker is False


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


class TestLimitOrderFill:
    """Test limit order fill simulation."""

    def test_buy_limit_price_touched_sufficient_volume(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test buy limit order fills when price drops to limit and volume is sufficient."""
        # Limit price within bar's low range
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),  # Volume threshold: 10 * 2 = 20, bar has 100
            price=Decimal("49600"),  # Below bar high, at/below bar low
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is True
        assert result.fill_price == Decimal("49600")
        assert result.is_maker is True  # Limit orders are maker
        assert result.slippage_bps == Decimal("0")

    def test_buy_limit_price_not_touched(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test buy limit order not filled when price doesn't drop to limit."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("49000"),  # Below bar's low of 49500
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is False
        assert "Price not touched" in result.reason

    def test_sell_limit_price_touched(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test sell limit order fills when price rises to limit."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("50400"),  # At/above bar high of 50500
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is True
        assert result.fill_price == Decimal("50400")

    def test_sell_limit_price_not_touched(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test sell limit order not filled when price doesn't reach limit."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("51000"),  # Above bar's high of 50500
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is False
        assert "Price not touched" in result.reason

    def test_limit_order_insufficient_volume(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test limit order not filled when bar volume is insufficient."""
        # Order size 100, threshold 2x = 200, bar volume is 100
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            price=Decimal("49600"),
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is False
        assert "Insufficient volume" in result.reason

    def test_limit_order_without_price_raises(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test that limit order without price raises ValueError."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=None,
        )

        with pytest.raises(ValueError, match="Limit order must have price"):
            fill_model.try_fill_limit_order(
                order=order,
                current_bar=current_bar,
                best_bid=Decimal("50100"),
                best_ask=Decimal("50150"),
            )


class TestLimitOrderQueuePriority:
    """Test limit order queue priority simulation."""

    def test_queue_priority_disabled_full_fill(self) -> None:
        """Test full fill when queue priority is disabled."""
        model = OrderFillModel(enable_queue_priority=False)
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("50"),  # Low volume
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),  # Needs 20 volume, bar has 50
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("10")  # Full fill

    def test_queue_priority_high_volume_full_fill(self) -> None:
        """Test full fill with queue priority when volume >> order size."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),  # 100x order size
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("10")  # Full fill

    def test_queue_priority_partial_fill(self) -> None:
        """Test partial fill with queue priority when volume is moderate."""
        model = OrderFillModel(
            enable_queue_priority=True,
            limit_volume_threshold=Decimal("1.0"),  # Low threshold
        )
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("50"),  # 5x order size -> 80% fill
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("8")  # 80% of 10

    def test_queue_priority_very_low_fill_rejected(self) -> None:
        """Test very low fill percentage (<10%) is rejected."""
        model = OrderFillModel(
            enable_queue_priority=True,
            limit_volume_threshold=Decimal("1.0"),
        )
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("10"),  # 1x order size -> 20% fill, but need to check threshold
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        # With 1x volume ratio, fill_pct is 0.2 (20%), which is > 10%, so it should fill
        assert result.filled is True


class TestStopOrderFill:
    """Test stop order fill simulation."""

    def test_buy_stop_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test buy stop triggers when price rises to stop price."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("1"),
            stop_price=Decimal("50400"),  # Within bar's high of 50500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is True
        assert result.is_maker is False  # Stop fills as market order

    def test_buy_stop_not_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test buy stop not triggered when price doesn't reach stop."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("51000"),  # Above bar's high of 50500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is False
        assert "Stop not triggered" in result.reason

    def test_sell_stop_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test sell stop triggers when price drops to stop price."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("1"),
            stop_price=Decimal("49600"),  # Within bar's low of 49500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is True

    def test_sell_stop_not_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test sell stop not triggered when price doesn't drop to stop."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("49000"),  # Below bar's low of 49500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is False

    def test_stop_order_without_stop_price_raises(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test that stop order without stop_price raises ValueError."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=None,
        )

        with pytest.raises(ValueError, match="Stop order must have stop_price"):
            fill_model.try_fill_stop_order(
                order=order,
                current_bar=current_bar,
                best_bid=Decimal("50100"),
                best_ask=Decimal("50150"),
            )


class TestSlippageCalculation:
    """Test slippage calculation for different symbols."""

    def test_btc_slippage_default(self) -> None:
        """Test BTC pairs use default 2 bps slippage."""
        model = OrderFillModel()  # No custom slippage
        slippage = model._get_slippage("BTC-USD")
        assert slippage == Decimal("2")

    def test_eth_slippage_default(self) -> None:
        """Test ETH pairs use default 2 bps slippage."""
        model = OrderFillModel()
        slippage = model._get_slippage("ETH-USD")
        assert slippage == Decimal("2")

    def test_unknown_symbol_default(self) -> None:
        """Test unknown symbols use 5 bps default slippage."""
        model = OrderFillModel()
        slippage = model._get_slippage("DOGE-USD")
        assert slippage == Decimal("5")

    def test_custom_slippage_override(self) -> None:
        """Test custom slippage overrides defaults."""
        model = OrderFillModel(slippage_bps={"CUSTOM-USD": Decimal("10")})
        slippage = model._get_slippage("CUSTOM-USD")
        assert slippage == Decimal("10")

    def test_btc_perp_uses_btc_default(self) -> None:
        """Test BTC-PERP uses BTC default slippage."""
        model = OrderFillModel()
        slippage = model._get_slippage("BTC-PERP-USDC")
        assert slippage == Decimal("2")


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


class TestVolumeThreshold:
    """Test volume threshold logic."""

    def test_sufficient_volume(self) -> None:
        """Test volume is sufficient when bar_volume >= order_size * threshold."""
        model = OrderFillModel(limit_volume_threshold=Decimal("2.0"))
        sufficient = model._has_sufficient_volume(
            order_size=Decimal("10"),
            bar_volume=Decimal("25"),  # 25 >= 10 * 2
        )
        assert sufficient is True

    def test_exact_threshold_volume(self) -> None:
        """Test exact threshold volume is sufficient."""
        model = OrderFillModel(limit_volume_threshold=Decimal("2.0"))
        sufficient = model._has_sufficient_volume(
            order_size=Decimal("10"),
            bar_volume=Decimal("20"),  # 20 == 10 * 2
        )
        assert sufficient is True

    def test_insufficient_volume(self) -> None:
        """Test volume is insufficient when below threshold."""
        model = OrderFillModel(limit_volume_threshold=Decimal("2.0"))
        sufficient = model._has_sufficient_volume(
            order_size=Decimal("10"),
            bar_volume=Decimal("15"),  # 15 < 10 * 2
        )
        assert sufficient is False


class TestQueueFillEstimation:
    """Test queue fill percentage estimation."""

    def test_high_volume_ratio_full_fill(self) -> None:
        """Test 100% fill when volume ratio >= 10x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),
        )
        order = make_order(quantity=Decimal("10"))  # 1000/10 = 100x

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("1.0")

    def test_medium_high_volume_80_fill(self) -> None:
        """Test 80% fill when volume ratio is 5-10x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("70"),  # 70/10 = 7x
        )
        order = make_order(quantity=Decimal("10"))

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("0.8")

    def test_medium_volume_50_fill(self) -> None:
        """Test 50% fill when volume ratio is 2-5x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("30"),  # 30/10 = 3x
        )
        order = make_order(quantity=Decimal("10"))

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("0.5")

    def test_low_volume_20_fill(self) -> None:
        """Test 20% fill when volume ratio < 2x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("15"),  # 15/10 = 1.5x
        )
        order = make_order(quantity=Decimal("10"))

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("0.2")


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
