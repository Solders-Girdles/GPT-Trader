"""Tests for FillResult, volume threshold, slippage, and spread impact."""

from datetime import datetime, timezone
from decimal import Decimal

from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import FillResult, OrderFillModel
from gpt_trader.core import Candle, OrderSide


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
