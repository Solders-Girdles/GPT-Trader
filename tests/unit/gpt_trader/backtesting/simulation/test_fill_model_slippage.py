"""Tests for OrderFillModel slippage calculations."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel


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
