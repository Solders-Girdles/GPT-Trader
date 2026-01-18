"""Tests for OrderFillModel volume threshold checks."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel


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
