"""Tests for FillResult."""

from datetime import datetime, timezone
from decimal import Decimal

from gpt_trader.backtesting.simulation.fill_model import FillResult


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
