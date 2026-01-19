"""Tests for `StateCollector.resolve_effective_price`."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from gpt_trader.core import Product
from gpt_trader.features.live_trade.execution.state_collection import StateCollector


class TestResolveEffectivePrice:
    """Tests for resolve_effective_price method."""

    def test_returns_provided_price(self, collector: StateCollector, mock_product: Product) -> None:
        """Test that provided price is returned directly."""
        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=Decimal("50000"),
            product=mock_product,
        )

        assert result == Decimal("50000")

    def test_uses_mark_price_for_market_orders(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test that mark price is used for market orders."""
        mock_broker.get_mark_price = MagicMock(return_value=Decimal("51000"))

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=mock_product,
        )

        assert result == Decimal("51000")

    def test_uses_mid_price_fallback(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test fallback to bid/ask mid-price."""
        del mock_broker.get_mark_price

        product = SimpleNamespace(
            symbol="BTC-PERP",
            bid_price=Decimal("49000"),
            ask_price=Decimal("51000"),
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("50000")  # (49000 + 51000) / 2

    def test_uses_broker_quote_fallback(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test fallback to broker quote."""
        mock_broker.get_mark_price = MagicMock(return_value=None)
        mock_broker.get_quote = MagicMock(return_value=SimpleNamespace(last=Decimal("52000")))

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=mock_product,
        )

        assert result == Decimal("52000")

    def test_uses_product_price_fallback(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test fallback to product price."""
        del mock_broker.get_mark_price
        del mock_broker.get_quote

        product = SimpleNamespace(
            symbol="BTC-PERP",
            price=Decimal("53000"),
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("53000")

    def test_uses_quote_increment_as_last_resort(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test last resort uses quote_increment * 100."""
        del mock_broker.get_mark_price
        del mock_broker.get_quote

        product = SimpleNamespace(
            symbol="BTC-PERP",
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("1")  # 0.01 * 100

    def test_handles_zero_price_as_none(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test that zero price is treated as None."""
        mock_broker.get_mark_price = MagicMock(return_value=Decimal("51000"))

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=Decimal("0"),
            product=mock_product,
        )

        # Should use mark price instead
        assert result == Decimal("51000")

    def test_handles_mark_price_exception(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test that mark price exceptions are handled."""
        mock_broker.get_mark_price = MagicMock(side_effect=RuntimeError("API error"))
        mock_broker.get_quote = MagicMock(return_value=SimpleNamespace(last=Decimal("52000")))

        product = SimpleNamespace(
            symbol="BTC-PERP",
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("52000")
