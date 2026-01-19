"""Coinbase perp enforcement and quantization tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import InvalidRequestError as CoreInvalidRequestError
from gpt_trader.core import MarketType, Product
from gpt_trader.features.brokerages.coinbase.utilities import (
    enforce_perp_rules,
    quantize_to_increment,
)

pytestmark = pytest.mark.endpoints


class TestCoinbasePerpRulesAndQuantize:
    def make_perp_product(self) -> Product:
        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.00001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=20,
            contract_size=Decimal("1"),
        )

    def test_enforce_quantizes_quantity(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.123456789"))
        assert quantity == Decimal("0.12345")
        assert price is None

    def test_enforce_quantizes_price(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.01"), Decimal("50123.456"))
        assert quantity == Decimal("0.01")
        assert price == Decimal("50123.45")

    def test_enforce_rejects_below_min_size(self) -> None:
        product = self.make_perp_product()
        with pytest.raises(CoreInvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.0001"))
        assert "below minimum size" in str(exc_info.value)

    def test_enforce_rejects_below_min_notional(self) -> None:
        product = self.make_perp_product()
        with pytest.raises(CoreInvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.001"), Decimal("100"))
        assert "below minimum" in str(exc_info.value)

    def test_enforce_accepts_valid_notional(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("20000"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("20000")

    def test_enforce_handles_no_min_notional(self) -> None:
        product = self.make_perp_product()
        product.min_notional = None
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("1"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("1")

    def test_enforce_complex_quantization(self) -> None:
        product = Product(
            symbol="ETH-PERP",
            base_asset="ETH",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("50"),
            price_increment=Decimal("0.1"),
            leverage_max=15,
        )
        quantity, price = enforce_perp_rules(product, Decimal("0.123456"), Decimal("2345.67"))
        assert quantity == Decimal("0.123")
        assert price == Decimal("2345.6")
        assert quantity * price >= product.min_notional

    def test_quantize_basic(self) -> None:
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_floors_not_rounds(self) -> None:
        result = quantize_to_increment(Decimal("1.2389"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_handles_zero_increment(self) -> None:
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0"))
        assert result == Decimal("1.2345")
        result = quantize_to_increment(Decimal("1.2345"), None)
        assert result == Decimal("1.2345")

    def test_quantize_arbitrary_increments(self) -> None:
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.025"))
        assert result == Decimal("1.225")
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.005"))
        assert result == Decimal("1.235")
