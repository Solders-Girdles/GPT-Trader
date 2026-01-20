"""Unit tests for ProductSpec and quantization helpers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.brokerages.coinbase.specs import (
    ProductSpec,
    quantize_price_side_aware,
    quantize_size,
)


class TestProductSpec:
    """Test ProductSpec initialization and fields."""

    def test_spec_initialization_with_defaults(self):
        """Test spec creates with default values."""
        spec = ProductSpec("BTC-PERP", {})
        assert spec.product_id == "BTC-PERP"
        assert spec.min_size == Decimal("0.001")
        assert spec.step_size == Decimal("0.001")
        assert spec.price_increment == Decimal("0.01")
        assert spec.min_notional == Decimal("10")
        assert spec.max_size == Decimal("1000000")
        assert spec.slippage_multiplier == Decimal("1.0")
        assert spec.safe_buffer == Decimal("0.1")

    def test_spec_initialization_with_overrides(self):
        """Test spec accepts custom values."""
        spec_data = {
            "min_size": "0.0001",
            "step_size": "0.0001",
            "price_increment": "0.1",
            "min_notional": "100",
            "max_size": "1000",
            "slippage_multiplier": "1.5",
            "safe_buffer": "0.2",
        }
        spec = ProductSpec("BTC-PERP", spec_data)
        assert spec.min_size == Decimal("0.0001")
        assert spec.step_size == Decimal("0.0001")
        assert spec.price_increment == Decimal("0.1")
        assert spec.min_notional == Decimal("100")
        assert spec.max_size == Decimal("1000")
        assert spec.slippage_multiplier == Decimal("1.5")
        assert spec.safe_buffer == Decimal("0.2")


class TestQuantization:
    """Test price and size quantization functions."""

    @pytest.mark.parametrize(
        "side,input_price,expected",
        [
            ("buy", Decimal("50123.45"), Decimal("50123.45")),
            ("buy", Decimal("50123.456"), Decimal("50123.45")),
            ("buy", Decimal("50123.454"), Decimal("50123.45")),
            ("sell", Decimal("50123.45"), Decimal("50123.45")),
            ("sell", Decimal("50123.456"), Decimal("50123.46")),
            ("sell", Decimal("50123.454"), Decimal("50123.46")),
            ("BUY", Decimal("50123.456"), Decimal("50123.45")),  # Case insensitive
            ("SELL", Decimal("50123.454"), Decimal("50123.46")),  # Case insensitive
        ],
    )
    def test_side_aware_price_quantization(self, side, input_price, expected):
        """Price quantization rounds in the safe direction for buy/sell."""
        price = quantize_price_side_aware(input_price, Decimal("0.01"), side)
        assert price == expected

    def test_price_quantization_zero_increment(self):
        """Test price quantization with zero increment returns input."""
        price = quantize_price_side_aware(Decimal("50123.45"), Decimal("0"), "buy")
        assert price == Decimal("50123.45")

    def test_price_quantization_invalid_side(self):
        """Test price quantization with invalid side defaults to buy behavior."""
        price = quantize_price_side_aware(Decimal("50123.456"), Decimal("0.01"), "invalid")
        assert price == Decimal("50123.45")  # Floors like buy

    def test_size_quantization_always_floors(self):
        """Test size is always floored to step size for safety."""
        # 0.1234 with 0.001 step -> 0.123 (floor)
        size = quantize_size(Decimal("0.1234"), Decimal("0.001"))
        assert size == Decimal("0.123")

        # 0.1239 with 0.001 step -> 0.123 (floor)
        size = quantize_size(Decimal("0.1239"), Decimal("0.001"))
        assert size == Decimal("0.123")

        # 0.123 with 0.001 step -> 0.123 (exact)
        size = quantize_size(Decimal("0.123"), Decimal("0.001"))
        assert size == Decimal("0.123")

    def test_size_quantization_zero_step(self):
        """Test size quantization with zero step returns input."""
        size = quantize_size(Decimal("0.1234"), Decimal("0"))
        assert size == Decimal("0.1234")

    def test_xrp_ten_unit_quantization(self):
        """Test XRP-PERP respects 10-unit step size."""
        # XRP requires 10-unit increments
        step = Decimal("10")

        # 125 XRP -> 120 (floor to 10s)
        size = quantize_size(Decimal("125"), step)
        assert size == Decimal("120")

        # 129 XRP -> 120 (floor to 10s)
        size = quantize_size(Decimal("129"), step)
        assert size == Decimal("120")

        # 130 XRP -> 130 (exact)
        size = quantize_size(Decimal("130"), step)
        assert size == Decimal("130")

        # 9 XRP -> 0 (below minimum)
        size = quantize_size(Decimal("9"), step)
        assert size == Decimal("0")
