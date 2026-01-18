"""Unit tests for `ProductSpec` defaults and overrides."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.specs import ProductSpec


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
