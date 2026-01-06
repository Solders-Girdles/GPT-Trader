"""
Unit tests for Coinbase perpetuals product specifications and quantization.

Tests side-aware price quantization, safe position sizing, and XRP 10-unit requirements.
"""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gpt_trader.features.brokerages.coinbase.specs import (
    ProductSpec,
    SpecsService,
    calculate_safe_position_size,
    quantize_price_side_aware,
    quantize_size,
    validate_order,
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


class TestSpecsService:
    """Test SpecsService loading and caching."""

    def test_service_loads_yaml_overrides(self):
        """Test service loads YAML config file."""
        # Create temp YAML config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {
                "products": {
                    "BTC-PERP": {
                        "min_size": 0.0001,
                        "step_size": 0.0001,
                        "price_increment": 0.01,
                        "min_notional": 10.0,
                    }
                }
            }
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            service = SpecsService(config_path)
            assert "BTC-PERP" in service.overrides
            assert service.overrides["BTC-PERP"]["min_size"] == 0.0001
        finally:
            Path(config_path).unlink()

    def test_service_handles_missing_config_file(self):
        """Test service handles missing config file gracefully."""
        service = SpecsService("/nonexistent/path.yaml")
        assert service.overrides == {}
        # Should still work with defaults
        spec = service.build_spec("BTC-PERP")
        assert spec.min_size == Decimal("0.001")  # Default value

    def test_service_handles_malformed_yaml(self):
        """Test service handles malformed YAML gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Malformed YAML
            config_path = f.name

        try:
            service = SpecsService(config_path)
            assert service.overrides == {}  # Should be empty on error
        finally:
            Path(config_path).unlink()

    def test_service_builds_spec_with_overrides(self):
        """Test spec building merges API data with overrides."""
        service = SpecsService()
        service.overrides = {"BTC-PERP": {"min_size": 0.0002, "slippage_multiplier": 2.0}}

        api_data = {
            "min_size": 0.001,  # Should be overridden
            "step_size": 0.001,
            "price_increment": 0.01,
        }

        spec = service.build_spec("BTC-PERP", api_data)
        assert spec.min_size == Decimal("0.0002")  # Override applied
        assert spec.step_size == Decimal("0.001")  # API data retained
        assert spec.slippage_multiplier == Decimal("2.0")  # Override applied

    def test_service_builds_spec_without_api_data(self):
        """Test spec building with no API data (uses defaults)."""
        service = SpecsService()
        spec = service.build_spec("BTC-PERP")
        assert spec.min_size == Decimal("0.001")
        assert spec.step_size == Decimal("0.001")
        assert spec.price_increment == Decimal("0.01")

    def test_service_caches_specs(self):
        """Test specs are cached after first build."""
        service = SpecsService()
        spec1 = service.build_spec("BTC-PERP")
        spec2 = service.build_spec("BTC-PERP")
        assert spec1 is spec2  # Same object reference

    def test_service_env_config_path(self):
        """Test service uses environment variable for config path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {"products": {"ETH-PERP": {"min_size": 0.002}}}
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            with patch.dict("os.environ", {"PERPS_SPECS_PATH": config_path}):
                service = SpecsService()
                assert "ETH-PERP" in service.overrides
                assert service.overrides["ETH-PERP"]["min_size"] == 0.002
        finally:
            Path(config_path).unlink()


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


class TestOrderValidation:
    """Test order validation with spec requirements."""

    def test_validate_order_size_below_minimum(self):
        """Test order rejected when size below minimum."""

        # Mock product with min_size
        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.01")
            price_increment = Decimal("0.01")
            min_notional = None

        result = validate_order(
            product=MockProduct(),
            side="buy",
            quantity=Decimal("0.005"),  # Below min_size of 0.01
            order_type="market",
            price=None,
        )

        assert result.ok
        assert result.adjusted_quantity == MockProduct().min_size

    def test_validate_order_notional_below_minimum(self):
        """Test order rejected when notional below minimum."""

        # Mock product with min_notional
        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.001")
            price_increment = Decimal("0.01")
            min_notional = Decimal("10")

        result = validate_order(
            product=MockProduct(),
            side="buy",
            quantity=Decimal("0.001"),
            order_type="limit",
            price=Decimal("1000"),  # Notional = 0.001 * 1000 = 1, below min of 10
        )

        assert not result.ok
        assert result.reason == "min_notional"
        assert result.adjusted_quantity == Decimal("0.01")  # Suggests corrected size

    def test_validate_order_adjusts_size_and_price(self):
        """Test validation adjusts size and price to increments."""

        # Mock product
        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.001")
            price_increment = Decimal("0.01")
            min_notional = None

        result = validate_order(
            product=MockProduct(),
            side="buy",
            quantity=Decimal("0.1234"),  # Will be floored to 0.123
            order_type="limit",
            price=Decimal("50123.456"),  # Will be floored to 50123.45
        )

        assert result.ok
        assert result.adjusted_quantity == Decimal("0.123")
        assert result.adjusted_price == Decimal("50123.45")


class TestSafePositionSizing:
    """Test safe position size calculation with buffers."""

    def test_calculate_safe_size_with_buffer(self):
        """Test safe sizing applies buffer to avoid violations."""

        # Mock product
        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.01")
            min_notional = Decimal("10")

        # Request size that meets min_size with buffer
        safe_size = calculate_safe_position_size(
            product=MockProduct(),
            side="buy",
            intended_quantity=Decimal("0.008"),  # Below min
            ref_price=Decimal("1000"),
        )

        # Should return at least min_size * 1.1
        assert safe_size >= Decimal("0.011")

    def test_calculate_safe_size_for_min_notional(self):
        """Test safe sizing meets minimum notional with buffer."""

        # Mock product
        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.001")
            min_notional = Decimal("100")

        # At $10,000 price, need 0.01 for $100 notional
        safe_size = calculate_safe_position_size(
            product=MockProduct(),
            side="buy",
            intended_quantity=Decimal("0.005"),  # Would be $50 notional
            ref_price=Decimal("10000"),
        )

        # Should return size for at least $110 notional (100 * 1.1)
        expected_min = Decimal("110") / Decimal("10000")  # 0.011
        assert safe_size >= expected_min

    def test_calculate_safe_size_already_safe(self):
        """Test safe sizing when input is already safe."""

        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.001")
            min_notional = Decimal("10")

        safe_size = calculate_safe_position_size(
            product=MockProduct(),
            side="buy",
            intended_quantity=Decimal("0.02"),  # Already safe
            ref_price=Decimal("1000"),
        )

        # Should return quantized version of input
        assert safe_size == Decimal("0.02")

    def test_calculate_safe_size_edge_cases(self):
        """Test safe sizing edge cases."""

        class MockProduct:
            step_size = Decimal("0.001")
            min_size = Decimal("0.01")
            min_notional = Decimal("10")

        # Zero price should not crash
        safe_size = calculate_safe_position_size(
            product=MockProduct(),
            side="buy",
            intended_quantity=Decimal("0.01"),
            ref_price=Decimal("0"),
        )
        assert safe_size == Decimal("0.011")  # Falls back to min_size * 1.1

        # Very small intended quantity
        safe_size = calculate_safe_position_size(
            product=MockProduct(),
            side="buy",
            intended_quantity=Decimal("0.0001"),
            ref_price=Decimal("1000"),
        )
        assert safe_size == Decimal("0.011")  # min_size * 1.1


class TestSlippageMultipliers:
    """Test slippage multiplier support."""

    def test_get_slippage_multiplier_from_spec(self):
        """Test retrieving slippage multiplier for product."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {"slippage_multiplier": 1.5},
            "ETH-PERP": {"slippage_multiplier": 2.0},
        }

        # Build specs and get multipliers
        service.build_spec("BTC-PERP")
        service.build_spec("ETH-PERP")

        assert service.get_slippage_multiplier("BTC-PERP") == 1.5
        assert service.get_slippage_multiplier("ETH-PERP") == 2.0

        # Default for unknown product
        assert service.get_slippage_multiplier("SOL-PERP") == 1.0


class TestSpecsServiceCalculateSafePositionSize:
    """Test SpecsService.calculate_safe_position_size method."""

    def test_calculate_safe_position_size_normal_case(self):
        """Test normal safe position size calculation."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 10.0,
                "safe_buffer": 0.1,
            }
        }

        size, reason = service.calculate_safe_position_size("BTC-PERP", 1000.0, 50000.0)
        # 1000 / 50000 = 0.02, buffer = 0.02 * 0.9 = 0.018, quantized to 0.018
        assert size == Decimal("0.018")
        assert reason == "within_limits"

    def test_calculate_safe_position_size_below_min_size(self):
        """Test safe position size when below minimum size."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.01,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 10.0,
                "safe_buffer": 0.1,
            }
        }

        size, reason = service.calculate_safe_position_size("BTC-PERP", 100.0, 50000.0)
        # 100 / 50000 = 0.002, below min_size 0.01, so adjust to min_size * (1 + buffer) = 0.011
        assert size == Decimal("0.011")
        assert reason == "adjusted_to_min_with_buffer"

    def test_calculate_safe_position_size_below_min_notional(self):
        """Test safe position size when below minimum notional."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 100.0,
                "safe_buffer": 0.1,
            }
        }

        size, reason = service.calculate_safe_position_size("BTC-PERP", 50.0, 50000.0)
        # 50 / 50000 = 0.001, notional = 0.001 * 50000 = 50 < 100, so adjust
        # Need (100 * 1.1) / 50000 = 110 / 50000 = 0.0022, quantized to 0.002
        assert size == Decimal("0.002")
        assert reason == "adjusted_for_min_notional"

    def test_calculate_safe_position_size_above_max_size(self):
        """Test safe position size when above maximum size."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 1.0,
                "min_notional": 10.0,
                "safe_buffer": 0.1,
            }
        }

        size, reason = service.calculate_safe_position_size("BTC-PERP", 100000.0, 50000.0)
        # 100000 / 50000 = 2.0, above max_size 1.0, so cap at max_size
        assert size == Decimal("1.0")
        assert reason == "capped_at_maximum"

    def test_calculate_safe_position_size_cannot_meet_min_notional(self):
        """Test safe position size when cannot meet minimum notional."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 0.001,  # Very small max_size
                "min_notional": 1000.0,  # Very high min_notional
                "safe_buffer": 0.1,
            }
        }

        size, reason = service.calculate_safe_position_size("BTC-PERP", 100.0, 50000.0)
        # Cannot meet min_notional even with max_size, so return 0
        assert size == Decimal("0")
        assert reason == "cannot_meet_min_notional"


class TestSpecsServiceValidateOrder:
    """Test SpecsService.validate_order method."""

    def test_validate_order_market_order(self):
        """Test order validation for market orders."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 10.0,
            }
        }

        result = service.validate_order("BTC-PERP", "buy", "market", 0.005, None)
        assert result["valid"] is True
        assert result["adjusted_size"] == 0.005
        assert result["adjusted_price"] is None

    def test_validate_order_limit_order_valid(self):
        """Test order validation for valid limit orders."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 10.0,
                "price_increment": 0.01,
            }
        }

        result = service.validate_order("BTC-PERP", "buy", "limit", 0.01, 50000.0)
        assert result["valid"] is True
        assert result["adjusted_size"] == 0.01
        assert result["adjusted_price"] == 50000.0

    def test_validate_order_size_too_small(self):
        """Test order validation when size is too small."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.01,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 10.0,
            }
        }

        result = service.validate_order("BTC-PERP", "buy", "market", 0.005, None)
        assert result["valid"] is False
        assert "size_below_minimum" in result["reasons"][0]

    def test_validate_order_size_too_large(self):
        """Test order validation when size is too large."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 1.0,
                "min_notional": 10.0,
            }
        }

        result = service.validate_order("BTC-PERP", "buy", "market", 2.0, None)
        assert result["valid"] is False
        assert "size_above_maximum" in result["reasons"][0]

    def test_validate_order_notional_too_small(self):
        """Test order validation when notional is too small."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 100.0,
                "price_increment": 0.01,
            }
        }

        result = service.validate_order("BTC-PERP", "buy", "limit", 0.001, 50.0)
        # 0.001 * 50.0 = 0.05 < 100.0
        assert result["valid"] is False
        assert "notional_below_minimum" in result["reasons"][0]

    def test_validate_order_price_quantization(self):
        """Test order validation with price quantization."""
        service = SpecsService()
        service.overrides = {
            "BTC-PERP": {
                "min_size": 0.001,
                "step_size": 0.001,
                "max_size": 100.0,
                "min_notional": 10.0,
                "price_increment": 0.01,
            }
        }

        result = service.validate_order("BTC-PERP", "buy", "limit", 0.01, 50000.123)
        assert result["valid"] is True
        assert result["adjusted_price"] == 50000.12  # Floored for buy
        assert "price_quantized_to_50000.12" in result["reasons"][0]


class TestEnforcePerpsRules:
    """Test enforce_perp_rules function from utilities module.

    These tests verify the enforce_perp_rules utility that combines
    quantization and validation for perps orders.
    """

    @staticmethod
    def make_product():
        """Helper to create test product."""
        from gpt_trader.core import MarketType, Product

        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            step_size=Decimal("0.001"),
            min_size=Decimal("0.001"),
            price_increment=Decimal("0.05"),  # 5c ticks to observe rounding
            min_notional=Decimal("10"),
            leverage_max=20,
        )

    @pytest.mark.perps
    def test_quantity_rounds_to_step_and_enforces_min_size(self):
        """Test enforce_perp_rules rounds quantity and enforces minimums."""
        from gpt_trader.features.brokerages.coinbase.utilities import (
            InvalidRequestError,
            enforce_perp_rules,
        )

        p = self.make_product()

        # Below min size should raise
        with pytest.raises(InvalidRequestError):
            enforce_perp_rules(p, quantity=Decimal("0.0005"), price=Decimal("50000"))

        # Rounds down to nearest step
        q, pr = enforce_perp_rules(p, quantity=Decimal("1.234567"), price=Decimal("50000.1234"))
        assert q == Decimal("1.234")
        assert pr == Decimal("50000.10")  # price increment 0.05 → rounds down

    @pytest.mark.perps
    def test_min_notional_enforced(self):
        """Test enforce_perp_rules enforces minimum notional value."""
        from gpt_trader.features.brokerages.coinbase.utilities import (
            InvalidRequestError,
            enforce_perp_rules,
        )

        p = self.make_product()

        # Small quantity with valid rounding but too small notional fails
        with pytest.raises(InvalidRequestError):
            enforce_perp_rules(p, quantity=Decimal("0.001"), price=Decimal("1000.00"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
