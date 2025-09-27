"""
Unit tests for Coinbase perpetuals product specifications and quantization.

Tests side-aware price quantization, safe position sizing, and XRP 10-unit requirements.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import tempfile
import yaml

from bot_v2.features.brokerages.coinbase.specs import (
    ProductSpec,
    SpecsService,
    get_specs_service,
    validate_order,
    quantize_price_side_aware,
    quantize_size,
    calculate_safe_position_size,
    ValidationResult
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
            'min_size': '0.0001',
            'step_size': '0.0001',
            'price_increment': '0.1',
            'min_notional': '100',
            'max_size': '1000',
            'slippage_multiplier': '1.5',
            'safe_buffer': '0.2'
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'products': {
                    'BTC-PERP': {
                        'min_size': 0.0001,
                        'step_size': 0.0001,
                        'price_increment': 0.01,
                        'min_notional': 10.0
                    }
                }
            }
            yaml.dump(yaml_content, f)
            config_path = f.name
        
        try:
            service = SpecsService(config_path)
            assert 'BTC-PERP' in service.overrides
            assert service.overrides['BTC-PERP']['min_size'] == 0.0001
        finally:
            Path(config_path).unlink()
    
    def test_service_builds_spec_with_overrides(self):
        """Test spec building merges API data with overrides."""
        service = SpecsService()
        service.overrides = {
            'BTC-PERP': {
                'min_size': 0.0002,
                'slippage_multiplier': 2.0
            }
        }
        
        api_data = {
            'min_size': 0.001,  # Should be overridden
            'step_size': 0.001,
            'price_increment': 0.01
        }
        
        spec = service.build_spec('BTC-PERP', api_data)
        assert spec.min_size == Decimal("0.0002")  # Override applied
        assert spec.step_size == Decimal("0.001")  # API data retained
        assert spec.slippage_multiplier == Decimal("2.0")  # Override applied
    
    def test_service_caches_specs(self):
        """Test specs are cached after first build."""
        service = SpecsService()
        spec1 = service.build_spec('BTC-PERP')
        spec2 = service.build_spec('BTC-PERP')
        assert spec1 is spec2  # Same object reference


class TestQuantization:
    """Test price and size quantization functions."""
    
    def test_side_aware_price_quantization_buy(self):
        """Test BUY orders floor price to increment."""
        # BUY at 50123.45 with 0.01 increment -> 50123.45 (already aligned)
        price = quantize_price_side_aware(
            Decimal("50123.45"), Decimal("0.01"), "buy"
        )
        assert price == Decimal("50123.45")
        
        # BUY at 50123.456 with 0.01 increment -> 50123.45 (floor)
        price = quantize_price_side_aware(
            Decimal("50123.456"), Decimal("0.01"), "buy"
        )
        assert price == Decimal("50123.45")
        
        # BUY at 50123.454 with 0.01 increment -> 50123.45 (floor)
        price = quantize_price_side_aware(
            Decimal("50123.454"), Decimal("0.01"), "buy"
        )
        assert price == Decimal("50123.45")
    
    def test_side_aware_price_quantization_sell(self):
        """Test SELL orders ceil price to increment."""
        # SELL at 50123.45 with 0.01 increment -> 50123.45 (already aligned)
        price = quantize_price_side_aware(
            Decimal("50123.45"), Decimal("0.01"), "sell"
        )
        assert price == Decimal("50123.45")
        
        # SELL at 50123.456 with 0.01 increment -> 50123.46 (ceil)
        price = quantize_price_side_aware(
            Decimal("50123.456"), Decimal("0.01"), "sell"
        )
        assert price == Decimal("50123.46")
        
        # SELL at 50123.454 with 0.01 increment -> 50123.46 (ceil)
        price = quantize_price_side_aware(
            Decimal("50123.454"), Decimal("0.01"), "sell"
        )
        assert price == Decimal("50123.46")
    
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
            qty=Decimal("0.005"),  # Below min_size of 0.01
            order_type="market",
            price=None
        )
        
        assert not result.ok
        assert result.reason == "min_size"
    
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
            qty=Decimal("0.001"),
            order_type="limit",
            price=Decimal("1000")  # Notional = 0.001 * 1000 = 1, below min of 10
        )
        
        assert not result.ok
        assert result.reason == "min_notional"
        assert result.adjusted_qty is not None  # Suggests corrected size
    
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
            qty=Decimal("0.1234"),  # Will be floored to 0.123
            order_type="limit",
            price=Decimal("50123.456")  # Will be floored to 50123.45
        )
        
        assert result.ok
        assert result.adjusted_qty == Decimal("0.123")
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
            intended_qty=Decimal("0.008"),  # Below min
            ref_price=Decimal("1000")
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
            intended_qty=Decimal("0.005"),  # Would be $50 notional
            ref_price=Decimal("10000")
        )
        
        # Should return size for at least $110 notional (100 * 1.1)
        expected_min = Decimal("110") / Decimal("10000")  # 0.011
        assert safe_size >= expected_min


class TestSlippageMultipliers:
    """Test slippage multiplier support."""
    
    def test_get_slippage_multiplier_from_spec(self):
        """Test retrieving slippage multiplier for product."""
        service = SpecsService()
        service.overrides = {
            'BTC-PERP': {'slippage_multiplier': 1.5},
            'ETH-PERP': {'slippage_multiplier': 2.0}
        }
        
        # Build specs and get multipliers
        service.build_spec('BTC-PERP')
        service.build_spec('ETH-PERP')
        
        assert service.get_slippage_multiplier('BTC-PERP') == 1.5
        assert service.get_slippage_multiplier('ETH-PERP') == 2.0
        
        # Default for unknown product
        assert service.get_slippage_multiplier('SOL-PERP') == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
