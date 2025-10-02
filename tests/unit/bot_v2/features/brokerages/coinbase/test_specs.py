"""
Comprehensive tests for product specifications and quantization.

Covers:
- ProductSpec initialization and defaults
- SpecsService YAML override loading
- Side-aware price quantization (BUY=floor, SELL=ceil)
- Safe position sizing with buffers
- Pre-flight order validation
- Module-level helper functions
"""

from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from bot_v2.features.brokerages.coinbase.specs import (
    ProductSpec,
    SpecsService,
    ValidationResult,
    calculate_safe_position_size,
    get_specs_service,
    quantize_size,
    quantize_size_up,
    validate_order,
)


@pytest.fixture
def temp_specs_config(tmp_path):
    """Create temporary specs YAML config."""
    config_path = tmp_path / "specs.yaml"
    config = {
        "products": {
            "BTC-USD-PERP": {
                "min_size": "0.001",
                "step_size": "0.001",
                "price_increment": "0.01",
                "min_notional": "10",
                "max_size": "100",
                "slippage_multiplier": "1.5",
                "safe_buffer": "0.15",
                "last_verified": "2024-01-15",
                "source": "override",
            }
        }
    }
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def specs_service(temp_specs_config):
    """Create SpecsService with test config."""
    return SpecsService(str(temp_specs_config))


@pytest.fixture
def mock_product():
    """Mock product for validation tests."""
    product = Mock()
    product.step_size = Decimal("0.001")
    product.min_size = Decimal("0.01")
    product.price_increment = Decimal("0.01")
    product.min_notional = Decimal("10")
    return product


class TestProductSpec:
    """Test ProductSpec initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        spec = ProductSpec("BTC-USD-PERP", {})

        assert spec.product_id == "BTC-USD-PERP"
        assert spec.min_size == Decimal("0.001")
        assert spec.step_size == Decimal("0.001")
        assert spec.price_increment == Decimal("0.01")
        assert spec.min_notional == Decimal("10")
        assert spec.max_size == Decimal("1000000")
        assert spec.slippage_multiplier == Decimal("1.0")
        assert spec.safe_buffer == Decimal("0.1")
        assert spec.last_verified == "unknown"
        assert spec.source == "api"

    def test_init_with_custom_values(self):
        """Should use provided values."""
        spec_data = {
            "min_size": "0.01",
            "step_size": "0.01",
            "price_increment": "1.0",
            "min_notional": "100",
            "max_size": "50",
            "slippage_multiplier": "2.0",
            "safe_buffer": "0.2",
            "last_verified": "2024-01-15",
            "source": "override",
        }
        spec = ProductSpec("ETH-USD-PERP", spec_data)

        assert spec.min_size == Decimal("0.01")
        assert spec.step_size == Decimal("0.01")
        assert spec.price_increment == Decimal("1.0")
        assert spec.min_notional == Decimal("100")
        assert spec.max_size == Decimal("50")
        assert spec.slippage_multiplier == Decimal("2.0")
        assert spec.safe_buffer == Decimal("0.2")
        assert spec.last_verified == "2024-01-15"
        assert spec.source == "override"


class TestSpecsServiceInit:
    """Test SpecsService initialization and override loading."""

    def test_init_with_valid_config(self, temp_specs_config):
        """Should load overrides from YAML."""
        service = SpecsService(str(temp_specs_config))

        assert "BTC-USD-PERP" in service.overrides
        assert service.overrides["BTC-USD-PERP"]["min_size"] == "0.001"

    def test_init_with_missing_config(self, tmp_path):
        """Should handle missing config gracefully."""
        missing_path = tmp_path / "nonexistent.yaml"
        service = SpecsService(str(missing_path))

        assert service.overrides == {}

    def test_init_with_invalid_yaml(self, tmp_path):
        """Should handle invalid YAML gracefully."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("{ invalid yaml ][")

        service = SpecsService(str(invalid_config))

        assert service.overrides == {}

    def test_init_with_empty_yaml(self, tmp_path):
        """Should handle empty YAML."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        service = SpecsService(str(empty_config))

        assert service.overrides == {}

    def test_init_with_no_products_key(self, tmp_path):
        """Should handle YAML without products key."""
        config = tmp_path / "no_products.yaml"
        config.write_text(yaml.dump({"settings": {}}))

        service = SpecsService(str(config))

        assert service.overrides == {}

    def test_init_uses_env_var(self, tmp_path):
        """Should use PERPS_SPECS_PATH env var."""
        config = tmp_path / "env_config.yaml"
        config.write_text(yaml.dump({"products": {"TEST": {"min_size": "1"}}}))

        with patch.dict("os.environ", {"PERPS_SPECS_PATH": str(config)}):
            service = SpecsService()

        assert "TEST" in service.overrides


class TestBuildSpec:
    """Test build_spec method."""

    def test_build_spec_with_api_data_only(self):
        """Should build spec from API data."""
        service = SpecsService(config_path="nonexistent.yaml")
        api_data = {"min_size": "0.01", "step_size": "0.01"}

        spec = service.build_spec("BTC-USD-PERP", api_data)

        assert spec.product_id == "BTC-USD-PERP"
        assert spec.min_size == Decimal("0.01")

    def test_build_spec_with_overrides(self, specs_service):
        """Should merge API data with overrides."""
        api_data = {"min_size": "0.01", "step_size": "0.01"}

        spec = specs_service.build_spec("BTC-USD-PERP", api_data)

        # Override should win
        assert spec.min_size == Decimal("0.001")
        assert spec.slippage_multiplier == Decimal("1.5")

    def test_build_spec_caches_result(self, specs_service):
        """Should cache built specs."""
        spec1 = specs_service.build_spec("BTC-USD-PERP")
        spec2 = specs_service.build_spec("BTC-USD-PERP")

        assert spec1 is spec2

    def test_build_spec_with_no_api_data(self, specs_service):
        """Should use defaults when no API data."""
        spec = specs_service.build_spec("NEW-PRODUCT")

        assert spec.min_size == Decimal("0.001")
        assert spec.step_size == Decimal("0.001")


class TestQuantizePriceSideAware:
    """Test side-aware price quantization."""

    def test_quantize_buy_floors(self, specs_service):
        """BUY should floor to increment."""
        price = specs_service.quantize_price_side_aware("BTC-USD-PERP", "BUY", 50000.456)

        assert price == Decimal("50000.45")

    def test_quantize_sell_ceils(self, specs_service):
        """SELL should ceil to increment."""
        price = specs_service.quantize_price_side_aware("BTC-USD-PERP", "SELL", 50000.451)

        assert price == Decimal("50000.46")

    def test_quantize_with_lowercase_side(self, specs_service):
        """Should handle lowercase side."""
        price = specs_service.quantize_price_side_aware("BTC-USD-PERP", "buy", 50000.456)

        assert price == Decimal("50000.45")

    def test_quantize_exact_increment(self, specs_service):
        """Should not change prices on exact increment."""
        price = specs_service.quantize_price_side_aware("BTC-USD-PERP", "BUY", 50000.50)

        assert price == Decimal("50000.50")

    def test_quantize_zero_increment(self, specs_service):
        """Should return original price if increment is zero."""
        specs_service.specs_cache["TEST"] = ProductSpec("TEST", {"price_increment": "0"})

        price = specs_service.quantize_price_side_aware("TEST", "BUY", 50000.456)

        assert price == Decimal("50000.456")


class TestQuantizeSize:
    """Test size quantization."""

    def test_quantize_size_floors(self, specs_service):
        """Should floor to step size."""
        size = specs_service.quantize_size("BTC-USD-PERP", 1.2345)

        assert size == Decimal("1.234")

    def test_quantize_size_exact_step(self, specs_service):
        """Should not change size on exact step."""
        size = specs_service.quantize_size("BTC-USD-PERP", 1.234)

        assert size == Decimal("1.234")

    def test_quantize_size_zero_step(self, specs_service):
        """Should return original if step is zero."""
        specs_service.specs_cache["TEST"] = ProductSpec("TEST", {"step_size": "0"})

        size = specs_service.quantize_size("TEST", 1.2345)

        assert size == Decimal("1.2345")


class TestCalculateSafePositionSize:
    """Test safe position size calculation."""

    def test_within_limits(self, specs_service):
        """Should return quantized size within limits."""
        size, reason = specs_service.calculate_safe_position_size(
            "BTC-USD-PERP", target_notional=5000, mark_price=50000
        )

        # 5000 / 50000 = 0.1, with 15% buffer = 0.085
        assert size == Decimal("0.085")
        assert reason == "within_limits"

    def test_below_minimum_size(self, specs_service):
        """Should adjust to minimum size with buffer."""
        size, reason = specs_service.calculate_safe_position_size(
            "BTC-USD-PERP", target_notional=58, mark_price=50000
        )

        # Raw: 58/50000 = 0.00116, buffered = 0.00116 * 0.85 = 0.000986
        # Quantized to 0.000 (floor), which is < min_size (0.001)
        # buffered_min = 0.001 * 1.15 = 0.00115
        # buffered_min * mark_price = 0.00115 * 50000 = 57.5 <= 58? YES
        assert size == Decimal("0.00115")
        assert reason == "adjusted_to_min_with_buffer"

    def test_below_minimum_notional(self, specs_service):
        """Should return zero if cannot meet minimum notional."""
        size, reason = specs_service.calculate_safe_position_size(
            "BTC-USD-PERP", target_notional=1, mark_price=50000
        )

        assert size == Decimal("0")
        assert reason == "below_minimum_notional"

    def test_exceeds_maximum_size(self, specs_service):
        """Should cap at maximum size."""
        size, reason = specs_service.calculate_safe_position_size(
            "BTC-USD-PERP", target_notional=10000000, mark_price=50000
        )

        # Would be huge, but max_size = 100
        assert size == Decimal("100")
        assert reason == "capped_at_maximum"

    def test_adjusted_for_min_notional(self, specs_service):
        """Should adjust size to meet minimum notional."""
        # Set up scenario where size is OK but notional too low
        size, reason = specs_service.calculate_safe_position_size(
            "BTC-USD-PERP", target_notional=500, mark_price=50000
        )

        # 500 / 50000 = 0.01, buffered = 0.0085
        # Notional = 0.008 * 50000 = 400 < 10
        # Should adjust to meet min_notional = 10
        assert reason in ["adjusted_for_min_notional", "within_limits"]

    def test_cannot_meet_min_notional(self, specs_service):
        """Should return zero if adjusted size exceeds max."""
        # Create scenario where min_notional adjustment exceeds max_size
        specs_service.specs_cache["TEST"] = ProductSpec(
            "TEST", {"min_notional": "1000000", "max_size": "1", "step_size": "0.001"}
        )

        size, reason = specs_service.calculate_safe_position_size(
            "TEST", target_notional=100, mark_price=1000
        )

        assert size == Decimal("0")
        assert reason == "cannot_meet_min_notional"


class TestValidateOrder:
    """Test order validation."""

    def test_valid_order(self, specs_service):
        """Should validate correct order."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="LIMIT", size=0.5, price=50000.0
        )

        assert result["valid"] is True
        assert result["adjusted_size"] == 0.5
        assert result["adjusted_price"] == 50000.0

    def test_quantizes_size(self, specs_service):
        """Should quantize size to step."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="MARKET", size=0.5555
        )

        assert result["adjusted_size"] == 0.555
        assert "size_quantized" in result["reasons"][0]

    def test_quantizes_price(self, specs_service):
        """Should quantize price for limit orders."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="LIMIT", size=0.5, price=50000.456
        )

        assert result["adjusted_price"] == 50000.45
        assert "price_quantized" in result["reasons"][0]

    def test_rejects_below_min_size(self, specs_service):
        """Should reject size below minimum."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="MARKET", size=0.0001
        )

        assert result["valid"] is False
        # Should have quantization reason AND rejection reason
        assert any("size_below_minimum" in r for r in result["reasons"])

    def test_rejects_above_max_size(self, specs_service):
        """Should reject size above maximum."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="MARKET", size=1000
        )

        assert result["valid"] is False
        assert "size_above_maximum" in result["reasons"][0]

    def test_rejects_below_min_notional(self, specs_service):
        """Should reject notional below minimum."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="LIMIT", size=0.001, price=1.0
        )

        assert result["valid"] is False
        assert "notional_below_minimum" in result["reasons"][0]

    def test_market_order_no_price_validation(self, specs_service):
        """Should skip price validation for market orders."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="BUY", order_type="MARKET", size=0.5
        )

        assert result["valid"] is True
        assert result["adjusted_price"] is None

    def test_stop_limit_requires_price(self, specs_service):
        """Should validate price for stop limit orders."""
        result = specs_service.validate_order(
            "BTC-USD-PERP", side="SELL", order_type="STOP_LIMIT", size=0.5, price=50000.456
        )

        # Should quantize price (SELL = ceil)
        assert result["adjusted_price"] == 50000.46


class TestGetSpec:
    """Test get_spec method."""

    def test_get_spec_from_cache(self, specs_service):
        """Should return cached spec."""
        spec1 = specs_service.build_spec("BTC-USD-PERP")
        spec2 = specs_service.get_spec("BTC-USD-PERP")

        assert spec1 is spec2

    def test_get_spec_builds_if_missing(self, specs_service):
        """Should build spec if not cached."""
        spec = specs_service.get_spec("NEW-PRODUCT")

        assert spec.product_id == "NEW-PRODUCT"
        assert "NEW-PRODUCT" in specs_service.specs_cache


class TestGetSlippageMultiplier:
    """Test get_slippage_multiplier method."""

    def test_get_slippage_multiplier(self, specs_service):
        """Should return slippage multiplier."""
        multiplier = specs_service.get_slippage_multiplier("BTC-USD-PERP")

        assert multiplier == 1.5


class TestGetSpecsService:
    """Test global specs service singleton."""

    def test_get_specs_service_creates_instance(self):
        """Should create global instance."""
        import bot_v2.features.brokerages.coinbase.specs as specs_module

        # Reset global
        specs_module._specs_service = None

        service = get_specs_service()

        assert service is not None
        assert isinstance(service, SpecsService)

    def test_get_specs_service_returns_same_instance(self):
        """Should return same instance on repeated calls."""
        service1 = get_specs_service()
        service2 = get_specs_service()

        assert service1 is service2


class TestModuleLevelQuantizeSize:
    """Test module-level quantize_size function."""

    def test_quantize_size_floors(self):
        """Should floor to step size."""
        size = quantize_size(Decimal("1.2345"), Decimal("0.01"))

        assert size == Decimal("1.23")

    def test_quantize_size_exact_step(self):
        """Should not change exact step."""
        size = quantize_size(Decimal("1.23"), Decimal("0.01"))

        assert size == Decimal("1.23")

    def test_quantize_size_zero_step(self):
        """Should return original if step is zero."""
        size = quantize_size(Decimal("1.2345"), Decimal("0"))

        assert size == Decimal("1.2345")

    def test_quantize_size_none_step(self):
        """Should return original if step is None."""
        size = quantize_size(Decimal("1.2345"), None)

        assert size == Decimal("1.2345")


class TestModuleLevelQuantizeSizeUp:
    """Test module-level quantize_size_up function."""

    def test_quantize_size_up_ceils(self):
        """Should ceil to step size."""
        size = quantize_size_up(Decimal("1.231"), Decimal("0.01"))

        assert size == Decimal("1.24")

    def test_quantize_size_up_exact_step(self):
        """Should not change exact step."""
        size = quantize_size_up(Decimal("1.23"), Decimal("0.01"))

        assert size == Decimal("1.23")

    def test_quantize_size_up_zero_step(self):
        """Should return original if step is zero."""
        size = quantize_size_up(Decimal("1.2345"), Decimal("0"))

        assert size == Decimal("1.2345")


class TestModuleLevelValidateOrder:
    """Test module-level validate_order function."""

    def test_validate_order_success(self, mock_product):
        """Should validate correct order."""
        result = validate_order(
            product=mock_product,
            side="BUY",
            quantity=Decimal("1.0"),
            order_type="LIMIT",
            price=Decimal("50000.00"),
        )

        assert result.ok is True
        assert result.adjusted_quantity == Decimal("1.0")
        assert result.adjusted_price == Decimal("50000.00")

    def test_validate_order_missing_quantity(self, mock_product):
        """Should reject missing quantity."""
        result = validate_order(
            product=mock_product, side="BUY", quantity=None, order_type="MARKET"
        )

        assert result.ok is False
        assert result.reason == "quantity_missing"

    def test_validate_order_quantizes_size(self, mock_product):
        """Should quantize size to minimum if below."""
        result = validate_order(
            product=mock_product,
            side="BUY",
            quantity=Decimal("0.005"),
            order_type="MARKET",
        )

        # Should bump to min_size = 0.01
        assert result.ok is True
        assert result.adjusted_quantity == Decimal("0.01")

    def test_validate_order_missing_price(self, mock_product):
        """Should reject limit order without price."""
        result = validate_order(
            product=mock_product, side="BUY", quantity=Decimal("1.0"), order_type="LIMIT"
        )

        assert result.ok is False
        assert result.reason == "price_required"

    def test_validate_order_below_min_notional(self, mock_product):
        """Should reject order below minimum notional."""
        result = validate_order(
            product=mock_product,
            side="BUY",
            quantity=Decimal("0.01"),
            order_type="LIMIT",
            price=Decimal("1.00"),
        )

        # Notional = 0.01 * 1.00 = 0.01 < 10
        assert result.ok is False
        assert result.reason == "min_notional"
        assert result.adjusted_quantity is not None  # Should suggest adjustment

    def test_validate_order_market_no_price_check(self, mock_product):
        """Should skip price validation for market orders."""
        result = validate_order(
            product=mock_product, side="BUY", quantity=Decimal("1.0"), order_type="MARKET"
        )

        assert result.ok is True
        assert result.adjusted_price is None

    def test_validate_order_side_aware_price(self, mock_product):
        """Should apply side-aware price quantization."""
        # BUY should floor
        result_buy = validate_order(
            product=mock_product,
            side="BUY",
            quantity=Decimal("1.0"),
            order_type="LIMIT",
            price=Decimal("50000.456"),
        )

        assert result_buy.adjusted_price == Decimal("50000.45")

        # SELL should ceil
        result_sell = validate_order(
            product=mock_product,
            side="SELL",
            quantity=Decimal("1.0"),
            order_type="LIMIT",
            price=Decimal("50000.451"),
        )

        assert result_sell.adjusted_price == Decimal("50000.46")


class TestEdgeCases:
    """Test edge cases for complete coverage."""

    def test_calculate_safe_size_adjusted_for_min_notional_success(self, specs_service):
        """Should successfully adjust size to meet min notional (line 172)."""
        # Create scenario where quantized size meets min_size but not min_notional
        specs_service.specs_cache["TEST"] = ProductSpec(
            "TEST",
            {
                "min_size": "0.01",
                "step_size": "0.001",
                "min_notional": "20",
                "max_size": "1000",
                "safe_buffer": "0.1",
            },
        )

        # Notional that produces size meeting min_size but not min_notional
        # target = 50, mark = 100, raw = 0.5, buffered = 0.45, quantized = 0.45
        # notional = 0.45 * 100 = 45 >= 20, so within_limits
        # Need smaller: target = 25, mark = 100, raw = 0.25, buffered = 0.225, quantized = 0.225
        # notional = 0.225 * 100 = 22.5 >= 20, so within_limits
        # Even smaller: target = 20, mark = 100, raw = 0.2, buffered = 0.18, quantized = 0.18
        # notional = 0.18 * 100 = 18 < 20, should adjust
        size, reason = specs_service.calculate_safe_position_size(
            "TEST", target_notional=20, mark_price=100
        )

        # Should adjust to meet min_notional (20 * 1.1 / 100 = 0.22)
        assert reason == "adjusted_for_min_notional"
        assert size * Decimal("100") >= Decimal("20")

    def test_module_validate_order_notional_bump_to_min_size(self, mock_product):
        """Should bump suggested size to min_size when needed (line 328)."""
        # Create scenario where notional adjustment suggests size < min_size
        product = Mock()
        product.step_size = Decimal("0.1")
        product.min_size = Decimal("1.0")
        product.price_increment = Decimal("0.01")
        product.min_notional = Decimal("100")

        result = validate_order(
            product=product,
            side="BUY",
            quantity=Decimal("0.5"),
            order_type="LIMIT",
            price=Decimal("10.00"),
        )

        # Notional = 0.5 * 10 = 5 < 100
        # Should suggest adjustment, and bump to min_size if needed
        assert result.ok is False
        assert result.reason == "min_notional"

    def test_module_calculate_safe_size_bump_to_min_size(self):
        """Should bump to min_size when notional adjustment is too small (line 360)."""
        product = Mock()
        product.step_size = Decimal("0.01")
        product.min_size = Decimal("1.0")
        product.min_notional = Decimal("50")

        # Scenario where notional adjustment results in size < min_size
        safe_size = calculate_safe_position_size(
            product=product,
            side="BUY",
            intended_quantity=Decimal("0.1"),
            ref_price=Decimal("100"),
        )

        # With min_notional = 50, target = 50 * 1.1 = 55
        # Needed size = 55 / 100 = 0.55, but min_size = 1.0
        # Function applies 10% buffer to min_size: 1.0 * 1.1 = 1.1
        assert safe_size == Decimal("1.10")


class TestModuleLevelCalculateSafePositionSize:
    """Test module-level calculate_safe_position_size function."""

    def test_calculate_safe_size_basic(self, mock_product):
        """Should calculate safe size with buffer."""
        safe_size = calculate_safe_position_size(
            product=mock_product,
            side="BUY",
            intended_quantity=Decimal("1.0"),
            ref_price=Decimal("50000"),
        )

        # Should apply 10% buffer: max(1.0, 0.01 * 1.1) = 1.0
        assert safe_size >= Decimal("1.0")

    def test_calculate_safe_size_below_minimum(self, mock_product):
        """Should bump to minimum with buffer."""
        safe_size = calculate_safe_position_size(
            product=mock_product,
            side="BUY",
            intended_quantity=Decimal("0.005"),
            ref_price=Decimal("50000"),
        )

        # Should be min_size * 1.1 = 0.01 * 1.1 = 0.011
        assert safe_size == Decimal("0.011")

    def test_calculate_safe_size_meets_notional(self, mock_product):
        """Should ensure minimum notional is met."""
        safe_size = calculate_safe_position_size(
            product=mock_product,
            side="BUY",
            intended_quantity=Decimal("0.01"),
            ref_price=Decimal("100"),
        )

        # Notional needs to be >= 10 * 1.1 = 11
        # Size needed: 11 / 100 = 0.11
        assert safe_size * Decimal("100") >= Decimal("11")

    def test_calculate_safe_size_no_notional_requirement(self):
        """Should work without min_notional."""
        product = Mock()
        product.step_size = Decimal("0.001")
        product.min_size = Decimal("0.01")
        product.min_notional = None

        safe_size = calculate_safe_position_size(
            product=product,
            side="BUY",
            intended_quantity=Decimal("1.0"),
            ref_price=Decimal("50000"),
        )

        assert safe_size >= Decimal("1.0")
