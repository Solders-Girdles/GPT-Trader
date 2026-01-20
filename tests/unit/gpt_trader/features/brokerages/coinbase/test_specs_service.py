"""Unit tests for SpecsService loading, caching, and helper functions."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from gpt_trader.core import MarketType, Product
from gpt_trader.features.brokerages.coinbase.specs import (
    SpecsService,
    calculate_safe_position_size,
    quantize_price_side_aware,
    validate_order,
)


def make_product(symbol: str) -> Product:
    return Product(
        symbol=symbol,
        base_asset=symbol.split("-")[0],
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=None,
    )


def test_side_aware_price_quantization_buy_sell():
    inc = Decimal("0.01")
    # BUY rounds down
    assert quantize_price_side_aware(Decimal("123.4567"), inc, "buy") == Decimal("123.45")
    # SELL rounds up
    assert quantize_price_side_aware(Decimal("123.4567"), inc, "sell") == Decimal("123.46")


def test_validate_order_enforces_min_size_with_buffer():
    p = make_product("BTC-PERP")
    # Intend size below min -> validator should bump to the minimum tradable size
    vr = validate_order(
        product=p,
        side="buy",
        quantity=Decimal("0.0001"),
        order_type="limit",
        price=Decimal("50000"),
    )
    assert vr.ok is True
    assert vr.adjusted_quantity == p.min_size


def test_validate_order_enforces_min_notional():
    p = make_product("ETH-PERP")
    # Price * quantity just below min_notional * 1.1 should be rejected with adjusted suggestion
    vr = validate_order(
        product=p,
        side="buy",
        quantity=Decimal("0.001"),
        order_type="limit",
        price=Decimal("9000"),
    )
    # Validator will suggest adjusted quantity; does not auto-bump
    assert vr.ok is False
    assert vr.reason == "min_notional"
    assert vr.adjusted_quantity == Decimal("0.002")


def test_calculate_safe_position_size_bumps_to_clear_thresholds():
    p = make_product("SOL-PERP")
    # Intended tiny size should be bumped to clear min_size and min_notional with buffer
    safe = calculate_safe_position_size(
        product=p, side="buy", intended_quantity=Decimal("0.0001"), ref_price=Decimal("100")
    )
    assert safe >= p.min_size


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

    def test_service_env_config_path(self, monkeypatch: pytest.MonkeyPatch):
        """Test service uses environment variable for config path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = {"products": {"ETH-PERP": {"min_size": 0.002}}}
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            monkeypatch.setenv("PERPS_SPECS_PATH", config_path)
            service = SpecsService()
            assert "ETH-PERP" in service.overrides
            assert service.overrides["ETH-PERP"]["min_size"] == 0.002
        finally:
            Path(config_path).unlink()


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
