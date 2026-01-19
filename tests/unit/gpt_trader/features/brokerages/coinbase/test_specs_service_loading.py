"""Unit tests for `SpecsService` loading and caching."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import yaml

from gpt_trader.features.brokerages.coinbase.specs import SpecsService


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
