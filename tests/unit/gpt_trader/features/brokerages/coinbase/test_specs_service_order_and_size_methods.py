"""Unit tests for `SpecsService` order and safe-size methods."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.specs import SpecsService


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
