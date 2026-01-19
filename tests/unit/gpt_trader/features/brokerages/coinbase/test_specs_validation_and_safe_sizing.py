"""Unit tests for order validation and safe position sizing helpers."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.specs import (
    calculate_safe_position_size,
    validate_order,
)


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
