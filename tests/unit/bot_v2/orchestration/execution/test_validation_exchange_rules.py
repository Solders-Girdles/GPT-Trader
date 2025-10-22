"""Tests for order validation exchange rules and quantization."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.features.live_trade.risk import ValidationError


class TestExchangeRulesValidation:
    """Test validate_exchange_rules method."""

    def test_validate_market_order_with_no_price(self, order_validator, sample_product) -> None:
        """Test market order validation with no price."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.MARKET
        order_quantity = Decimal("0.1")
        price = None
        effective_price = Decimal("50000.0")

        # Mock spec validation to pass
        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_spec.return_value = MagicMock(ok=True, adjusted_price=None, adjusted_quantity=None)

            result_quantity, result_price = order_validator.validate_exchange_rules(
                symbol, side, order_type, order_quantity, price, effective_price, sample_product
            )

            # Should pass through market order unchanged
            assert result_quantity == order_quantity
            assert result_price is None
            mock_spec.assert_called_once_with(
                product=sample_product,
                side=side.value,
                quantity=order_quantity,
                order_type=order_type.value.lower(),
                price=None,
            )

    def test_validate_limit_order_with_price_quantization(
        self, order_validator, sample_product
    ) -> None:
        """Test limit order validation with price quantization."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = Decimal("50000.05")  # Needs quantization
        effective_price = Decimal("50000.05")

        # Mock spec validation and quantization
        with (
            patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec,
            patch(
                "bot_v2.orchestration.execution.validation.quantize_price_side_aware"
            ) as mock_quantize,
        ):

            mock_spec.return_value = MagicMock(ok=True, adjusted_price=None, adjusted_quantity=None)
            mock_quantize.return_value = Decimal("50000.0")  # Quantized price

            result_quantity, result_price = order_validator.validate_exchange_rules(
                symbol, side, order_type, order_quantity, price, effective_price, sample_product
            )

            assert result_quantity == order_quantity
            assert result_price == Decimal("50000.0")
            mock_quantize.assert_called_once_with(
                Decimal("50000.05"), sample_product.price_increment, side.value
            )

    def test_validate_limit_order_with_spec_adjustments(
        self, order_validator, sample_product
    ) -> None:
        """Test limit order validation with spec-provided adjustments."""
        symbol = "BTC-PERP"
        side = OrderSide.SELL
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = Decimal("50000.0")
        effective_price = Decimal("50000.0")

        # Mock spec validation with adjustments
        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.adjusted_price = Decimal("50000.10")  # Price adjusted by spec
            mock_result.adjusted_quantity = Decimal("0.099")  # Quantity adjusted by spec
            mock_spec.return_value = mock_result

            result_quantity, result_price = order_validator.validate_exchange_rules(
                symbol, side, order_type, order_quantity, price, effective_price, sample_product
            )

            assert result_quantity == Decimal("0.099")
            assert result_price == Decimal("50000.10")

    def test_validate_limit_order_uses_effective_price_when_none(
        self, order_validator, sample_product
    ) -> None:
        """Test limit order uses effective_price when price is None."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = None
        effective_price = Decimal("50000.0")

        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_spec.return_value = MagicMock(ok=True, adjusted_price=None, adjusted_quantity=None)

            result_quantity, result_price = order_validator.validate_exchange_rules(
                symbol, side, order_type, order_quantity, price, effective_price, sample_product
            )

            # Should use effective_price when price is None
            mock_spec.assert_called_once_with(
                product=sample_product,
                side=side.value,
                quantity=order_quantity,
                order_type=order_type.value.lower(),
                price=Decimal("50000.0"),
            )

    def test_validate_spec_failure_records_rejection(self, order_validator, sample_product) -> None:
        """Test that spec validation failure records rejection and raises error."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = Decimal("50000.0")
        effective_price = Decimal("50000.0")

        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_result = MagicMock()
            mock_result.ok = False
            mock_result.reason = "invalid_quantity"
            mock_spec.return_value = mock_result

            with pytest.raises(ValidationError, match="Spec validation failed: invalid_quantity"):
                order_validator.validate_exchange_rules(
                    symbol, side, order_type, order_quantity, price, effective_price, sample_product
                )

            # Verify rejection was recorded
            order_validator._record_rejection.assert_called_once_with(
                symbol, side.value, order_quantity, price, "invalid_quantity"
            )

    def test_validate_spec_failure_with_no_reason(self, order_validator, sample_product) -> None:
        """Test spec validation failure with no reason provided."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = Decimal("50000.0")
        effective_price = Decimal("50000.0")

        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_result = MagicMock()
            mock_result.ok = False
            mock_result.reason = None
            mock_spec.return_value = mock_result

            with pytest.raises(ValidationError, match="Spec validation failed: spec_violation"):
                order_validator.validate_exchange_rules(
                    symbol, side, order_type, order_quantity, price, effective_price, sample_product
                )

            # Should use default reason when none provided
            order_validator._record_rejection.assert_called_once_with(
                symbol, side.value, order_quantity, price, "spec_violation"
            )

    def test_validate_spec_failure_uses_effective_price_for_rejection(
        self, order_validator, sample_product
    ) -> None:
        """Test spec failure rejection uses effective_price when price is None."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = None
        effective_price = Decimal("50000.0")

        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_result = MagicMock()
            mock_result.ok = False
            mock_result.reason = "invalid_quantity"
            mock_spec.return_value = mock_result

            with pytest.raises(ValidationError):
                order_validator.validate_exchange_rules(
                    symbol, side, order_type, order_quantity, price, effective_price, sample_product
                )

            # Should use effective_price for rejection recording
            order_validator._record_rejection.assert_called_once_with(
                symbol, side.value, order_quantity, effective_price, "invalid_quantity"
            )

    def test_validate_limit_order_no_quantization_when_no_price(
        self, order_validator, sample_product
    ) -> None:
        """Test limit order skips quantization when price is None."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        price = None
        effective_price = Decimal("50000.0")

        with (
            patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec,
            patch(
                "bot_v2.orchestration.execution.validation.quantize_price_side_aware"
            ) as mock_quantize,
        ):

            mock_spec.return_value = MagicMock(ok=True, adjusted_price=None, adjusted_quantity=None)

            order_validator.validate_exchange_rules(
                symbol, side, order_type, order_quantity, price, effective_price, sample_product
            )

            # Quantization should not be called when price is None
            mock_quantize.assert_not_called()

    def test_validate_market_order_price_conversion(self, order_validator, sample_product) -> None:
        """Test market order validation properly handles price conversion."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.MARKET
        order_quantity = Decimal("0.1")
        price = None
        effective_price = 50000.0  # Float instead of Decimal

        with patch("bot_v2.orchestration.execution.validation.spec_validate_order") as mock_spec:
            mock_spec.return_value = MagicMock(ok=True, adjusted_price=None, adjusted_quantity=None)

            order_validator.validate_exchange_rules(
                symbol, side, order_type, order_quantity, price, effective_price, sample_product
            )

            # Should convert effective_price to Decimal for validator_price calculation
            # But still pass None for market orders
            mock_spec.assert_called_once_with(
                product=sample_product,
                side=side.value,
                quantity=order_quantity,
                order_type=order_type.value.lower(),
                price=None,
            )
