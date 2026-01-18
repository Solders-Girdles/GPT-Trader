"""Tests for OrderValidator.validate_exchange_rules."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import OrderSide, OrderType, Product
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.features.live_trade.risk import ValidationError


class TestValidateExchangeRules:
    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_market_order_uses_none_price(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_spec_validate.assert_called_once()
        assert mock_spec_validate.call_args.kwargs["price"] is None

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_limit_order_uses_provided_price(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_spec_validate.assert_called_once()
        assert mock_spec_validate.call_args.kwargs["price"] == Decimal("49000")

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_limit_order_falls_back_to_effective_price(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_spec_validate.assert_called_once()
        assert mock_spec_validate.call_args.kwargs["price"] == Decimal("50000")

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_validation_failure_raises_error(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=False,
            reason="size_below_minimum",
            adjusted_quantity=None,
            adjusted_price=None,
        )

        with pytest.raises(ValidationError, match="size_below_minimum"):
            validator.validate_exchange_rules(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.0001"),
                price=None,
                effective_price=Decimal("50000"),
                product=mock_product,
            )

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_validation_uses_adjusted_quantity(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=Decimal("0.5"),
            adjusted_price=None,
        )

        adjusted_quantity, _ = validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        assert adjusted_quantity == Decimal("0.5")
        validator._record_rejection.assert_not_called()

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_validation_failure_with_none_reason(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=False,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        with pytest.raises(ValidationError, match="spec_violation"):
            validator.validate_exchange_rules(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.0001"),
                price=None,
                effective_price=Decimal("50000"),
                product=mock_product,
            )

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    @patch("gpt_trader.features.live_trade.execution.validation.quantize_price_side_aware")
    def test_limit_order_price_quantization(
        self,
        mock_quantize: MagicMock,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )
        mock_quantize.return_value = Decimal("49000.50")

        qty, price = validator.validate_exchange_rules(  # naming: allow
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000.555"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_quantize.assert_called_once()
        assert price == Decimal("49000.50")

    @patch("gpt_trader.features.live_trade.execution.validation.spec_validate_order")
    def test_adjusted_values_from_validation(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=Decimal("1.001"),
            adjusted_price=Decimal("49000.00"),
        )

        qty, price = validator.validate_exchange_rules(  # naming: allow
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        assert qty == Decimal("1.001")  # naming: allow
        assert price == Decimal("49000.00")
