"""Tests for OrderValidator.validate_exchange_rules."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.validation as validation_module
from gpt_trader.core import OrderSide, OrderType, Product
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.features.live_trade.risk import ValidationError


def _make_validation_result(
    ok: bool = True,
    reason: str | None = None,
    adjusted_quantity: Decimal | None = None,
    adjusted_price: Decimal | None = None,
) -> MagicMock:
    return MagicMock(
        ok=ok,
        reason=reason,
        adjusted_quantity=adjusted_quantity,
        adjusted_price=adjusted_price,
    )


@pytest.fixture
def spec_validate_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_validate = MagicMock()
    monkeypatch.setattr(validation_module, "spec_validate_order", mock_validate)
    return mock_validate


@pytest.fixture
def quantize_price_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_quantize = MagicMock()
    monkeypatch.setattr(validation_module, "quantize_price_side_aware", mock_quantize)
    return mock_quantize


class TestValidateExchangeRules:
    def test_market_order_uses_none_price(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result()

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        spec_validate_mock.assert_called_once()
        assert spec_validate_mock.call_args.kwargs["price"] is None

    def test_limit_order_uses_provided_price(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result()

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        spec_validate_mock.assert_called_once()
        assert spec_validate_mock.call_args.kwargs["price"] == Decimal("49000")

    def test_limit_order_falls_back_to_effective_price(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result()

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        spec_validate_mock.assert_called_once()
        assert spec_validate_mock.call_args.kwargs["price"] == Decimal("50000")

    def test_validation_failure_raises_error(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result(
            ok=False,
            reason="size_below_minimum",
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

    def test_validation_uses_adjusted_quantity(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result(adjusted_quantity=Decimal("0.5"))

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

    def test_validation_failure_with_none_reason(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result(ok=False)

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

    def test_limit_order_price_quantization(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
        quantize_price_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result()
        quantize_price_mock.return_value = Decimal("49000.50")

        qty, price = validator.validate_exchange_rules(  # naming: allow
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000.555"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        quantize_price_mock.assert_called_once()
        assert price == Decimal("49000.50")

    def test_adjusted_values_from_validation(
        self,
        validator: OrderValidator,
        mock_product: Product,
        spec_validate_mock: MagicMock,
    ) -> None:
        spec_validate_mock.return_value = _make_validation_result(
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
