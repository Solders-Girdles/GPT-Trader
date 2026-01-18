"""Tests for OrderValidator.maybe_preview_order."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.features.live_trade.risk import ValidationError


class TestMaybePreviewOrder:
    def test_preview_disabled_skips(
        self,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=False,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        validator._record_preview.assert_not_called()

    def test_broker_without_preview_skips(
        self,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        broker = MagicMock(spec=[])

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        validator._record_preview.assert_not_called()

    def test_preview_success_records_result(
        self,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        broker = MagicMock()
        broker.preview_order = MagicMock(return_value={"estimated_fee": "0.1"})
        broker.edit_order_preview = MagicMock()

        record_preview = MagicMock()
        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=record_preview,
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=Decimal("49000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        broker.preview_order.assert_called_once()
        preview_kwargs = broker.preview_order.call_args.kwargs
        assert preview_kwargs["quantity"] == Decimal("1.0")
        assert preview_kwargs["stop_price"] == Decimal("49000")
        record_preview.assert_called_once()
        call_args = record_preview.call_args[0]
        assert call_args[0] == "BTC-PERP"
        assert call_args[5] == {"estimated_fee": "0.1"}

    def test_preview_validation_error_propagates(
        self,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        broker = MagicMock()
        broker.preview_order = MagicMock(side_effect=ValidationError("Insufficient margin"))
        broker.edit_order_preview = MagicMock()

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        with pytest.raises(ValidationError, match="Insufficient margin"):
            validator.maybe_preview_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                order_quantity=Decimal("1.0"),
                effective_price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=10,
            )

    def test_preview_generic_exception_is_suppressed(
        self,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        broker = MagicMock()
        broker.preview_order = MagicMock(side_effect=RuntimeError("API error"))
        broker.edit_order_preview = MagicMock()

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )
        broker.preview_order.assert_called_once()
        mock_failure_tracker.record_failure.assert_called_once_with("order_preview")

    def test_preview_with_non_tif_value(
        self,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        broker = MagicMock()
        broker.preview_order = MagicMock(return_value={})
        broker.edit_order_preview = MagicMock()

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=10,
        )

        assert broker.preview_order.call_args.kwargs["tif"] == TimeInForce.GTC
