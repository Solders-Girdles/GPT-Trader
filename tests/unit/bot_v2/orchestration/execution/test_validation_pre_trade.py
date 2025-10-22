"""Tests for pre-trade validation and order preview functionality."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    OrderSide,
    OrderType,
    TimeInForce,
)
from bot_v2.features.live_trade.risk import ValidationError


class TestPreTradeValidation:
    """Test run_pre_trade_validation method."""

    def test_pre_trade_validation_success(self, order_validator, sample_product) -> None:
        """Test successful pre-trade validation."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")
        equity = Decimal("10000.0")
        current_positions = {"BTC-PERP": {"size": Decimal("0.05"), "side": "long"}}

        # Should not raise any error
        order_validator.run_pre_trade_validation(
            symbol=symbol,
            side=side,
            order_quantity=order_quantity,
            effective_price=effective_price,
            product=sample_product,
            equity=equity,
            current_positions=current_positions,
        )

        # Verify risk manager was called with correct parameters
        order_validator.risk_manager.pre_trade_validate.assert_called_once_with(
            symbol=symbol,
            side=side.value,
            quantity=order_quantity,
            price=effective_price,
            product=sample_product,
            equity=equity,
            current_positions=current_positions,
        )

    def test_pre_trade_validation_propagates_validation_error(
        self, order_validator, sample_product
    ) -> None:
        """Test that ValidationError from risk manager is propagated."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("1.0")  # Large quantity
        effective_price = Decimal("50000.0")
        equity = Decimal("1000.0")  # Insufficient equity
        current_positions = {}

        # Mock risk manager to raise ValidationError
        order_validator.risk_manager.pre_trade_validate.side_effect = ValidationError(
            "Insufficient margin for this order"
        )

        with pytest.raises(ValidationError, match="Insufficient margin for this order"):
            order_validator.run_pre_trade_validation(
                symbol=symbol,
                side=side,
                order_quantity=order_quantity,
                effective_price=effective_price,
                product=sample_product,
                equity=equity,
                current_positions=current_positions,
            )

    def test_pre_trade_validation_with_complex_positions(
        self, order_validator, sample_product
    ) -> None:
        """Test pre-trade validation with complex position scenarios."""
        symbol = "ETH-PERP"
        side = OrderSide.SELL
        order_quantity = Decimal("2.0")
        effective_price = Decimal("3000.0")
        equity = Decimal("15000.0")
        current_positions = {
            "BTC-PERP": {"size": Decimal("0.1"), "side": "long"},
            "ETH-PERP": {"size": Decimal("1.5"), "side": "long"},
            "SOL-PERP": {"size": Decimal("0.5"), "side": "short"},
        }

        order_validator.run_pre_trade_validation(
            symbol=symbol,
            side=side,
            order_quantity=order_quantity,
            effective_price=effective_price,
            product=sample_product,
            equity=equity,
            current_positions=current_positions,
        )

        # Verify all position data was passed through
        order_validator.risk_manager.pre_trade_validate.assert_called_once_with(
            symbol=symbol,
            side=side.value,
            quantity=order_quantity,
            price=effective_price,
            product=sample_product,
            equity=equity,
            current_positions=current_positions,
        )


class TestOrderPreview:
    """Test maybe_preview_order method."""

    def test_order_preview_disabled_no_action(self, order_validator_no_preview) -> None:
        """Test no action when preview is disabled."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")
        stop_price = None
        tif = TimeInForce.GTC
        reduce_only = False
        leverage = 10

        # Should not call preview or record callback
        order_validator_no_preview.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Verify preview was not recorded (preview method may not exist on all brokers)
        order_validator_no_preview._record_preview.assert_not_called()

    def test_order_preview_broker_not_supported_no_action(self, order_validator) -> None:
        """Test no action when broker doesn't support preview."""
        # Mock broker that doesn't support preview
        broker_without_preview = MagicMock(spec=IBrokerage)
        del broker_without_preview.preview_order
        order_validator.broker = broker_without_preview

        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")
        stop_price = None
        tif = TimeInForce.GTC
        reduce_only = False
        leverage = 10

        order_validator.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Should not record any preview
        order_validator._record_preview.assert_not_called()

    def test_order_preview_successful(self, order_validator) -> None:
        """Test successful order preview."""

        # Create a broker that implements the _PreviewBroker protocol
        class MockPreviewBroker:
            def __init__(self):
                self.get_market_snapshot = MagicMock(
                    return_value={
                        "spread_bps": 5,
                        "depth_l1": Decimal("1000000"),
                    }
                )
                self.preview_order = MagicMock()
                self.edit_order_preview = MagicMock()

        mock_preview_broker = MockPreviewBroker()
        order_validator.broker = mock_preview_broker

        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")
        stop_price = None
        tif = TimeInForce.GTC
        reduce_only = False
        leverage = 10

        # Mock preview data
        preview_data = {
            "order_id": "preview_123",
            "estimated_cost": Decimal("5000.00"),
            "estimated_fee": Decimal("5.00"),
        }
        mock_preview_broker.preview_order.return_value = preview_data

        order_validator.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Verify broker was called with correct parameters
        mock_preview_broker.preview_order.assert_called_once_with(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=effective_price,
            stop_price=stop_price,
            tif=TimeInForce.GTC,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Verify preview was recorded
        order_validator._record_preview.assert_called_once_with(
            symbol, side, order_type, order_quantity, effective_price, preview_data
        )

    def test_order_preview_with_string_tif(self, order_validator, preview_broker) -> None:
        """Test order preview with string TimeInForce."""
        order_validator.broker = preview_broker

        symbol = "BTC-PERP"
        side = OrderSide.SELL
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.2")
        effective_price = Decimal("51000.0")
        stop_price = None
        tif = "IOC"  # String instead of enum
        reduce_only = True
        leverage = 5

        preview_data = {"order_id": "preview_456"}
        preview_broker.preview_order.return_value = preview_data

        order_validator.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Should default string TIF to GTC
        preview_broker.preview_order.assert_called_once_with(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=effective_price,
            stop_price=stop_price,
            tif=TimeInForce.GTC,  # Defaulted from string
            reduce_only=reduce_only,
            leverage=leverage,
        )

    def test_order_preview_with_stop_order(self, order_validator, preview_broker) -> None:
        """Test order preview with stop order."""
        order_validator.broker = preview_broker

        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.STOP
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50100.0")
        stop_price = Decimal("50200.0")
        tif = TimeInForce.GTC
        reduce_only = False
        leverage = 10

        preview_data = {"order_id": "preview_stop_789"}
        preview_broker.preview_order.return_value = preview_data

        order_validator.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Verify stop_price was passed through
        preview_broker.preview_order.assert_called_once_with(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

    def test_order_preview_handles_validation_error(self, order_validator, preview_broker) -> None:
        """Test that ValidationError from preview is propagated."""
        order_validator.broker = preview_broker

        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("10.0")  # Very large order
        effective_price = Decimal("50000.0")
        stop_price = None
        tif = TimeInForce.GTC
        reduce_only = False
        leverage = 10

        # Mock preview to raise ValidationError
        preview_broker.preview_order.side_effect = ValidationError("Order size exceeds limit")

        with pytest.raises(ValidationError, match="Order size exceeds limit"):
            order_validator.maybe_preview_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                effective_price=effective_price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
                leverage=leverage,
            )

        # Should not record preview when validation fails
        order_validator._record_preview.assert_not_called()

    def test_order_preview_handles_general_exception(self, order_validator, preview_broker) -> None:
        """Test graceful handling of general exceptions during preview."""
        order_validator.broker = preview_broker

        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.LIMIT
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")
        stop_price = None
        tif = TimeInForce.GTC
        reduce_only = False
        leverage = 10

        # Mock preview to raise general exception
        preview_broker.preview_order.side_effect = RuntimeError("Preview service unavailable")

        # Should not raise any error
        order_validator.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Should not record preview when exception occurs
        order_validator._record_preview.assert_not_called()

    def test_order_preview_with_none_tif(self, order_validator, preview_broker) -> None:
        """Test order preview with None TimeInForce."""
        order_validator.broker = preview_broker

        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_type = OrderType.MARKET
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")
        stop_price = None
        tif = None
        reduce_only = False
        leverage = None

        preview_data = {"order_id": "preview_none_tif"}
        preview_broker.preview_order.return_value = preview_data

        order_validator.maybe_preview_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_quantity=order_quantity,
            effective_price=effective_price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
            leverage=leverage,
        )

        # Should default None TIF to GTC
        preview_broker.preview_order.assert_called_once_with(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=effective_price,
            stop_price=stop_price,
            tif=TimeInForce.GTC,  # Defaulted from None
            reduce_only=reduce_only,
            leverage=leverage,
        )
