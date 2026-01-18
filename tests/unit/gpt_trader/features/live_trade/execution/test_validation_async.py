"""Tests for async methods in features/live_trade/execution/validation.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.features.live_trade.risk import ValidationError


@pytest.fixture
def validator(
    mock_broker: MagicMock,
    mock_risk_manager: MagicMock,
    mock_failure_tracker: MagicMock,
) -> OrderValidator:
    """Create an OrderValidator instance."""
    record_preview = MagicMock()
    record_rejection = MagicMock()
    return OrderValidator(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        enable_order_preview=True,
        record_preview_callback=record_preview,
        record_rejection_callback=record_rejection,
        failure_tracker=mock_failure_tracker,
    )


@pytest.mark.asyncio
async def test_maybe_preview_order_async_success(
    mock_risk_manager: MagicMock,
    mock_failure_tracker: MagicMock,
) -> None:
    """Test that successful async preview is recorded."""
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

    await validator.maybe_preview_order_async(
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

    # Check that preview_order was called
    broker.preview_order.assert_called_once()
    preview_kwargs = broker.preview_order.call_args.kwargs
    assert preview_kwargs["quantity"] == Decimal("1.0")

    # Check that record_preview was called
    record_preview.assert_called_once()

    # Check success record
    mock_failure_tracker.record_success.assert_called_once_with("order_preview")


@pytest.mark.asyncio
async def test_maybe_preview_order_async_validation_error(
    mock_risk_manager: MagicMock,
    mock_failure_tracker: MagicMock,
) -> None:
    """Test that ValidationError propagates in async preview."""
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
        await validator.maybe_preview_order_async(
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


@pytest.mark.asyncio
async def test_maybe_preview_order_async_generic_error(
    mock_risk_manager: MagicMock,
    mock_failure_tracker: MagicMock,
) -> None:
    """Test that generic exceptions are suppressed in async preview."""
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

    # Should not raise
    await validator.maybe_preview_order_async(
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


@pytest.mark.asyncio
async def test_maybe_preview_order_async_disabled(
    mock_broker: MagicMock,
    mock_risk_manager: MagicMock,
    mock_failure_tracker: MagicMock,
) -> None:
    """Test that disabled preview is skipped in async."""
    validator = OrderValidator(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        enable_order_preview=False,
        record_preview_callback=MagicMock(),
        record_rejection_callback=MagicMock(),
        failure_tracker=mock_failure_tracker,
    )

    await validator.maybe_preview_order_async(
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

    mock_broker.preview_order.assert_not_called()
