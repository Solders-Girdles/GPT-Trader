"""Shared fixtures for execution layer tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.orchestration.execution.validation import OrderValidator


@pytest.fixture
def mock_brokerage():
    """Mock brokerage adapter for tests."""
    broker = MagicMock(spec=IBrokerage)

    # Add optional methods that tests might need
    broker.get_market_snapshot = MagicMock(return_value={
        "spread_bps": 5,
        "depth_l1": Decimal("1000000"),
    })

    return broker


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager for tests."""
    manager = MagicMock(spec=LiveRiskManager)

    # Default behavior - no staleness, not reduce-only
    manager.check_mark_staleness.return_value = False
    manager.is_reduce_only_mode.return_value = False
    manager.pre_trade_validate.return_value = None

    # Mock config for slippage guard
    manager.config = MagicMock()
    manager.config.slippage_guard_bps = 50

    return manager


@pytest.fixture
def sample_product():
    """Sample product for order validation tests."""
    product = MagicMock(spec=Product)
    product.symbol = "BTC-PERP"
    product.price_increment = Decimal("0.1")
    product.size_increment = Decimal("0.001")
    product.min_size = Decimal("0.001")
    return product


@pytest.fixture
def mock_preview_callbacks():
    """Mock preview and rejection callbacks."""
    preview_callback = MagicMock()
    rejection_callback = MagicMock()
    return preview_callback, rejection_callback


@pytest.fixture
def order_validator(mock_brokerage, mock_risk_manager, mock_preview_callbacks):
    """OrderValidator instance with mocked dependencies."""
    preview_callback, rejection_callback = mock_preview_callbacks
    return OrderValidator(
        broker=mock_brokerage,
        risk_manager=mock_risk_manager,
        enable_order_preview=True,
        record_preview_callback=preview_callback,
        record_rejection_callback=rejection_callback,
    )


@pytest.fixture
def order_validator_no_preview(mock_brokerage, mock_risk_manager, mock_preview_callbacks):
    """OrderValidator instance with order preview disabled."""
    preview_callback, rejection_callback = mock_preview_callbacks
    return OrderValidator(
        broker=mock_brokerage,
        risk_manager=mock_risk_manager,
        enable_order_preview=False,
        record_preview_callback=preview_callback,
        record_rejection_callback=rejection_callback,
    )


@pytest.fixture
def sample_order_params():
    """Standard order parameters for tests."""
    return {
        "symbol": "BTC-PERP",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "order_quantity": Decimal("0.1"),
        "price": Decimal("50000.0"),
        "effective_price": Decimal("50000.0"),
    }


@pytest.fixture
def preview_broker():
    """Mock broker that supports order preview."""
    from bot_v2.orchestration.execution.validation import _PreviewBroker

    broker = MagicMock(spec=_PreviewBroker)  # Use _PreviewBroker spec instead

    # Add optional methods
    broker.get_market_snapshot = MagicMock(return_value={
        "spread_bps": 5,
        "depth_l1": Decimal("1000000"),
    })

    # Mock preview_order method
    preview_data = {
        "order_id": "preview_123",
        "estimated_cost": Decimal("5000.00"),
        "estimated_fee": Decimal("5.00"),
    }
    broker.preview_order = MagicMock(return_value=preview_data)
    broker.edit_order_preview = MagicMock(return_value={"success": True})

    return broker


@pytest.fixture
def validation_error_cases():
    """Common validation error scenarios for testing."""
    return {
        "stale_mark": "Mark price is stale for BTC-PERP; halting order placement",
        "spec_violation": "Spec validation failed: invalid_quantity",
        "slippage_exceeded": "Expected slippage 75 bps exceeds guard 50",
    }