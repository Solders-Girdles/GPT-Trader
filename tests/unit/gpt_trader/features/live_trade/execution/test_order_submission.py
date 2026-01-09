"""Tests for features/live_trade/execution/order_submission.py."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.logging.correlation import correlation_context, get_domain_context

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_broker() -> MagicMock:
    """Create a mock broker."""
    broker = MagicMock()
    broker.place_order = MagicMock()
    return broker


@pytest.fixture
def mock_event_store() -> MagicMock:
    """Create a mock event store."""
    store = MagicMock()
    store.store_event = MagicMock()
    store.append_trade = MagicMock()
    store.append_error = MagicMock()
    return store


@pytest.fixture
def open_orders() -> list[str]:
    """Create an open orders list."""
    return []


@pytest.fixture
def submitter(
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    open_orders: list[str],
) -> OrderSubmitter:
    """Create an OrderSubmitter instance."""
    return OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test-bot-123",
        open_orders=open_orders,
        integration_mode=False,
    )


@pytest.fixture
def mock_order() -> Order:
    """Create a mock successful order."""
    return Order(
        id="order-123",
        client_id="client-123",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.PENDING,
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# ============================================================
# Test: __init__
# ============================================================


class TestOrderSubmitterInit:
    """Tests for OrderSubmitter initialization."""

    def test_init_stores_dependencies(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that dependencies are stored correctly."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        assert submitter.broker is mock_broker
        assert submitter.event_store is mock_event_store
        assert submitter.bot_id == "test-bot"
        assert submitter.open_orders is open_orders
        assert submitter.integration_mode is True

    def test_init_defaults_integration_mode_to_false(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that integration_mode defaults to False."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        assert submitter.integration_mode is False


# ============================================================
# Test: record_preview
# ============================================================


class TestRecordPreview:
    """Tests for record_preview method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_with_preview_data(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording a preview with data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"estimated_fee": "0.1"},
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[1] == "test-bot-123"
        assert call_args[2]["event_type"] == "order_preview"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    def test_record_preview_with_none_preview_skips(
        self,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that None preview is skipped."""
        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview=None,
        )

        mock_emit_metric.assert_not_called()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_market_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording a preview with market price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            preview={"estimated_fee": "0.1"},
        )

        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["price"] == "market"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_handles_logger_exception(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that logger exceptions are suppressed."""
        mock_logger = MagicMock()
        mock_logger.log_event.side_effect = RuntimeError("Log error")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"estimated_fee": "0.1"},
        )


# ============================================================
# Test: record_rejection
# ============================================================


class TestRecordRejection:
    """Tests for record_rejection method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_logs_and_emits(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that rejection is logged and metric emitted."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["event_type"] == "order_rejected"
        assert call_args[2]["reason"] == "insufficient_margin"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_includes_client_order_id(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that client_order_id is included in rejection telemetry."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
            client_order_id="custom-order-123",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["client_order_id"] == "custom-order-123"

        # Also verify it's passed to the monitoring logger
        mock_logger.log_order_status_change.assert_called_once()
        log_call_kwargs = mock_logger.log_order_status_change.call_args[1]
        assert log_call_kwargs["client_order_id"] == "custom-order-123"
        assert log_call_kwargs["order_id"] == "custom-order-123"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_without_client_order_id_uses_empty_string(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that missing client_order_id defaults to empty string."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["client_order_id"] == ""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_with_none_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording rejection with None price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=None,
            reason="min_notional",
        )

        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["price"] == "market"


# ============================================================
# Test: _generate_submit_id
# ============================================================


class TestGenerateSubmitId:
    """Tests for _generate_submit_id method."""

    def test_uses_provided_client_order_id(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that provided client order ID is used."""
        result = submitter._generate_submit_id("custom-id-123")
        assert result == "custom-id-123"

    def test_generates_id_when_none(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that ID is generated when None provided."""
        result = submitter._generate_submit_id(None)
        assert result.startswith("test-bot-123_")
        assert len(result) > len("test-bot-123_")

    @patch.dict("os.environ", {"INTEGRATION_TEST_ORDER_ID": "forced-id"})
    def test_integration_mode_uses_env_override(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that integration mode uses environment override."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        result = submitter._generate_submit_id(None)
        assert result == "forced-id"


# ============================================================
# Test: submit_order
# ============================================================


class TestSubmitOrder:
    """Tests for submit_order method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_order_submission(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
        mock_order: Order,
        open_orders: list[str],
    ) -> None:
        """Test successful order submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id=None,
        )

        assert result == "order-123"
        assert "order-123" in open_orders

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejected_order_returns_none(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that rejected orders return None (error is caught internally)."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order = MagicMock()
        rejected_order.id = "order-rejected"
        rejected_order.status = OrderStatus.CANCELLED
        mock_broker.place_order.return_value = rejected_order

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id=None,
        )

        # Rejection is caught internally and returns None
        assert result is None
        # Order should NOT be added to open_orders
        assert "order-rejected" not in open_orders

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_broker_exception_returns_none(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that broker exceptions result in None return."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("API error")

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id=None,
        )

        assert result is None

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_with_none_result_returns_none(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that None order result returns None."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = None

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
            client_order_id="test-id",
        )

        assert result is None


# ============================================================
# Test: _execute_broker_order
# ============================================================


class TestExecuteBrokerOrder:
    """Tests for _execute_broker_order method."""

    def test_normal_order_execution(
        self,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
        mock_order: Order,
    ) -> None:
        """Test normal order execution."""
        mock_broker.place_order.return_value = mock_order

        result = submitter._execute_broker_order(
            submit_id="test-id",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        assert result is mock_order
        mock_broker.place_order.assert_called_once()

    def test_type_error_propagates(
        self,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that TypeError from broker is propagated."""
        mock_broker.place_order.side_effect = TypeError("unexpected keyword argument")

        with pytest.raises(TypeError):
            submitter._execute_broker_order(
                submit_id="test-id",
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                order_quantity=Decimal("1.0"),
                price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=10,
            )


# ============================================================
# Test: _handle_order_result
# ============================================================


class TestHandleOrderResult:
    """Tests for _handle_order_result method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_order_tracked(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_order: Order,
        open_orders: list[str],
    ) -> None:
        """Test that successful orders are tracked."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        result = submitter._handle_order_result(
            order=mock_order,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            reduce_only=False,
            submit_id="test-id",
        )

        assert result == "order-123"
        assert "order-123" in open_orders

    def test_none_order_returns_none(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that None order returns None."""
        result = submitter._handle_order_result(
            order=None,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            reduce_only=False,
            submit_id="test-id",
        )

        assert result is None

    def test_order_without_id_returns_none(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that order without ID returns None."""
        order = MagicMock()
        order.id = None

        result = submitter._handle_order_result(
            order=order,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            reduce_only=False,
            submit_id="test-id",
        )

        assert result is None

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejected_status_raises_error(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that rejected status raises RuntimeError."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order = MagicMock()
        order.id = "rejected-order"
        order.status = MagicMock()
        order.status.value = "REJECTED"

        with pytest.raises(RuntimeError, match="rejected by broker"):
            submitter._handle_order_result(
                order=order,
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                effective_price=Decimal("50000"),
                reduce_only=False,
                submit_id="test-id",
            )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_integration_mode_returns_order_object(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that integration mode returns the order object."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        result = submitter._handle_order_result(
            order=mock_order,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            reduce_only=False,
            submit_id="test-id",
        )

        # In integration mode, returns the order object instead of order.id
        assert result is mock_order


# ============================================================
# Test: _process_rejection
# ============================================================


class TestProcessRejection:
    """Tests for _process_rejection method."""

    def test_integration_mode_stores_event(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that integration mode stores rejection event."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        order = MagicMock()
        order.id = "rejected-order"

        result = submitter._process_rejection(
            order=order,
            status_name="REJECTED",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
        )

        assert result is order
        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "rejected-order",
                "symbol": "BTC-PERP",
                "status": "REJECTED",
            },
        )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_normal_mode_raises_error(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that normal mode raises RuntimeError."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order = MagicMock()
        order.id = "rejected-order"

        with pytest.raises(RuntimeError, match="CANCELLED"):
            submitter._process_rejection(
                order=order,
                status_name="CANCELLED",
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                price=None,
                effective_price=Decimal("50000"),
            )


# ============================================================
# Test: _handle_order_failure
# ============================================================


class TestHandleOrderFailure:
    """Tests for _handle_order_failure method."""

    def test_logs_error_and_returns_none(
        self,
        submitter: OrderSubmitter,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that failure is logged and None returned."""
        result = submitter._handle_order_failure(
            exc=RuntimeError("API error"),
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )

        assert result is None
        mock_event_store.append_error.assert_called_once()

    def test_handles_event_store_exception(
        self,
        submitter: OrderSubmitter,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that event store exceptions are suppressed."""
        mock_event_store.append_error.side_effect = RuntimeError("Store error")

        # Should not raise
        result = submitter._handle_order_failure(
            exc=RuntimeError("API error"),
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )

        assert result is None


# ============================================================
# Test: Integration scenarios
# ============================================================


class TestOrderSubmissionIntegration:
    """Integration tests for order submission workflows."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_full_order_submission_flow(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test complete order submission flow."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id="custom-id",
        )

        # Verify result
        assert result == "order-123"

        # Verify order tracked
        assert "order-123" in open_orders

        # Verify broker called
        mock_broker.place_order.assert_called_once()

        # Verify trade recorded
        mock_event_store.append_trade.assert_called_once()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejection_flow(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test order rejection flow."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order = MagicMock()
        rejected_order.id = "rejected-order"
        rejected_order.status = MagicMock()
        rejected_order.status.value = "FAILED"
        mock_broker.place_order.return_value = rejected_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        # Rejection is caught internally and returns None
        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
            client_order_id=None,
        )

        assert result is None

        # Order should not be tracked
        assert "rejected-order" not in open_orders

        # Rejection should be recorded
        mock_emit_metric.assert_called()


# ============================================================
# Test: _classify_rejection_reason
# ============================================================


class TestClassifyRejectionReason:
    """Tests for _classify_rejection_reason helper."""

    def test_rate_limit_detection(self) -> None:
        """Test rate limit detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("rate_limit exceeded") == "rate_limit"
        assert _classify_rejection_reason("HTTP 429 Too Many Requests") == "rate_limit"
        assert _classify_rejection_reason("too many requests") == "rate_limit"

    def test_insufficient_funds_detection(self) -> None:
        """Test insufficient funds detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Insufficient balance") == "insufficient_funds"
        assert _classify_rejection_reason("Not enough funds") == "insufficient_funds"
        assert _classify_rejection_reason("insufficient margin") == "insufficient_funds"

    def test_invalid_size_detection(self) -> None:
        """Test invalid size detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Invalid size") == "invalid_size"
        assert _classify_rejection_reason("quantity below min_size") == "invalid_size"
        assert _classify_rejection_reason("amount too small") == "invalid_size"

    def test_invalid_price_detection(self) -> None:
        """Test invalid price detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Invalid price") == "invalid_price"
        assert _classify_rejection_reason("price tick increment") == "invalid_price"

    def test_timeout_detection(self) -> None:
        """Test timeout detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Request timeout") == "timeout"
        assert _classify_rejection_reason("Connection timed out") == "timeout"
        assert _classify_rejection_reason("deadline exceeded") == "timeout"

    def test_network_detection(self) -> None:
        """Test network error detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Connection refused") == "network"
        assert _classify_rejection_reason("Network error") == "network"
        assert _classify_rejection_reason("socket closed") == "network"

    def test_generic_rejection(self) -> None:
        """Test generic rejection detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order rejected by broker") == "rejected"
        assert _classify_rejection_reason("Request rejected") == "rejected"

    def test_generic_failure(self) -> None:
        """Test generic failure detection."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order failed") == "failed"
        assert _classify_rejection_reason("Server error") == "failed"

    def test_unknown_fallback(self) -> None:
        """Test unknown fallback."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Something weird happened") == "unknown"
        assert _classify_rejection_reason("") == "unknown"


# ============================================================
# Test: Metrics Recording
# ============================================================


class TestOrderSubmissionMetrics:
    """Tests for order submission metrics recording."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before and after each test."""
        from gpt_trader.monitoring.metrics_collector import reset_all

        reset_all()
        yield
        reset_all()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_order_records_metric(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that successful order records metric with success labels."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id=None,
        )

        collector = get_metrics_collector()
        # Check for success metric with correct labels
        success_key = "gpt_trader_order_submission_total{reason=none,result=success,side=buy}"
        assert success_key in collector.counters
        assert collector.counters[success_key] == 1

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_failed_order_records_metric_with_reason(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failed order records metric with failure reason."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Insufficient balance")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.5"),
            price=None,
            effective_price=Decimal("3000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        collector = get_metrics_collector()
        # Check for failure metric with insufficient_funds reason
        failed_key = (
            "gpt_trader_order_submission_total{reason=insufficient_funds,result=failed,side=sell}"
        )
        assert failed_key in collector.counters
        assert collector.counters[failed_key] == 1


# ============================================================
# Test: Correlation Context Propagation
# ============================================================


class TestCorrelationContextPropagation:
    """Tests for correlation context propagation during order submission."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_context_set_during_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that order context is set during order submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        captured_context: dict = {}

        def capture_context(*args, **kwargs):
            # Capture the domain context during broker call
            captured_context.update(get_domain_context())
            return mock_order

        mock_broker.place_order.side_effect = capture_context

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id="test-order-id",
        )

        # Verify order context was set
        assert captured_context.get("order_id") == "test-order-id"
        assert captured_context.get("symbol") == "BTC-USD"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_correlation_context_preserved_during_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that outer correlation context is preserved during submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        captured_context: dict = {}

        def capture_context(*args, **kwargs):
            captured_context.update(get_domain_context())
            return mock_order

        mock_broker.place_order.side_effect = capture_context

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        # Wrap in outer correlation context (simulating StrategyEngine cycle)
        with correlation_context(cycle=42):
            submitter.submit_order(
                symbol="ETH-USD",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.5"),
                price=None,
                effective_price=Decimal("3000"),
                stop_price=None,
                tif=None,
                reduce_only=True,
                leverage=None,
                client_order_id=None,
            )

        # Verify both outer cycle and inner order context were present
        assert captured_context.get("cycle") == 42
        assert captured_context.get("symbol") == "ETH-USD"
        assert "order_id" in captured_context  # Auto-generated ID

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_context_cleared_after_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that order context is cleared after submission completes."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        with correlation_context(cycle=1):
            submitter.submit_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                order_quantity=Decimal("1.0"),
                price=Decimal("50000"),
                effective_price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=10,
                client_order_id="test-id",
            )

            # After submission, order_id should be cleared but cycle should remain
            context = get_domain_context()
            assert context.get("cycle") == 1
            assert "order_id" not in context
            assert "symbol" not in context


# ============================================================
# Test: Retry Path Idempotency
# ============================================================


class TestRetryPathIdempotency:
    """Tests ensuring retry paths don't create duplicate client_order_ids."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_provided_client_order_id_is_reused_on_retry(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that a provided client_order_id is reused across retries."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        captured_client_ids: list[str] = []

        def capture_client_id(**kwargs):
            captured_client_ids.append(kwargs.get("client_id", ""))
            raise RuntimeError("Simulated transient error")

        mock_broker.place_order.side_effect = capture_client_id

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        # First attempt with explicit client_order_id
        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id="retry-test-123",
        )

        # Second attempt with same client_order_id (simulating retry)
        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id="retry-test-123",
        )

        # Both calls should use the same client_order_id
        assert len(captured_client_ids) == 2
        assert captured_client_ids[0] == "retry-test-123"
        assert captured_client_ids[1] == "retry-test-123"
        assert captured_client_ids[0] == captured_client_ids[1]

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_generated_client_order_id_differs_per_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that auto-generated client_order_ids are unique per submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        captured_client_ids: list[str] = []

        def capture_client_id(**kwargs):
            captured_client_ids.append(kwargs.get("client_id", ""))
            raise RuntimeError("Simulated error")

        mock_broker.place_order.side_effect = capture_client_id

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        # Two separate submissions without explicit client_order_id
        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        # Auto-generated IDs should be unique
        assert len(captured_client_ids) == 2
        assert captured_client_ids[0] != captured_client_ids[1]
        assert captured_client_ids[0].startswith("test-bot_")
        assert captured_client_ids[1].startswith("test-bot_")


# ============================================================
# Test: Broker Status Classification
# ============================================================


class TestBrokerStatusClassification:
    """Tests for _classify_rejection_reason with broker status strings."""

    def test_broker_rejected_status(self) -> None:
        """Test classification of broker REJECTED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order rejected by broker: REJECTED") == "rejected"
        assert _classify_rejection_reason("rejected by exchange") == "rejected"

    def test_broker_cancelled_status(self) -> None:
        """Test classification of broker CANCELLED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        # CANCELLED status should map to "rejected" category
        result = _classify_rejection_reason("Order rejected by broker: CANCELLED")
        assert result == "rejected"

    def test_broker_failed_status(self) -> None:
        """Test classification of broker FAILED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        # Pure "failed" without "reject" in the message
        assert _classify_rejection_reason("Order failed") == "failed"
        assert _classify_rejection_reason("Execution failure") == "failed"
        assert _classify_rejection_reason("FAILED status") == "failed"

        # Note: "Order rejected by broker: FAILED" returns "rejected" because
        # "reject" is checked before "fail" in the classification order

    def test_timeout_variations(self) -> None:
        """Test various timeout error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Request timeout") == "timeout"
        assert _classify_rejection_reason("Connection timed out") == "timeout"
        assert _classify_rejection_reason("deadline exceeded") == "timeout"
        assert _classify_rejection_reason("context deadline exceeded") == "timeout"

    def test_network_variations(self) -> None:
        """Test various network error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Connection refused") == "network"
        assert _classify_rejection_reason("Network error") == "network"
        assert _classify_rejection_reason("socket closed") == "network"
        assert _classify_rejection_reason("connection reset") == "network"
        assert _classify_rejection_reason("DNS resolution failed") == "network"


# ============================================================
# Test: Record Rejection Consistency
# ============================================================


class TestRecordRejectionConsistency:
    """Tests for consistent record_rejection calls with reason and client_order_id."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejection_from_broker_status_includes_classified_reason(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that broker rejections use classified reasons in telemetry."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order = MagicMock()
        rejected_order.id = "rejected-order"
        rejected_order.status = MagicMock()
        rejected_order.status.value = "REJECTED"
        mock_broker.place_order.return_value = rejected_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id="test-client-123",
        )

        # Verify emit_metric was called with rejection event
        calls = mock_emit_metric.call_args_list
        rejection_calls = [c for c in calls if c[0][2].get("event_type") == "order_rejected"]
        assert len(rejection_calls) >= 1

        # Verify reason format includes broker status
        rejection_data = rejection_calls[0][0][2]
        assert "broker_status" in rejection_data["reason"]

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_exception_rejection_uses_classified_reason(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that exception-based rejections use classified reasons."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Insufficient balance for order")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        # Verify error was recorded
        mock_event_store.append_error.assert_called_once()
        call_kwargs = mock_event_store.append_error.call_args[1]
        assert call_kwargs["message"] == "order_placement_failed"


# ============================================================
# Test: Latency Metrics
# ============================================================


class TestOrderSubmissionLatencyMetrics:
    """Tests for order submission latency metrics recording."""

    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_latency"
    )
    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_metric"
    )
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_submission_records_latency_histogram(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_record_metric: MagicMock,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that successful submission records latency histogram."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        # Verify latency histogram was recorded exactly once
        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["result"] == "success"
        assert call_kwargs["side"].lower() == "buy"  # Normalize case for comparison
        assert call_kwargs["latency_seconds"] >= 0  # Latency should be non-negative

    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_latency"
    )
    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_metric"
    )
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_failed_submission_records_latency_with_failure_result(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_record_metric: MagicMock,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failed submission records latency with failure result."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Connection error")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        # Verify latency histogram was recorded with failure
        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["result"] == "failed"
        assert call_kwargs["side"].lower() == "sell"  # Normalize case for comparison

    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_metric"
    )
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_classification_label_used_in_metrics(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_record_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failure reason classification is used in metrics."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Rate limit exceeded")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        # Verify metric was recorded with classified reason
        mock_record_metric.assert_called()
        call_kwargs = mock_record_metric.call_args[1]
        assert call_kwargs["reason"] == "rate_limit"
        assert call_kwargs["result"] == "failed"
