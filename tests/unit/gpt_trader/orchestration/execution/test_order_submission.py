"""Tests for orchestration/execution/order_submission.py."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.orchestration.execution.broker_executor import BrokerExecutor
from gpt_trader.orchestration.execution.order_submission import OrderSubmitter

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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    def test_type_error_in_non_integration_mode_raises(
        self,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that TypeError is raised in non-integration mode."""
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

    def test_type_error_without_keyword_message_raises(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that TypeError without keyword message is raised even in integration mode."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )
        mock_broker.place_order.side_effect = TypeError("some other error")

        with pytest.raises(TypeError, match="some other error"):
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

    def test_awaitable_in_non_integration_mode_raises(
        self,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that awaitable result in non-integration mode raises TypeError."""

        async def async_place():
            return MagicMock()

        mock_broker.place_order.return_value = async_place()

        with pytest.raises(TypeError, match="awaitable"):
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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
# Test: BrokerExecutor._invoke_legacy_place_order
# ============================================================


class TestInvokeLegacyPlaceOrder:
    """Tests for BrokerExecutor._invoke_legacy_place_order method."""

    def test_creates_order_object_and_calls_broker(
        self,
        mock_broker: MagicMock,
    ) -> None:
        """Test that an Order object is created and passed to broker."""
        executor = BrokerExecutor(
            broker=mock_broker,
            integration_mode=True,
        )

        mock_order = MagicMock()
        mock_broker.place_order.return_value = mock_order

        result = executor._invoke_legacy_place_order(
            submit_id="test-id",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.IOC,
        )

        assert result is mock_order
        mock_broker.place_order.assert_called_once()

        # Verify the order object passed to place_order
        call_args = mock_broker.place_order.call_args[0]
        order = call_args[0]
        assert isinstance(order, Order)
        assert order.id == "test-id"
        assert order.symbol == "BTC-PERP"

    def test_defaults_tif_to_gtc(
        self,
        mock_broker: MagicMock,
    ) -> None:
        """Test that None tif defaults to GTC."""
        executor = BrokerExecutor(
            broker=mock_broker,
            integration_mode=True,
        )

        mock_broker.place_order.return_value = MagicMock()

        executor._invoke_legacy_place_order(
            submit_id="test-id",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
        )

        call_args = mock_broker.place_order.call_args[0]
        order = call_args[0]
        assert order.tif == TimeInForce.GTC


# ============================================================
# Test: BrokerExecutor._await_coroutine
# ============================================================


class TestAwaitCoroutine:
    """Tests for BrokerExecutor._await_coroutine static method."""

    def test_awaits_coroutine(self) -> None:
        """Test that coroutine is awaited correctly."""

        async def async_result():
            return "result"

        result = BrokerExecutor._await_coroutine(async_result())
        assert result == "result"

    def test_handles_async_exception(self) -> None:
        """Test that async exceptions are propagated."""

        async def async_error():
            raise ValueError("Async error")

        with pytest.raises(ValueError, match="Async error"):
            BrokerExecutor._await_coroutine(async_error())


# ============================================================
# Test: Integration scenarios
# ============================================================


class TestOrderSubmissionIntegration:
    """Integration tests for order submission workflows."""

    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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

    @patch("gpt_trader.orchestration.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.orchestration.execution.order_event_recorder.get_monitoring_logger")
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
