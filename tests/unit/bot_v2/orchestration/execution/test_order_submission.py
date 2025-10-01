"""Tests for order submission"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch
from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderType
from bot_v2.orchestration.execution.order_submission import OrderSubmitter


@pytest.fixture
def mock_broker():
    """Mock broker"""
    broker = Mock()
    broker.place_order = Mock()
    return broker


@pytest.fixture
def mock_event_store():
    """Mock event store"""
    store = Mock()
    store.append_metric = Mock()
    store.append_trade = Mock()
    store.append_error = Mock()
    store._write = Mock()
    return store


@pytest.fixture
def open_orders_list():
    """Mock open orders list"""
    return []


@pytest.fixture
def order_submitter(mock_broker, mock_event_store, open_orders_list):
    """Create OrderSubmitter instance"""
    return OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test_bot",
        open_orders=open_orders_list,
    )


class TestOrderSubmitter:
    """Test suite for OrderSubmitter"""

    def test_initialization(self, order_submitter):
        """Test submitter initialization"""
        assert order_submitter.broker is not None
        assert order_submitter.event_store is not None
        assert order_submitter.bot_id == "test_bot"

    @patch("bot_v2.orchestration.execution.order_submission.get_logger")
    def test_record_preview_success(self, mock_get_logger, order_submitter, mock_event_store):
        """Test order preview recording"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        preview_data = {"estimated_fee": "5.00", "quote_size": "5000.00"}

        order_submitter.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            preview=preview_data,
        )

        mock_event_store.append_metric.assert_called_once()
        call_kwargs = mock_event_store.append_metric.call_args[1]
        assert call_kwargs["metrics"]["event_type"] == "order_preview"
        assert call_kwargs["metrics"]["symbol"] == "BTC-USD"

    def test_record_preview_none(self, order_submitter, mock_event_store):
        """Test recording None preview"""
        order_submitter.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            preview=None,
        )

        # Should not record anything
        mock_event_store.append_metric.assert_not_called()

    @patch("bot_v2.orchestration.execution.order_submission.get_logger")
    def test_record_rejection(self, mock_get_logger, order_submitter, mock_event_store):
        """Test order rejection recording"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        order_submitter.record_rejection(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            reason="insufficient_funds",
        )

        mock_event_store._write.assert_called_once()
        call_args = mock_event_store._write.call_args[0][0]
        assert call_args["type"] == "order_rejected"
        assert call_args["reason"] == "insufficient_funds"

    @patch("bot_v2.orchestration.execution.order_submission.get_logger")
    def test_submit_order_success(
        self,
        mock_get_logger,
        order_submitter,
        mock_broker,
        mock_event_store,
        open_orders_list,
    ):
        """Test successful order submission"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.side = OrderSide.BUY
        mock_order.quantity = Decimal("0.1")
        mock_order.price = Decimal("50000")
        mock_order.status = "SUBMITTED"
        mock_broker.place_order.return_value = mock_order

        order_id = order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("0.1"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        assert order_id == "order_123"
        assert "order_123" in open_orders_list
        mock_event_store.append_trade.assert_called_once()

    def test_submit_order_failure(self, order_submitter, mock_broker, mock_event_store):
        """Test failed order submission"""
        mock_broker.place_order.return_value = None

        order_id = order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        assert order_id is None
        mock_event_store.append_error.assert_called_once()

    def test_submit_order_with_custom_client_id(self, order_submitter, mock_broker):
        """Test order submission with custom client order ID"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.price = Decimal("50000")
        mock_broker.place_order.return_value = mock_order

        order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id="custom_client_123",
        )

        call_kwargs = mock_broker.place_order.call_args[1]
        assert call_kwargs["client_id"] == "custom_client_123"

    def test_submit_order_generates_client_id(self, order_submitter, mock_broker):
        """Test order submission generates client ID"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.price = None
        mock_broker.place_order.return_value = mock_order

        order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        call_kwargs = mock_broker.place_order.call_args[1]
        # Should generate client_id starting with bot_id
        assert call_kwargs["client_id"].startswith("test_bot_")

    @patch("bot_v2.orchestration.execution.order_submission.get_logger")
    def test_submit_order_reduce_only(self, mock_get_logger, order_submitter, mock_broker):
        """Test reduce-only order submission"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.price = Decimal("50000")
        mock_broker.place_order.return_value = mock_order

        order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
            client_order_id=None,
        )

        call_kwargs = mock_broker.place_order.call_args[1]
        assert call_kwargs["reduce_only"] is True

    def test_submit_order_with_stop_price(self, order_submitter, mock_broker):
        """Test order submission with stop price"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.price = Decimal("50000")
        mock_broker.place_order.return_value = mock_order

        order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            order_quantity=Decimal("0.1"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=Decimal("49000"),
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        call_kwargs = mock_broker.place_order.call_args[1]
        assert call_kwargs["stop_price"] == Decimal("49000")

    def test_submit_order_with_leverage(self, order_submitter, mock_broker):
        """Test order submission with leverage"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-PERP"
        mock_order.price = Decimal("50000")
        mock_broker.place_order.return_value = mock_order

        order_submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=5,
            client_order_id=None,
        )

        call_kwargs = mock_broker.place_order.call_args[1]
        assert call_kwargs["leverage"] == 5

    def test_submit_order_records_trade_payload(
        self, order_submitter, mock_broker, mock_event_store
    ):
        """Test trade payload recording"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.price = Decimal("50000")
        mock_order.quantity = Decimal("0.1")
        mock_order.status = "FILLED"
        mock_order.client_order_id = "client_123"
        mock_broker.place_order.return_value = mock_order

        order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id="client_123",
        )

        mock_event_store.append_trade.assert_called_once()
        call_args = mock_event_store.append_trade.call_args[0]
        trade_payload = call_args[1]
        assert trade_payload["order_id"] == "order_123"
        assert trade_payload["status"] == "FILLED"

    def test_record_preview_exception_handling(self, order_submitter, mock_event_store):
        """Test preview recording handles exceptions"""
        mock_event_store.append_metric.side_effect = Exception("Store error")

        # Should not raise
        order_submitter.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            preview={"test": "data"},
        )

    def test_record_rejection_exception_handling(self, order_submitter, mock_event_store):
        """Test rejection recording handles exceptions"""
        mock_event_store._write.side_effect = Exception("Store error")

        # Should not raise
        order_submitter.record_rejection(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            price=None,
            reason="test",
        )

    def test_submit_order_no_order_id(self, order_submitter, mock_broker):
        """Test submission when order has no ID"""
        mock_order = Mock(spec=Order)
        mock_order.id = None  # No ID
        mock_broker.place_order.return_value = mock_order

        order_id = order_submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.1"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        assert order_id is None
