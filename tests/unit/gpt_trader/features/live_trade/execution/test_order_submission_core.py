"""Core unit tests for OrderSubmitter init, ID generation, and broker execution."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


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

    def test_integration_mode_uses_env_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that integration mode uses environment override."""
        monkeypatch.setenv("INTEGRATION_TEST_ORDER_ID", "forced-id")
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        result = submitter._generate_submit_id(None)
        assert result == "forced-id"


class TestExecuteBrokerOrder:
    """Tests for _execute_broker_order method."""

    def test_passes_retry_toggle_to_broker_executor(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )
        submitter._broker_executor.execute_order = MagicMock(return_value=mock_order)

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

        assert submitter._broker_executor.execute_order.call_args.kwargs["use_retry"] is False

        submitter_with_retries = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            enable_retries=True,
        )
        submitter_with_retries._broker_executor.execute_order = MagicMock(return_value=mock_order)

        submitter_with_retries._execute_broker_order(
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

        assert (
            submitter_with_retries._broker_executor.execute_order.call_args.kwargs["use_retry"]
            is True
        )

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
            stop_price=Decimal("49000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        assert result is mock_order
        mock_broker.place_order.assert_called_once()
        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["quantity"] == Decimal("1.0")
        assert call_kwargs["stop_price"] == Decimal("49000")
        assert call_kwargs["tif"] == TimeInForce.GTC

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
