"""Tests for `OrderService.list_orders` and `OrderService.get_order`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order
from gpt_trader.features.brokerages.coinbase.errors import (
    BrokerageError,
    NotFoundError,
    OrderQueryError,
)
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestListOrders(OrderServiceTestBase):
    def test_list_orders_returns_orders(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        mock_client.list_orders.return_value = {
            "orders": [sample_order_response],
            "cursor": None,
        }

        result = order_service.list_orders()

        assert len(result) == 1
        assert isinstance(result[0], Order)

    def test_list_orders_empty_response_returns_empty_list(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.list_orders.return_value = {"orders": []}

        result = order_service.list_orders()

        assert result == []

    @pytest.mark.parametrize(
        ("kwargs", "expected_call_kwargs"),
        [
            ({"product_id": "BTC-USD"}, {"product_id": "BTC-USD"}),
            ({"status": ["PENDING", "OPEN"]}, {"order_status": ["PENDING", "OPEN"]}),
            ({"limit": 50}, {"limit": 50}),
        ],
    )
    def test_list_orders_passes_filters_and_limit(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        kwargs: dict,
        expected_call_kwargs: dict,
    ) -> None:
        mock_client.list_orders.return_value = {"orders": [], "cursor": None}

        order_service.list_orders(**kwargs)

        call_kwargs = mock_client.list_orders.call_args.kwargs
        for key, value in expected_call_kwargs.items():
            assert call_kwargs[key] == value

    def test_list_orders_pagination(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        page1_response = sample_order_response.copy()
        page1_response["order_id"] = "order-1"
        page2_response = sample_order_response.copy()
        page2_response["order_id"] = "order-2"

        mock_client.list_orders.side_effect = [
            {"orders": [page1_response], "cursor": "cursor-123"},
            {"orders": [page2_response], "cursor": None},
        ]

        result = order_service.list_orders()

        assert len(result) == 2
        assert mock_client.list_orders.call_count == 2

    def test_list_orders_handles_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.list_orders.side_effect = RuntimeError("API error")

        with pytest.raises(OrderQueryError, match="Failed to list orders"):
            order_service.list_orders()

    def test_list_orders_re_raises_brokerage_error(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.list_orders.side_effect = BrokerageError("rate limited")

        with pytest.raises(BrokerageError, match="rate limited"):
            order_service.list_orders()


class TestGetOrder(OrderServiceTestBase):
    def test_get_order_returns_order(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        mock_client.get_order_historical.return_value = {"order": sample_order_response}

        result = order_service.get_order("order-123")

        assert isinstance(result, Order)
        mock_client.get_order_historical.assert_called_once_with("order-123")

    def test_get_order_returns_none_when_order_missing(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.get_order_historical.return_value = {"order": None}

        result = order_service.get_order("nonexistent-order")

        assert result is None

    def test_get_order_not_found_error_returns_none(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.get_order_historical.side_effect = NotFoundError("missing")

        result = order_service.get_order("missing")

        assert result is None

    def test_get_order_handles_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.get_order_historical.side_effect = RuntimeError("API error")

        with pytest.raises(OrderQueryError, match="Failed to get order"):
            order_service.get_order("order-123")

    def test_get_order_re_raises_brokerage_error(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.get_order_historical.side_effect = BrokerageError("auth failed")

        with pytest.raises(BrokerageError, match="auth failed"):
            order_service.get_order("order-123")
