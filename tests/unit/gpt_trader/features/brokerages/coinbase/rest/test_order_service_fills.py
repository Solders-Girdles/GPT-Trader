"""Tests for `OrderService.list_fills`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.order_service as order_service_module
from gpt_trader.features.brokerages.coinbase.errors import BrokerageError, OrderQueryError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestListFills(OrderServiceTestBase):
    def test_list_fills_returns_fills(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        fill_data = {
            "fill_id": "fill-123",
            "order_id": "order-123",
            "product_id": "BTC-USD",
            "price": "50000",
            "size": "0.1",
        }
        mock_client.list_fills.return_value = {"fills": [fill_data], "cursor": None}

        result = order_service.list_fills()

        assert len(result) == 1
        assert result[0]["fill_id"] == "fill-123"

    def test_list_fills_empty_response_returns_empty_list(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.list_fills.return_value = {"fills": []}

        result = order_service.list_fills()

        assert result == []

    @pytest.mark.parametrize(
        ("kwargs", "expected_call_kwargs"),
        [
            ({"product_id": "ETH-USD"}, {"product_id": "ETH-USD"}),
            ({"order_id": "order-456"}, {"order_id": "order-456"}),
        ],
    )
    def test_list_fills_passes_filters(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        kwargs: dict,
        expected_call_kwargs: dict,
    ) -> None:
        mock_client.list_fills.return_value = {"fills": [], "cursor": None}

        order_service.list_fills(**kwargs)

        call_kwargs = mock_client.list_fills.call_args.kwargs
        for key, value in expected_call_kwargs.items():
            assert call_kwargs[key] == value

    def test_list_fills_pagination(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.list_fills.side_effect = [
            {"fills": [{"fill_id": "fill-1"}], "cursor": "cursor-123"},
            {"fills": [{"fill_id": "fill-2"}], "cursor": None},
        ]

        result = order_service.list_fills()

        assert len(result) == 2
        assert mock_client.list_fills.call_count == 2

    def test_list_fills_handles_exception_logs_and_raises(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_client.list_fills.side_effect = RuntimeError("API error")
        mock_logger = MagicMock()
        monkeypatch.setattr(order_service_module, "logger", mock_logger)

        with pytest.raises(OrderQueryError, match="Failed to list fills"):
            order_service.list_fills()

        mock_logger.error.assert_called_once()

    def test_list_fills_re_raises_brokerage_error(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.list_fills.side_effect = BrokerageError("rate limited")

        with pytest.raises(BrokerageError, match="rate limited"):
            order_service.list_fills()
