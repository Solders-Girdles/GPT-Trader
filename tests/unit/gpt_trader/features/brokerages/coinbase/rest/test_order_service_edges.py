"""Edge coverage for Coinbase REST OrderService error paths."""

from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.order_service as order_service_module
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.errors import BrokerageError, OrderQueryError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from gpt_trader.features.brokerages.coinbase.rest.protocols import (
    OrderPayloadBuilder,
    OrderPayloadExecutor,
    PositionProvider,
)


def _make_service() -> OrderService:
    client = Mock(spec=CoinbaseClient)
    payload_builder = Mock(spec=OrderPayloadBuilder)
    payload_executor = Mock(spec=OrderPayloadExecutor)
    position_provider = Mock(spec=PositionProvider)
    return OrderService(
        client=client,
        payload_builder=payload_builder,
        payload_executor=payload_executor,
        position_provider=position_provider,
    )


def test_cancel_order_brokerage_error_reraises() -> None:
    service = _make_service()
    service._client.cancel_orders.side_effect = BrokerageError("rate limited")

    with pytest.raises(BrokerageError, match="rate limited"):
        service.cancel_order("order-1")


def test_list_fills_exception_logs_and_raises_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _make_service()
    service._client.list_fills.side_effect = Exception("boom")
    mock_logger = MagicMock()
    monkeypatch.setattr(order_service_module, "logger", mock_logger)

    with pytest.raises(OrderQueryError, match="Failed to list fills"):
        service.list_fills()

    mock_logger.error.assert_called_once()
