"""Contract tests for Coinbase REST retry behavior and telemetry emission."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

from gpt_trader.core import InvalidRequestError, Order, OrderSide, OrderType
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuiteRetryAndTelemetry(CoinbaseRestContractSuiteBase):
    def test_order_retry_on_duplicate_client_id(
        self, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test order retry logic on duplicate client ID error."""
        mock_product_catalog.get.return_value = mock_product

        mock_client.place_order.side_effect = [
            InvalidRequestError("duplicate client_order_id"),
            {"order_id": "retry_success_123"},
        ]
        mock_client.list_orders.return_value = {"orders": []}

        with patch("gpt_trader.features.brokerages.coinbase.models.to_order") as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            order = order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
                client_id="duplicate_id",
            )

        assert order is not None
        assert mock_client.place_order.call_count == 2

    def test_telemetry_logging_on_api_calls(self, portfolio_service, mock_endpoints, mock_client):
        """Test telemetry logging on various API calls."""
        mock_endpoints.mode = "advanced"

        mock_client.intx_allocate.return_value = {"status": "success"}
        portfolio_service.intx_allocate({"amount": "1000"})

        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_balance_summary.return_value = {"balance_summary": {}}
        portfolio_service.get_cfm_balance_summary()

        assert portfolio_service._event_store.append_metric.call_count >= 2

    def test_pagination_cursor_handling(self, order_service, mock_client):
        """Test pagination cursor handling in list operations."""
        mock_client.list_fills.side_effect = [
            {"fills": [{"id": "1"}], "cursor": "page2"},
            {"fills": [{"id": "2"}]},
        ]

        fills = order_service.list_fills(limit=1)

        assert len(fills) == 2
        second_call = mock_client.list_fills.call_args_list[1]
        assert second_call[1]["cursor"] == "page2"
