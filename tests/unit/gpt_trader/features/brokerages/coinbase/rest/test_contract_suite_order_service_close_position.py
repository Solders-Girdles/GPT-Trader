"""Contract tests for Coinbase REST OrderService close-position behavior."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.core import Order, Position
from gpt_trader.errors import ValidationError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuiteOrderServiceClosePosition(CoinbaseRestContractSuiteBase):
    def test_close_position_success(self, portfolio_service, order_service, mock_client):
        """Test successful position closing."""
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")

        with patch.object(portfolio_service, "list_positions", return_value=[mock_position]):
            mock_client.close_position.return_value = {"order": {"order_id": "close_123"}}

            with patch(
                "gpt_trader.features.brokerages.coinbase.rest.order_service.to_order"
            ) as mock_to_order:
                mock_to_order.return_value = Mock(spec=Order)
                order = order_service.close_position("BTC-USD")

            assert order is not None

    def test_close_position_no_position(self, portfolio_service, order_service):
        """Test position closing when no position exists."""
        with patch.object(portfolio_service, "list_positions", return_value=[]):
            with pytest.raises(ValidationError, match="No open position"):
                order_service.close_position("BTC-USD")

    def test_close_position_fallback(
        self, portfolio_service, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test position closing with fallback when API fails."""
        mock_product_catalog.get.return_value = mock_product

        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")

        with patch.object(portfolio_service, "list_positions", return_value=[mock_position]):
            mock_client.close_position.side_effect = Exception("API failed")
            fallback_order = Mock(spec=Order)
            fallback_func = Mock(return_value=fallback_order)

            order = order_service.close_position("BTC-USD", fallback=fallback_func)

            assert order == fallback_order
            fallback_func.assert_called_once()
