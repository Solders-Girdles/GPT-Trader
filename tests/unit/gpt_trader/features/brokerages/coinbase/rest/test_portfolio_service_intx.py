"""Tests for PortfolioService INTX operations."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.core import InvalidRequestError
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService

from .portfolio_service_test_base import PortfolioServiceTestBase


class TestPortfolioServiceIntx(PortfolioServiceTestBase):
    def test_intx_allocate_requires_advanced_mode(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        """Test intx_allocate raises when not in advanced mode."""
        mock_endpoints.mode = "exchange"

        with pytest.raises(InvalidRequestError, match="advanced mode"):
            portfolio_service.intx_allocate({"amount": "1000"})

    def test_intx_allocate_success(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        """Test successful INTX allocation."""
        mock_endpoints.mode = "advanced"
        mock_client.intx_allocate.return_value = {
            "allocated_amount": "1000.00",
            "source_amount": "1000.00",
        }

        result = portfolio_service.intx_allocate({"amount": "1000"})

        assert result["allocated_amount"] == Decimal("1000.00")
        mock_event_store.append_metric.assert_called_once()

    def test_get_intx_balances_returns_empty_when_not_advanced(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        """Test get_intx_balances returns empty when not in advanced mode."""
        mock_endpoints.mode = "exchange"

        result = portfolio_service.get_intx_balances("portfolio_123")

        assert result == []

    def test_get_intx_balances_returns_balances(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test get_intx_balances returns balances."""
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_portfolio.return_value = {
            "balances": [
                {"currency": "BTC", "amount": "1.5", "hold": "0.1"},
                {"currency": "USD", "amount": "10000.00", "hold": "0"},
            ]
        }

        result = portfolio_service.get_intx_balances("portfolio_123")

        assert len(result) == 2
        assert result[0]["amount"] == Decimal("1.5")

    def test_get_intx_portfolio_returns_empty_when_not_advanced(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        """Test get_intx_portfolio returns empty when not in advanced mode."""
        mock_endpoints.mode = "exchange"

        result = portfolio_service.get_intx_portfolio("portfolio_123")

        assert result == {}

    def test_get_intx_portfolio_success(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test successful INTX portfolio retrieval."""
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_portfolio.return_value = {
            "portfolio_id": "portfolio_123",
            "portfolio_value": "50000.00",
        }

        result = portfolio_service.get_intx_portfolio("portfolio_123")

        assert result["portfolio_value"] == Decimal("50000.00")

    def test_list_intx_positions_returns_positions(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test listing INTX positions."""
        mock_endpoints.mode = "advanced"
        mock_client.list_intx_positions.return_value = {
            "positions": [
                {
                    "product_id": "BTC-PERP",
                    "side": "LONG",
                    "number_of_contracts": "0.5",
                    "entry_vwap": {"value": "50000.00"},
                }
            ]
        }

        result = portfolio_service.list_intx_positions("portfolio_123")

        assert len(result) == 1
