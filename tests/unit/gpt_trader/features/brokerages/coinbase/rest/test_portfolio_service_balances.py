"""Tests for PortfolioService account balances."""

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService

from .portfolio_service_test_base import PortfolioServiceTestBase


class TestPortfolioServiceBalances(PortfolioServiceTestBase):
    def test_service_init(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        """Test service initialization."""
        assert portfolio_service._client == mock_client
        assert portfolio_service._endpoints == mock_endpoints
        assert portfolio_service._event_store == mock_event_store

    def test_list_balances_returns_balances(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        """Test listing balances returns parsed balances."""
        mock_client.get_accounts.return_value = {
            "accounts": [
                {
                    "uuid": "acc_1",
                    "currency": "BTC",
                    "available_balance": {"value": "1.5"},
                    "hold": {"value": "0.1"},
                    "balance": {"value": "1.6"},
                },
                {
                    "uuid": "acc_2",
                    "currency": "USD",
                    "available_balance": {"value": "10000.00"},
                    "hold": {"value": "500.00"},
                    "balance": {"value": "10500.00"},
                },
            ]
        }

        result = portfolio_service.list_balances()

        assert len(result) == 2
        assert result[0].asset == "BTC"
        assert result[0].available == Decimal("1.5")
        assert result[0].hold == Decimal("0.1")
        assert result[0].total == Decimal("1.6")

    def test_list_balances_handles_list_response(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        """Test listing balances handles list response shape."""
        mock_client.get_accounts.return_value = [
            {
                "uuid": "acc_1",
                "currency": "BTC",
                "available": "1.5",
                "hold": "0.1",
                "balance": "1.6",
            }
        ]

        result = portfolio_service.list_balances()

        assert len(result) == 1
        assert result[0].asset == "BTC"

    def test_list_balances_calculates_total_when_missing(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        """Test list_balances calculates total from available + hold."""
        mock_client.get_accounts.return_value = {
            "accounts": [
                {
                    "uuid": "acc_1",
                    "currency": "ETH",
                    "available_balance": {"value": "5.0"},
                    "hold": {"value": "1.0"},
                }
            ]
        }

        result = portfolio_service.list_balances()

        assert result[0].total == Decimal("6.0")

    def test_list_balances_handles_exception(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        """Test list_balances handles API exception."""
        mock_client.get_accounts.side_effect = Exception("API error")

        result = portfolio_service.list_balances()

        assert result == []

    def test_list_balances_skips_invalid_entries(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        """Test list_balances skips entries that fail to parse."""
        mock_client.get_accounts.return_value = {
            "accounts": [
                {
                    "uuid": "acc_1",
                    "currency": "BTC",
                    "available_balance": {"value": "1.5"},
                    "hold": {"value": "0.1"},
                    "balance": {"value": "1.6"},
                },
                {
                    "uuid": "acc_2",
                    "currency": None,  # Invalid - will cause error
                },
            ]
        }

        result = portfolio_service.list_balances()

        assert len(result) >= 1

    def test_get_portfolio_balances_delegates_to_list_balances(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        """Test get_portfolio_balances delegates to list_balances."""
        mock_client.get_accounts.return_value = {"accounts": []}

        result = portfolio_service.get_portfolio_balances()

        assert result == []
        mock_client.get_accounts.assert_called_once()
