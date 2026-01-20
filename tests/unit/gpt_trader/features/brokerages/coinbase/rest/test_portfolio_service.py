"""Tests for PortfolioService core: balances and positions."""

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.core import Position
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService

from .portfolio_service_test_base import PortfolioServiceTestBase


class TestPortfolioServiceInit(PortfolioServiceTestBase):
    """Test service initialization."""

    def test_service_init(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        assert portfolio_service._client == mock_client
        assert portfolio_service._endpoints == mock_endpoints
        assert portfolio_service._event_store == mock_event_store


class TestPortfolioServiceBalances(PortfolioServiceTestBase):
    """Tests for balance-related operations."""

    def test_list_balances_returns_balances(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
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
        mock_client.get_accounts.side_effect = Exception("API error")
        result = portfolio_service.list_balances()
        assert result == []

    def test_list_balances_skips_invalid_entries(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        mock_client.get_accounts.return_value = {
            "accounts": [
                {
                    "uuid": "acc_1",
                    "currency": "BTC",
                    "available_balance": {"value": "1.5"},
                    "hold": {"value": "0.1"},
                    "balance": {"value": "1.6"},
                },
                {"uuid": "acc_2", "currency": None},
            ]
        }
        result = portfolio_service.list_balances()
        assert len(result) >= 1

    def test_get_portfolio_balances_delegates_to_list_balances(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
    ) -> None:
        mock_client.get_accounts.return_value = {"accounts": []}
        result = portfolio_service.get_portfolio_balances()
        assert result == []
        mock_client.get_accounts.assert_called_once()


class TestPortfolioServicePositions(PortfolioServiceTestBase):
    """Tests for position-related operations."""

    def test_list_positions_returns_positions(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.list_positions.return_value = [
            Position(
                symbol="BTC-PERP",
                quantity=Decimal("0.5"),
                entry_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
                mark_price=Decimal("51000.00"),
                realized_pnl=Decimal("0.00"),
                side="LONG",
            )
        ]
        result = portfolio_service.list_positions()
        assert len(result) == 1
        assert result[0].symbol == "BTC-PERP"

    def test_list_positions_returns_empty_when_derivatives_not_supported(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = False
        result = portfolio_service.list_positions()
        assert result == []
        mock_client.list_positions.assert_not_called()

    def test_list_positions_handles_exception(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.list_positions.side_effect = Exception("API error")
        result = portfolio_service.list_positions()
        assert result == []

    def test_get_position_returns_position(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.get_cfm_position.return_value = {
            "product_id": "BTC-PERP",
            "side": "LONG",
            "contracts": "0.5",
            "entry_price": "50000.00",
            "unrealized_pnl": "100.00",
            "realized_pnl": "100.00",
        }
        result = portfolio_service.get_position("BTC-PERP")
        assert result is not None
        assert result.symbol == "BTC-PERP"

    def test_get_position_returns_none_when_not_supported(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = False
        result = portfolio_service.get_position("BTC-PERP")
        assert result is None
