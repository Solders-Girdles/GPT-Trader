"""Tests for PortfolioService INTX (International Exchange) operations."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.core import InvalidRequestError
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService

from .portfolio_service_test_base import PortfolioServiceTestBase


class TestIntxAllocate(PortfolioServiceTestBase):
    """Tests for INTX allocation operations."""

    def test_intx_allocate_requires_advanced_mode(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
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
        mock_endpoints.mode = "advanced"
        mock_client.intx_allocate.return_value = {
            "allocated_amount": "1000.00",
            "source_amount": "1000.00",
        }
        result = portfolio_service.intx_allocate({"amount": "1000"})
        assert result["allocated_amount"] == Decimal("1000.00")
        mock_event_store.append_metric.assert_called_once()

    def test_intx_allocate_normalises_and_emits_metric(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.intx_allocate.return_value = {"status": "ok", "allocated_amount": "10.5"}
        response = portfolio_service.intx_allocate({"allocated_amount": "10.5"})
        assert response["allocated_amount"] == Decimal("10.5")
        mock_event_store.append_metric.assert_called_once()
        metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
        assert metrics_payload["event_type"] == "intx_allocation"


class TestIntxBalances(PortfolioServiceTestBase):
    """Tests for INTX balance operations."""

    def test_get_intx_balances_returns_empty_when_not_advanced(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.mode = "exchange"
        result = portfolio_service.get_intx_balances("portfolio_123")
        assert result == []

    def test_get_intx_balances_returns_balances(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
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

    def test_get_intx_balances_normalises_entries(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_portfolio.return_value = {
            "balances": [
                {"asset": "USD", "amount": "100.5", "hold": "0"},
                {"asset": "BTC", "amount": "0.25", "hold": "0"},
            ]
        }
        balances = portfolio_service.get_intx_balances("pf-1")
        assert balances[0]["amount"] == Decimal("100.5")
        assert balances[1]["amount"] == Decimal("0.25")
        assert mock_event_store.append_metric.called

    def test_get_intx_balances_handles_errors(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_portfolio.side_effect = RuntimeError("boom")
        balances = portfolio_service.get_intx_balances("pf-1")
        assert balances == []
        mock_event_store.append_metric.assert_not_called()


class TestIntxPortfolio(PortfolioServiceTestBase):
    """Tests for INTX portfolio operations."""

    def test_get_intx_portfolio_returns_empty_when_not_advanced(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.mode = "exchange"
        result = portfolio_service.get_intx_portfolio("portfolio_123")
        assert result == {}

    def test_get_intx_portfolio_success(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_portfolio.return_value = {
            "portfolio_id": "portfolio_123",
            "portfolio_value": "50000.00",
        }
        result = portfolio_service.get_intx_portfolio("portfolio_123")
        assert result["portfolio_value"] == Decimal("50000.00")

    def test_get_intx_portfolio_returns_normalised_dict(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_portfolio.return_value = {"portfolio_value": "2500.75"}
        portfolio = portfolio_service.get_intx_portfolio("pf-1")
        assert portfolio["portfolio_value"] == Decimal("2500.75")


class TestIntxPositions(PortfolioServiceTestBase):
    """Tests for INTX position operations."""

    def test_list_intx_positions_returns_positions(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
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

    def test_list_intx_positions_returns_normalised_list(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.list_intx_positions.return_value = {
            "positions": [
                {
                    "product_id": "BTC-USD",
                    "quantity": "1.5",
                    "side": "long",
                    "entry_price": "100",
                    "mark_price": "101",
                    "unrealized_pnl": "1",
                    "realized_pnl": "1",
                }
            ]
        }
        positions = portfolio_service.list_intx_positions("pf-1")
        assert positions[0].quantity == Decimal("1.5")

    def test_get_intx_position_handles_missing(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_position.side_effect = RuntimeError("no position")
        position = portfolio_service.get_intx_position("pf-1", "BTC-USD")
        assert position is None


class TestIntxCollateral(PortfolioServiceTestBase):
    """Tests for INTX multi-asset collateral operations."""

    def test_get_intx_multi_asset_collateral_emits_metric(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.mode = "advanced"
        mock_client.get_intx_multi_asset_collateral.return_value = {"total_usd_value": "5000.25"}
        collateral = portfolio_service.get_intx_multi_asset_collateral()
        assert collateral["total_usd_value"] == Decimal("5000.25")
        assert mock_event_store.append_metric.called
        metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
        assert metrics_payload["event_type"] == "intx_multi_asset_collateral"
