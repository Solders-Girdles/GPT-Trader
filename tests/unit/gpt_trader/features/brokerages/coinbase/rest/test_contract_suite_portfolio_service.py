"""Contract tests for Coinbase REST PortfolioService behavior."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.core import Balance, Position
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuitePortfolioService(CoinbaseRestContractSuiteBase):
    def test_list_balances_success(self, portfolio_service, mock_client):
        """Test successful balance listing."""
        mock_accounts = [
            {
                "currency": "BTC",
                "available_balance": {"value": "1.5"},
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            },
            {
                "currency": "USD",
                "available_balance": {"value": "1000.00"},
                "hold": {"value": "0.00"},
                "balance": {"value": "1000.00"},
            },
        ]
        mock_client.get_accounts.return_value = {"accounts": mock_accounts}

        balances = portfolio_service.list_balances()

        assert len(balances) == 2
        btc_balance = next(b for b in balances if b.asset == "BTC")
        assert btc_balance.total == Decimal("1.6")
        assert btc_balance.available == Decimal("1.5")
        assert btc_balance.hold == Decimal("0.1")

    def test_list_balances_error_handling(self, portfolio_service, mock_client):
        """Test balance listing with malformed data."""
        mock_accounts = [
            {
                "currency": "BTC",
                "available_balance": "invalid",
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            }
        ]
        mock_client.get_accounts.return_value = {"accounts": mock_accounts}

        balances = portfolio_service.list_balances()

        assert len(balances) == 0

    def test_get_portfolio_balances_fallback(self, portfolio_service, mock_client):
        """Test portfolio balances fallback to account balances."""
        mock_client.get_accounts.return_value = {"accounts": []}

        with patch.object(portfolio_service, "list_balances") as mock_list_balances:
            mock_list_balances.return_value = [Mock(spec=Balance)]
            _ = portfolio_service.get_portfolio_balances()

        mock_list_balances.assert_called_once()

    def test_list_positions_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful position listing."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_positions = [
            Position(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.5"),
                entry_price=Decimal("50000.00"),
                mark_price=Decimal("51000.00"),
                unrealized_pnl=Decimal("750.00"),
                realized_pnl=Decimal("0.00"),
                leverage=5,
            )
        ]
        mock_client.list_positions.return_value = mock_positions

        positions = portfolio_service.list_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "BTC-USD"
        assert pos.quantity == Decimal("1.5")
        assert pos.side == "long"

    def test_list_positions_error_handling(self, portfolio_service, mock_endpoints, mock_client):
        """Test position listing error handling."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.list_positions.side_effect = Exception("API error")

        positions = portfolio_service.list_positions()

        assert positions == []

    def test_get_position_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful position retrieval."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_position = {
            "product_id": "BTC-USD",
            "side": "long",
            "size": "1.0",
            "entry_price": "50000.00",
        }
        mock_client.get_cfm_position.return_value = mock_position

        position = portfolio_service.get_position("BTC-USD")

        assert position is not None
        assert position.symbol == "BTC-USD"

    def test_get_position_not_found(self, portfolio_service, mock_endpoints, mock_client):
        """Test position retrieval when not found."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.get_cfm_position.side_effect = Exception("Position not found")

        position = portfolio_service.get_position("BTC-USD")

        assert position is None

    def test_intx_allocate_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful INTX allocation."""
        mock_endpoints.mode = "advanced"
        mock_response = {"allocation_id": "alloc_123", "status": "confirmed"}
        mock_client.intx_allocate.return_value = mock_response

        result = portfolio_service.intx_allocate({"amount": "1000", "currency": "USD"})

        assert result["allocation_id"] == "alloc_123"
        portfolio_service._event_store.append_metric.assert_called()

    def test_intx_allocate_error_handling(self, portfolio_service, mock_endpoints, mock_client):
        """Test INTX allocation error handling."""
        mock_endpoints.mode = "advanced"
        mock_client.intx_allocate.side_effect = Exception("Allocation failed")

        with pytest.raises(Exception, match="Allocation failed"):
            portfolio_service.intx_allocate({"amount": "1000"})

    def test_cfm_balance_summary_telemetry(self, portfolio_service, mock_endpoints, mock_client):
        """Test CFM balance summary with telemetry."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_summary = {
            "total_balance": "10000.00",
            "available_balance": "9500.00",
            "timestamp": "2024-01-01T12:00:00Z",
        }
        mock_client.cfm_balance_summary.return_value = {"balance_summary": mock_summary}

        result = portfolio_service.get_cfm_balance_summary()

        assert result["total_balance"] == Decimal("10000.00")
        portfolio_service._event_store.append_metric.assert_called()

    def test_cfm_sweeps_telemetry(self, portfolio_service, mock_endpoints, mock_client):
        """Test CFM sweeps with telemetry."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_sweeps = [
            {"sweep_id": "sweep_1", "amount": "100.00", "status": "completed"},
            {"sweep_id": "sweep_2", "amount": "200.00", "status": "pending"},
        ]
        mock_client.cfm_sweeps.return_value = {"sweeps": mock_sweeps}

        result = portfolio_service.list_cfm_sweeps()

        assert len(result) == 2
        assert result[0]["amount"] == Decimal("100.00")
        portfolio_service._event_store.append_metric.assert_called()

    def test_update_cfm_margin_window_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful CFM margin window update."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_response = {"margin_window": "maintenance", "effective_time": "2024-01-01T12:00:00Z"}
        mock_client.cfm_intraday_margin_setting.return_value = mock_response

        result = portfolio_service.update_cfm_margin_window("maintenance")

        assert result["margin_window"] == "maintenance"
        portfolio_service._event_store.append_metric.assert_called()
