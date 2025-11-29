"""Tests for Coinbase REST portfolio service functionality."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.features.brokerages.core.interfaces import InvalidRequestError
from gpt_trader.persistence.event_store import EventStore


class TestPortfolioService:
    """Test PortfolioService class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.client = Mock()  # Don't use spec to allow dynamic method mocking
        self.endpoints = Mock(spec=CoinbaseEndpoints)
        self.event_store = Mock(spec=EventStore)

        self.service = PortfolioService(
            client=self.client,
            endpoints=self.endpoints,
            event_store=self.event_store,
        )

    def test_service_init(self) -> None:
        """Test service initialization."""
        assert self.service._client == self.client
        assert self.service._endpoints == self.endpoints
        assert self.service._event_store == self.event_store

    def test_list_balances_returns_balances(self) -> None:
        """Test listing balances returns parsed balances."""
        self.client.get_accounts.return_value = {
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

        result = self.service.list_balances()

        assert len(result) == 2
        assert result[0].asset == "BTC"
        assert result[0].available == Decimal("1.5")
        assert result[0].hold == Decimal("0.1")
        assert result[0].total == Decimal("1.6")

    def test_list_balances_handles_list_response(self) -> None:
        """Test listing balances handles list response shape."""
        self.client.get_accounts.return_value = [
            {
                "uuid": "acc_1",
                "currency": "BTC",
                "available": "1.5",
                "hold": "0.1",
                "balance": "1.6",
            }
        ]

        result = self.service.list_balances()

        assert len(result) == 1
        assert result[0].asset == "BTC"

    def test_list_balances_calculates_total_when_missing(self) -> None:
        """Test list_balances calculates total from available + hold."""
        self.client.get_accounts.return_value = {
            "accounts": [
                {
                    "uuid": "acc_1",
                    "currency": "ETH",
                    "available_balance": {"value": "5.0"},
                    "hold": {"value": "1.0"},
                }
            ]
        }

        result = self.service.list_balances()

        assert result[0].total == Decimal("6.0")

    def test_list_balances_handles_exception(self) -> None:
        """Test list_balances handles API exception."""
        self.client.get_accounts.side_effect = Exception("API error")

        result = self.service.list_balances()

        assert result == []

    def test_list_balances_skips_invalid_entries(self) -> None:
        """Test list_balances skips entries that fail to parse."""
        self.client.get_accounts.return_value = {
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

        result = self.service.list_balances()

        # Should return at least the valid entry
        assert len(result) >= 1

    def test_get_portfolio_balances_delegates_to_list_balances(self) -> None:
        """Test get_portfolio_balances delegates to list_balances."""
        self.client.get_accounts.return_value = {"accounts": []}

        result = self.service.get_portfolio_balances()

        assert result == []
        self.client.get_accounts.assert_called_once()

    def test_list_positions_returns_positions(self) -> None:
        """Test listing positions when derivatives are supported."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.list_positions.return_value = {
            "positions": [
                {
                    "product_id": "BTC-PERP",
                    "side": "LONG",
                    "contracts": "0.5",
                    "entry_price": "50000.00",
                    "unrealized_pnl": "100.00",
                    "realized_pnl": "100.00",
                }
            ]
        }

        result = self.service.list_positions()

        assert len(result) == 1
        assert result[0].symbol == "BTC-PERP"

    def test_list_positions_returns_empty_when_derivatives_not_supported(self) -> None:
        """Test list_positions returns empty when derivatives not supported."""
        self.endpoints.supports_derivatives.return_value = False

        result = self.service.list_positions()

        assert result == []
        self.client.list_positions.assert_not_called()

    def test_list_positions_handles_exception(self) -> None:
        """Test list_positions handles API exception."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.list_positions.side_effect = Exception("API error")

        result = self.service.list_positions()

        assert result == []

    def test_get_position_returns_position(self) -> None:
        """Test getting a single position."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.get_position.return_value = {
            "product_id": "BTC-PERP",
            "side": "LONG",
            "contracts": "0.5",
            "entry_price": "50000.00",
            "unrealized_pnl": "100.00",
            "realized_pnl": "100.00",
        }

        result = self.service.get_position("BTC-PERP")

        assert result is not None
        assert result.symbol == "BTC-PERP"

    def test_get_position_returns_none_when_not_supported(self) -> None:
        """Test get_position returns None when derivatives not supported."""
        self.endpoints.supports_derivatives.return_value = False

        result = self.service.get_position("BTC-PERP")

        assert result is None

    def test_intx_allocate_requires_advanced_mode(self) -> None:
        """Test intx_allocate raises when not in advanced mode."""
        self.endpoints.mode = "exchange"

        with pytest.raises(InvalidRequestError, match="advanced mode"):
            self.service.intx_allocate({"amount": "1000"})

    def test_intx_allocate_success(self) -> None:
        """Test successful INTX allocation."""
        self.endpoints.mode = "advanced"
        self.client.intx_allocate.return_value = {
            "allocated_amount": "1000.00",
            "source_amount": "1000.00",
        }

        result = self.service.intx_allocate({"amount": "1000"})

        assert result["allocated_amount"] == Decimal("1000.00")
        self.event_store.append_metric.assert_called_once()

    def test_get_intx_balances_returns_empty_when_not_advanced(self) -> None:
        """Test get_intx_balances returns empty when not in advanced mode."""
        self.endpoints.mode = "exchange"

        result = self.service.get_intx_balances("portfolio_123")

        assert result == []

    def test_get_intx_balances_returns_balances(self) -> None:
        """Test get_intx_balances returns balances."""
        self.endpoints.mode = "advanced"
        self.client.get_intx_portfolio.return_value = {
            "balances": [
                {"currency": "BTC", "amount": "1.5", "hold": "0.1"},
                {"currency": "USD", "amount": "10000.00", "hold": "0"},
            ]
        }

        result = self.service.get_intx_balances("portfolio_123")

        assert len(result) == 2
        assert result[0]["amount"] == Decimal("1.5")

    def test_get_intx_portfolio_returns_empty_when_not_advanced(self) -> None:
        """Test get_intx_portfolio returns empty when not in advanced mode."""
        self.endpoints.mode = "exchange"

        result = self.service.get_intx_portfolio("portfolio_123")

        assert result == {}

    def test_get_intx_portfolio_success(self) -> None:
        """Test successful INTX portfolio retrieval."""
        self.endpoints.mode = "advanced"
        self.client.get_intx_portfolio.return_value = {
            "portfolio_id": "portfolio_123",
            "portfolio_value": "50000.00",
        }

        result = self.service.get_intx_portfolio("portfolio_123")

        assert result["portfolio_value"] == Decimal("50000.00")

    def test_list_intx_positions_returns_positions(self) -> None:
        """Test listing INTX positions."""
        self.endpoints.mode = "advanced"
        self.client.list_intx_positions.return_value = {
            "positions": [
                {
                    "product_id": "BTC-PERP",
                    "side": "LONG",
                    "number_of_contracts": "0.5",
                    "entry_vwap": {"value": "50000.00"},
                }
            ]
        }

        result = self.service.list_intx_positions("portfolio_123")

        assert len(result) == 1

    def test_get_cfm_balance_summary_returns_summary(self) -> None:
        """Test getting CFM balance summary."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.cfm_balance_summary.return_value = {
            "balance_summary": {
                "total_balance": "50000.00",
                "available_balance": "45000.00",
            }
        }

        result = self.service.get_cfm_balance_summary()

        assert result["total_balance"] == Decimal("50000.00")
        self.event_store.append_metric.assert_called_once()

    def test_get_cfm_balance_summary_returns_empty_when_not_supported(self) -> None:
        """Test CFM balance summary returns empty when derivatives not supported."""
        self.endpoints.supports_derivatives.return_value = False

        result = self.service.get_cfm_balance_summary()

        assert result == {}

    def test_list_cfm_sweeps_returns_sweeps(self) -> None:
        """Test listing CFM sweeps."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.cfm_sweeps.return_value = {
            "sweeps": [
                {"sweep_id": "sweep_1", "amount": "100.00"},
                {"sweep_id": "sweep_2", "amount": "200.00"},
            ]
        }

        result = self.service.list_cfm_sweeps()

        assert len(result) == 2
        assert result[0]["amount"] == Decimal("100.00")

    def test_get_cfm_sweeps_schedule_returns_schedule(self) -> None:
        """Test getting CFM sweeps schedule."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.cfm_sweeps_schedule.return_value = {
            "schedule": {"frequency": "daily", "time": "00:00"}
        }

        result = self.service.get_cfm_sweeps_schedule()

        assert result["frequency"] == "daily"

    def test_get_cfm_margin_window_returns_window(self) -> None:
        """Test getting CFM margin window."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.cfm_intraday_current_margin_window.return_value = {
            "margin_window": "INTRADAY",
            "leverage": "10",
        }

        result = self.service.get_cfm_margin_window()

        assert result["margin_window"] == "INTRADAY"

    def test_update_cfm_margin_window_success(self) -> None:
        """Test updating CFM margin window."""
        self.endpoints.supports_derivatives.return_value = True
        self.client.cfm_intraday_margin_setting.return_value = {
            "margin_window": "OVERNIGHT",
            "leverage": "5",
        }

        result = self.service.update_cfm_margin_window("OVERNIGHT")

        assert result["leverage"] == Decimal("5")
        self.event_store.append_metric.assert_called_once()

    def test_update_cfm_margin_window_raises_when_not_supported(self) -> None:
        """Test update_cfm_margin_window raises when derivatives not supported."""
        self.endpoints.supports_derivatives.return_value = False

        with pytest.raises(InvalidRequestError, match="Derivatives not supported"):
            self.service.update_cfm_margin_window("OVERNIGHT")
