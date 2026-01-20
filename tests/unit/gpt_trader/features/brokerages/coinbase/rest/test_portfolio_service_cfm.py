"""Tests for PortfolioService CFM (Coinbase Financial Markets) operations."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.core import InvalidRequestError
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService

from .portfolio_service_test_base import PortfolioServiceTestBase


class TestCfmBalanceSummary(PortfolioServiceTestBase):
    """Tests for CFM balance summary operations."""

    def test_get_cfm_balance_summary_returns_summary(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_balance_summary.return_value = {
            "balance_summary": {"total_balance": "50000.00", "available_balance": "45000.00"}
        }
        result = portfolio_service.get_cfm_balance_summary()
        assert result["total_balance"] == Decimal("50000.00")
        mock_event_store.append_metric.assert_called_once()

    def test_get_cfm_balance_summary_returns_empty_when_not_supported(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = False
        result = portfolio_service.get_cfm_balance_summary()
        assert result == {}

    def test_get_cfm_balance_summary_normalises_decimals_and_emits_metric(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_balance_summary.return_value = {
            "balance_summary": {
                "portfolio_value": "100.50",
                "available_margin": "25.75",
                "timestamp": "2024-05-01T00:00:00Z",
            }
        }
        summary = portfolio_service.get_cfm_balance_summary()
        assert summary["portfolio_value"] == Decimal("100.50")
        assert summary["available_margin"] == Decimal("25.75")
        mock_event_store.append_metric.assert_called_once()
        metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
        assert metrics_payload["event_type"] == "cfm_balance_summary"


class TestCfmSweeps(PortfolioServiceTestBase):
    """Tests for CFM sweeps operations."""

    def test_list_cfm_sweeps_returns_sweeps(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_sweeps.return_value = {
            "sweeps": [
                {"sweep_id": "sweep_1", "amount": "100.00"},
                {"sweep_id": "sweep_2", "amount": "200.00"},
            ]
        }
        result = portfolio_service.list_cfm_sweeps()
        assert len(result) == 2
        assert result[0]["amount"] == Decimal("100.00")

    def test_list_cfm_sweeps_returns_empty_when_derivatives_disabled(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = False
        sweeps = portfolio_service.list_cfm_sweeps()
        assert sweeps == []
        mock_client.cfm_sweeps.assert_not_called()

    def test_list_cfm_sweeps_normalises_entries(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_sweeps.return_value = {
            "sweeps": [{"sweep_id": "sw1", "amount": "10.0"}, {"sweep_id": "sw2", "amount": "5.5"}]
        }
        sweeps = portfolio_service.list_cfm_sweeps()
        assert sweeps[0]["amount"] == Decimal("10.0")
        assert sweeps[1]["amount"] == Decimal("5.5")
        assert mock_event_store.append_metric.called

    def test_get_cfm_sweeps_schedule_returns_schedule(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_sweeps_schedule.return_value = {
            "schedule": {"frequency": "daily", "time": "00:00"}
        }
        result = portfolio_service.get_cfm_sweeps_schedule()
        assert result["frequency"] == "daily"


class TestCfmMarginWindow(PortfolioServiceTestBase):
    """Tests for CFM margin window operations."""

    def test_get_cfm_margin_window_returns_window(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_intraday_current_margin_window.return_value = {
            "margin_window": "INTRADAY",
            "leverage": "10",
        }
        result = portfolio_service.get_cfm_margin_window()
        assert result["margin_window"] == "INTRADAY"

    def test_get_cfm_margin_window_handles_errors(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_intraday_current_margin_window.side_effect = RuntimeError("boom")
        result = portfolio_service.get_cfm_margin_window()
        assert result == {}

    def test_update_cfm_margin_window_success(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_intraday_margin_setting.return_value = {
            "margin_window": "OVERNIGHT",
            "leverage": "5",
        }
        result = portfolio_service.update_cfm_margin_window("OVERNIGHT")
        assert result["leverage"] == Decimal("5")
        mock_event_store.append_metric.assert_called_once()

    def test_update_cfm_margin_window_raises_when_not_supported(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = False
        with pytest.raises(InvalidRequestError, match="Derivatives not supported"):
            portfolio_service.update_cfm_margin_window("OVERNIGHT")

    def test_update_cfm_margin_window_enforces_derivatives(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = False
        with pytest.raises(InvalidRequestError):
            portfolio_service.update_cfm_margin_window("INTRADAY_STANDARD")

    def test_update_cfm_margin_window_calls_client_and_emits(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> None:
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_intraday_margin_setting.return_value = {
            "status": "accepted",
            "leverage": "3",
        }
        response = portfolio_service.update_cfm_margin_window(
            "INTRADAY_STANDARD", effective_time="2024-05-01T00:00:00Z"
        )
        mock_client.cfm_intraday_margin_setting.assert_called_once()
        payload = mock_client.cfm_intraday_margin_setting.call_args.args[0]
        assert payload["margin_window"] == "INTRADAY_STANDARD"
        assert payload["effective_time"] == "2024-05-01T00:00:00Z"
        assert response["leverage"] == Decimal("3")
        metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
        assert metrics_payload["event_type"] == "cfm_margin_setting"
