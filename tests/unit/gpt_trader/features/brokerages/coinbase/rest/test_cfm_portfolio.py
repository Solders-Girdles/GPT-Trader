"""Tests for CFM telemetry helpers in the Coinbase portfolio REST mixin."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin
from gpt_trader.features.brokerages.core.interfaces import InvalidRequestError


class DummyPortfolioService(PortfolioRestMixin):
    def __init__(self, client, endpoints, event_store):
        self.client = client
        self.endpoints = endpoints
        self._event_store = event_store


@pytest.fixture()
def service():
    client = Mock()
    endpoints = Mock()
    event_store = Mock()
    event_store.append_metric = Mock()
    svc = DummyPortfolioService(client, endpoints, event_store)  # naming: allow
    return svc, client, endpoints, event_store  # naming: allow


def test_get_cfm_balance_summary_normalises_decimals_and_emits_metric(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.supports_derivatives.return_value = True
    client.cfm_balance_summary.return_value = {
        "balance_summary": {
            "portfolio_value": "100.50",
            "available_margin": "25.75",
            "timestamp": "2024-05-01T00:00:00Z",
        }
    }

    summary = svc.get_cfm_balance_summary()  # naming: allow

    assert summary["portfolio_value"] == Decimal("100.50")
    assert summary["available_margin"] == Decimal("25.75")
    event_store.append_metric.assert_called_once()
    metrics_payload = event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "cfm_balance_summary"
    assert metrics_payload["summary"]["portfolio_value"] == "100.50"


def test_list_cfm_sweeps_returns_empty_when_derivatives_disabled(service):
    svc, client, endpoints, _ = service  # naming: allow
    endpoints.supports_derivatives.return_value = False

    sweeps = svc.list_cfm_sweeps()  # naming: allow

    assert sweeps == []
    client.cfm_sweeps.assert_not_called()


def test_list_cfm_sweeps_normalises_entries(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.supports_derivatives.return_value = True
    client.cfm_sweeps.return_value = {
        "sweeps": [
            {"sweep_id": "sw1", "amount": "10.0"},
            {"sweep_id": "sw2", "amount": "5.5"},
        ]
    }

    sweeps = svc.list_cfm_sweeps()  # naming: allow

    assert sweeps[0]["amount"] == Decimal("10.0")
    assert sweeps[1]["amount"] == Decimal("5.5")
    assert event_store.append_metric.called


def test_get_cfm_margin_window_handles_errors(service):
    svc, client, endpoints, _ = service  # naming: allow
    endpoints.supports_derivatives.return_value = True
    client.cfm_intraday_current_margin_window.side_effect = RuntimeError("boom")

    result = svc.get_cfm_margin_window()  # naming: allow

    assert result == {}


def test_update_cfm_margin_window_enforces_derivatives(service):
    svc, _, endpoints, _ = service  # naming: allow
    endpoints.supports_derivatives.return_value = False

    with pytest.raises(InvalidRequestError):
        svc.update_cfm_margin_window("INTRADAY_STANDARD")  # naming: allow


def test_update_cfm_margin_window_calls_client_and_emits(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.supports_derivatives.return_value = True
    client.cfm_intraday_margin_setting.return_value = {"status": "accepted", "leverage": "3"}

    response = svc.update_cfm_margin_window(  # naming: allow
        "INTRADAY_STANDARD", effective_time="2024-05-01T00:00:00Z"
    )

    client.cfm_intraday_margin_setting.assert_called_once()
    payload = client.cfm_intraday_margin_setting.call_args.args[0]
    assert payload["margin_window"] == "INTRADAY_STANDARD"
    assert payload["effective_time"] == "2024-05-01T00:00:00Z"
    assert response["leverage"] == Decimal("3")
    metrics_payload = event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "cfm_margin_setting"
    assert metrics_payload["margin_window"] == "INTRADAY_STANDARD"
