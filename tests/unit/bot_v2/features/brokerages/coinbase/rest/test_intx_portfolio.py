"""Tests for INTX helpers in the Coinbase portfolio REST mixin."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin
from bot_v2.features.brokerages.core.interfaces import InvalidRequestError


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
    return DummyPortfolioService(client, endpoints, event_store), client, endpoints, event_store


def test_intx_allocate_normalises_and_emits_metric(service):
    svc, client, endpoints, event_store = service
    endpoints.supports_intx.return_value = True
    client.intx_allocate.return_value = {"status": "ok", "allocated_amount": "10.5"}

    response = svc.intx_allocate({"allocated_amount": "10.5"})

    assert response["allocated_amount"] == Decimal("10.5")
    event_store.append_metric.assert_called_once()
    metrics_payload = event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "intx_allocate"
    assert metrics_payload["response"]["allocated_amount"] == "10.5"


def test_intx_allocate_requires_support(service):
    svc, _, endpoints, _ = service
    endpoints.supports_intx.return_value = False

    with pytest.raises(InvalidRequestError):
        svc.intx_allocate({"allocated_amount": "5"})


def test_get_intx_balances_normalises_entries(service):
    svc, client, endpoints, event_store = service
    endpoints.supports_intx.return_value = True
    client.intx_balances.return_value = {
        "balances": [
            {"asset": "USD", "available": "100.5"},
            {"asset": "BTC", "available": "0.25"},
        ]
    }

    balances = svc.get_intx_balances("pf-1")

    assert balances[0]["available"] == Decimal("100.5")
    assert balances[1]["available"] == Decimal("0.25")
    assert event_store.append_metric.called


def test_get_intx_balances_handles_errors(service):
    svc, client, endpoints, event_store = service
    endpoints.supports_intx.return_value = True
    client.intx_balances.side_effect = RuntimeError("boom")

    balances = svc.get_intx_balances("pf-1")

    assert balances == []
    event_store.append_metric.assert_not_called()


def test_get_intx_portfolio_returns_normalised_dict(service):
    svc, client, endpoints, _ = service
    endpoints.supports_intx.return_value = True
    client.intx_portfolio.return_value = {"portfolio": {"nav": "2500.75"}}

    portfolio = svc.get_intx_portfolio("pf-1")

    assert portfolio["nav"] == Decimal("2500.75")


def test_list_intx_positions_returns_normalised_list(service):
    svc, client, endpoints, _ = service
    endpoints.supports_intx.return_value = True
    client.intx_positions.return_value = {"positions": [{"symbol": "BTC-USD", "quantity": "1.5"}]}

    positions = svc.list_intx_positions("pf-1")

    assert positions[0]["quantity"] == Decimal("1.5")


def test_get_intx_position_handles_missing(service):
    svc, client, endpoints, _ = service
    endpoints.supports_intx.return_value = True
    client.intx_position.side_effect = RuntimeError("no position")

    position = svc.get_intx_position("pf-1", "BTC-USD")

    assert position == {}


def test_get_intx_multi_asset_collateral_emits_metric(service):
    svc, client, endpoints, event_store = service
    endpoints.supports_intx.return_value = True
    client.intx_multi_asset_collateral.return_value = {"collateral_value": "5000.25"}

    collateral = svc.get_intx_multi_asset_collateral()

    assert collateral["collateral_value"] == Decimal("5000.25")
    assert event_store.append_metric.called
