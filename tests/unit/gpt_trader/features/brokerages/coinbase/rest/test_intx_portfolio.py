"""Tests for INTX helpers in the Coinbase portfolio REST service."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.features.brokerages.core.interfaces import InvalidRequestError


@pytest.fixture()
def service():
    client = Mock()
    endpoints = Mock()
    event_store = Mock()
    # test expects append_metric to be mocked
    event_store.append_metric = Mock()

    svc = PortfolioService(  # naming: allow
        client=client, endpoints=endpoints, event_store=event_store
    )
    return svc, client, endpoints, event_store  # naming: allow


def test_intx_allocate_normalises_and_emits_metric(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.mode = "advanced"  # Set mode for intx_allocate
    client.intx_allocate.return_value = {"status": "ok", "allocated_amount": "10.5"}

    response = svc.intx_allocate({"allocated_amount": "10.5"})  # naming: allow

    assert response["allocated_amount"] == Decimal("10.5")
    event_store.append_metric.assert_called_once()
    metrics_payload = event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "intx_allocation"
    assert metrics_payload["response"]["allocated_amount"] == Decimal("10.5")


def test_intx_allocate_requires_support(service):
    svc, _, endpoints, _ = service  # naming: allow
    endpoints.mode = "simple"  # Set mode for intx_allocate
    # test expects InvalidRequestError when not advanced
    with pytest.raises(InvalidRequestError):
        svc.intx_allocate({"allocated_amount": "5"})  # naming: allow


def test_get_intx_balances_normalises_entries(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.mode = "advanced"  # Set mode for intx balances
    client.get_intx_portfolio.return_value = {  # Renamed from intx_balances to get_intx_portfolio
        "balances": [
            {
                "asset": "USD",
                "amount": "100.5",
                "hold": "0",
            },  # Changed available to amount to match portfolio.py
            {"asset": "BTC", "amount": "0.25", "hold": "0"},
        ]
    }

    balances = svc.get_intx_balances("pf-1")  # naming: allow

    assert balances[0]["amount"] == Decimal("100.5")
    assert balances[1]["amount"] == Decimal("0.25")
    assert event_store.append_metric.called


def test_get_intx_balances_handles_errors(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.mode = "advanced"
    client.get_intx_portfolio.side_effect = RuntimeError("boom")

    balances = svc.get_intx_balances("pf-1")  # naming: allow

    assert balances == []
    event_store.append_metric.assert_not_called()


def test_get_intx_portfolio_returns_normalised_dict(service):
    svc, client, endpoints, _ = service  # naming: allow
    endpoints.mode = "advanced"
    client.get_intx_portfolio.return_value = {
        "portfolio_value": "2500.75"
    }  # Renamed nav to portfolio_value

    portfolio = svc.get_intx_portfolio("pf-1")  # naming: allow

    assert portfolio["portfolio_value"] == Decimal("2500.75")


def test_list_intx_positions_returns_normalised_list(service):
    svc, client, endpoints, _ = service  # naming: allow
    endpoints.mode = "advanced"
    client.list_intx_positions.return_value = {
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
    }  # Add required fields for to_position

    positions = svc.list_intx_positions("pf-1")  # naming: allow

    assert positions[0].quantity == Decimal("1.5")  # to_position returns Position object, not dict


def test_get_intx_position_handles_missing(service):
    svc, client, endpoints, _ = service  # naming: allow
    endpoints.mode = "advanced"
    client.get_intx_position.side_effect = RuntimeError("no position")

    position = svc.get_intx_position("pf-1", "BTC-USD")  # naming: allow

    assert position is None


def test_get_intx_multi_asset_collateral_emits_metric(service):
    svc, client, endpoints, event_store = service  # naming: allow
    endpoints.mode = "advanced"
    client.get_intx_multi_asset_collateral.return_value = {
        "total_usd_value": "5000.25"
    }  # Renamed collateral_value to total_usd_value

    collateral = svc.get_intx_multi_asset_collateral()  # naming: allow

    assert collateral["total_usd_value"] == Decimal("5000.25")
    assert event_store.append_metric.called
    metrics_payload = event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "intx_multi_asset_collateral"
    assert metrics_payload["data"]["total_usd_value"] == Decimal("5000.25")
