"""Edge coverage for Coinbase REST PortfolioService unified helpers."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.core import Balance
from gpt_trader.core.account import CFMBalance
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.persistence.event_store import EventStore


def _make_service() -> tuple[PortfolioService, Mock, Mock]:
    client = Mock()
    endpoints = Mock(spec=CoinbaseEndpoints)
    event_store = Mock(spec=EventStore)
    service = PortfolioService(client=client, endpoints=endpoints, event_store=event_store)
    return service, client, endpoints


def test_list_cfm_positions_invalid_expiry_sets_none() -> None:
    service, client, endpoints = _make_service()
    endpoints.supports_derivatives.return_value = True
    client.cfm_positions.return_value = {
        "positions": [
            {
                "product_id": "BTC-2025",
                "number_of_contracts": "1",
                "avg_entry_price": "100",
                "current_price": "110",
                "unrealized_pnl": "5",
                "daily_realized_pnl": "1",
                "side": "LONG",
                "expiration_time": "not-a-date",
            }
        ]
    }

    positions = service.list_cfm_positions()

    assert len(positions) == 1
    assert positions[0].contract_expiry is None


def test_list_spot_positions_skips_usd_and_zero() -> None:
    service, _client, _endpoints = _make_service()
    service.list_balances = Mock(
        return_value=[
            Balance(asset="USD", total=Decimal("100"), available=Decimal("80")),
            Balance(asset="BTC", total=Decimal("0"), available=Decimal("0")),
            Balance(asset="ETH", total=Decimal("2"), available=Decimal("2")),
        ]
    )

    positions = service.list_spot_positions_as_core()

    assert len(positions) == 1
    assert positions[0].symbol == "ETH-USD"
    assert positions[0].quantity == Decimal("2")


def test_get_unified_balance_combines_spot_and_cfm() -> None:
    service, _client, _endpoints = _make_service()
    service.list_balances = Mock(
        return_value=[
            Balance(asset="USD", total=Decimal("100"), available=Decimal("75")),
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]
    )
    service.get_cfm_balance = Mock(
        return_value=CFMBalance(
            futures_buying_power=Decimal("300"),
            total_usd_balance=Decimal("200"),
            available_margin=Decimal("50"),
            initial_margin=Decimal("25"),
            unrealized_pnl=Decimal("10"),
            daily_realized_pnl=Decimal("5"),
            liquidation_threshold=Decimal("150"),
            liquidation_buffer_amount=Decimal("20"),
            liquidation_buffer_percentage=60.0,
        )
    )

    balance = service.get_unified_balance()

    assert balance.spot_balance == Decimal("75")
    assert balance.cfm_balance == Decimal("200")
    assert balance.cfm_available_margin == Decimal("50")
    assert balance.cfm_buying_power == Decimal("300")
    assert balance.total_equity == Decimal("275")


def test_has_cfm_access_false_without_summary() -> None:
    service, client, endpoints = _make_service()
    endpoints.supports_derivatives.return_value = True
    client.cfm_balance_summary.return_value = {"status": "ok"}

    assert service.has_cfm_access() is False
