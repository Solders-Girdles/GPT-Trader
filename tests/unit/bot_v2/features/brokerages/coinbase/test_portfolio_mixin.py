from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from bot_v2.features.brokerages.coinbase.rest import portfolio
from bot_v2.features.brokerages.core.interfaces import Balance, Position


class DummyEndpoints:
    def __init__(self, derivatives: bool = True) -> None:
        self._derivatives = derivatives

    def supports_derivatives(self) -> bool:
        return self._derivatives

    def get_position(self, symbol: str) -> str:
        return f"/position/{symbol}"


class DummyClient:
    def __init__(self) -> None:
        self.accounts_response: dict[str, Any] = {}
        self.portfolio_breakdown: dict[str, Any] = {}
        self.positions_response: dict[str, Any] = {}
        self.payment_methods: dict[str, Any] = {"payment_methods": [{"id": "pm-1"}]}
        self.generic_responses: dict[str, Any] = {}
        self.position_payload: dict[str, Any] = {}
        self.limits = {"limits": {"daily": 1000}}

    # Account and portfolio endpoints
    def get_accounts(self) -> dict[str, Any]:
        return self.accounts_response

    def get_portfolio_breakdown(self, portfolio_id: str) -> dict[str, Any]:
        return self.portfolio_breakdown

    def list_positions(self) -> dict[str, Any]:
        return self.positions_response

    def get(self, path: str) -> dict[str, Any]:  # noqa: ARG002
        return self.position_payload

    def get_key_permissions(self) -> dict[str, Any]:
        return {"key_permissions": {"view": True}}

    def get_fees(self) -> dict[str, Any]:
        return {"maker": 0.001}

    def get_limits(self) -> dict[str, Any]:
        return self.limits

    def get_transaction_summary(self) -> dict[str, Any]:
        return {"summary": True}

    def list_payment_methods(self) -> dict[str, Any]:
        return self.payment_methods

    def get_payment_method(self, payment_method_id: str) -> dict[str, Any]:  # noqa: ARG002
        return {"payment_method": {"id": "pm-1"}}

    def list_portfolios(self) -> dict[str, Any]:
        return {"portfolios": [{"id": "p-1"}]}

    def get_portfolio(self, portfolio_uuid: str) -> dict[str, Any]:  # noqa: ARG002
        return {"portfolio": {"id": portfolio_uuid}}

    def move_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"moved": payload}

    def convert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"quote": payload}

    def commit_convert_trade(self, trade_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"trade": trade_id, "payload": payload}

    def get_convert_trade(self, trade_id: str) -> dict[str, Any]:
        return {"id": trade_id}


class DummyPortfolioService(portfolio.PortfolioRestMixin):
    def __init__(self, derivatives: bool = True) -> None:
        self.client = DummyClient()
        self.endpoints = DummyEndpoints(derivatives)

    def _map_position(self, data: dict[str, Any]) -> Position | None:  # type: ignore[override]
        return super()._map_position(data)


@pytest.fixture
def service() -> DummyPortfolioService:
    svc = DummyPortfolioService()
    svc.client.accounts_response = {
        "accounts": [
            {
                "currency": "BTC",
                "available_balance": {"value": "1.5"},
                "hold": {"value": "0.25"},
                "balance": {"value": "1.75"},
                "retail_portfolio_id": "portfolio-1",
            }
        ]
    }
    svc.client.portfolio_breakdown = {
        "breakdown": {
            "spot_positions": [
                {
                    "asset": "BTC",
                    "total_balance_crypto": {"value": "2"},
                    "hold": {"value": "0.5"},
                }
            ]
        }
    }
    svc.client.positions_response = {
        "positions": [
            {
                "product_id": "BTC-USD-PERP",
                "quantity": "2",
                "entry_price": "25000",
                "mark_price": "25500",
                "unrealized_pnl": "1000",
                "realized_pnl": "-100",
                "leverage": "5",
                "side": "long",
            }
        ]
    }
    svc.client.position_payload = {
        "product_id": "BTC-USD-PERP",
        "quantity": "1",
        "entry_price": "25000",
        "mark_price": "26000",
        "unrealized_pnl": "500",
        "realized_pnl": "-50",
        "leverage": "3",
        "side": "long",
    }
    return svc


def test_list_balances_parses_accounts(service: DummyPortfolioService) -> None:
    balances = service.list_balances()
    assert balances == [
        Balance(asset="BTC", total=Decimal("1.75"), available=Decimal("1.5"), hold=Decimal("0.25"))
    ]


def test_list_balances_handles_invalid_numbers(
    service: DummyPortfolioService, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    service.client.accounts_response["accounts"].append({"currency": "BAD", "balance": "nan"})
    balances = service.list_balances()
    assert any(b.asset == "BTC" for b in balances)
    assert any("Could not parse" in record.message for record in caplog.records)


def test_get_portfolio_balances_falls_back(
    service: DummyPortfolioService, monkeypatch: pytest.MonkeyPatch
) -> None:
    balances = service.get_portfolio_balances()
    assert balances[0].asset == "BTC"

    # Force failure path
    monkeypatch.setattr(
        service.client,
        "get_portfolio_breakdown",
        lambda _: (_ for _ in ()).throw(RuntimeError("down")),
    )
    balances = service.get_portfolio_balances()
    assert balances[0].asset == "BTC"


def test_list_positions_and_get_position(service: DummyPortfolioService) -> None:
    positions = service.list_positions()
    assert positions and positions[0].symbol == "BTC-USD-PERP"

    position = service.get_position("BTC-USD-PERP")
    assert position and position.symbol == "BTC-USD-PERP"


def test_list_positions_derivatives_disabled() -> None:
    service = DummyPortfolioService(derivatives=False)
    assert service.list_positions() == []
    assert service.get_position("ANY") is None


def test_portfolio_misc_wrappers(service: DummyPortfolioService) -> None:
    assert service.get_key_permissions()["view"]
    assert service.get_fee_schedule()["maker"] == 0.001
    assert service.get_account_limits()["daily"] == 1000
    assert service.get_transaction_summary()["summary"]
    assert service.list_payment_methods()[0]["id"] == "pm-1"
    assert service.get_payment_method("pm-1")["id"] == "pm-1"
    assert service.list_portfolios()[0]["id"] == "p-1"
    assert service.get_portfolio("p-1")["id"] == "p-1"
    assert service.get_portfolio_breakdown("p-1")["spot_positions"]
    assert service.move_portfolio_funds({"amount": "1"})["moved"]["amount"] == "1"
    assert service.create_convert_quote({"from": "BTC", "to": "USD"})["quote"]
    assert service.commit_convert_trade("id-1", {"commit": True})["trade"] == "id-1"
    assert service.get_convert_trade("id-1")["id"] == "id-1"
