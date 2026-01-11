from __future__ import annotations

import pytest

from gpt_trader.features.brokerages.coinbase.client.accounts import AccountClientMixin
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError


class _StubAccountClient(AccountClientMixin):
    def __init__(self, api_mode: str = "advanced") -> None:
        self.api_mode = api_mode
        self.calls: list[tuple[str, str]] = []

    def _get_endpoint_path(self, endpoint_name: str, **kwargs: str) -> str:
        if endpoint_name == "account":
            account_id = kwargs.get("account_uuid") or kwargs.get("account_id")
            return f"/accounts/{account_id}"
        return f"/{endpoint_name}"

    def _request(self, method: str, path: str) -> dict:
        self.calls.append((method, path))
        return {"ok": True}


def test_get_accounts_calls_endpoint() -> None:
    client = _StubAccountClient()

    result = client.get_accounts()

    assert result == {"ok": True}
    assert client.calls == [("GET", "/accounts")]


def test_get_account_uses_account_uuid() -> None:
    client = _StubAccountClient()

    client.get_account("acc-1")

    assert client.calls == [("GET", "/accounts/acc-1")]


def test_get_time_calls_endpoint() -> None:
    client = _StubAccountClient()

    client.get_time()

    assert client.calls == [("GET", "/time")]


def test_get_key_permissions_calls_endpoint() -> None:
    client = _StubAccountClient()

    client.get_key_permissions()

    assert client.calls == [("GET", "/key_permissions")]


def test_get_fees_calls_endpoint() -> None:
    client = _StubAccountClient()

    client.get_fees()

    assert client.calls == [("GET", "/fees")]


def test_get_limits_calls_endpoint() -> None:
    client = _StubAccountClient()

    client.get_limits()

    assert client.calls == [("GET", "/limits")]


def test_get_transaction_summary_advanced() -> None:
    client = _StubAccountClient(api_mode="advanced")

    client.get_transaction_summary()

    assert client.calls == [("GET", "/transaction_summary")]


def test_get_transaction_summary_exchange_raises() -> None:
    client = _StubAccountClient(api_mode="exchange")

    with pytest.raises(InvalidRequestError):
        client.get_transaction_summary()
