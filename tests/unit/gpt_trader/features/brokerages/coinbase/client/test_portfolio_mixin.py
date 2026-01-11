from __future__ import annotations

import json

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError


def _make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("list_portfolios", ()),
        ("get_portfolio", ("pf-1",)),
        ("get_portfolio_breakdown", ("pf-1",)),
        ("move_funds", ({"amount": "1"},)),
        ("list_payment_methods", ()),
        ("get_payment_method", ("pm-1",)),
        ("commit_convert_trade", ("trade-1",)),
        ("list_cfm_positions", ()),
        ("get_cfm_position", ("BTC-USD",)),
        ("list_positions_raw", ()),
    ],
)
def test_exchange_mode_blocks_portfolio_methods(method_name: str, args: tuple) -> None:
    client = _make_client(api_mode="exchange")

    with pytest.raises(InvalidRequestError):
        getattr(client, method_name)(*args)


def test_commit_convert_trade_defaults_to_empty_payload() -> None:
    client = _make_client()
    captured: dict[str, object] = {}

    def transport(method, url, headers, body, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["body"] = json.loads(body or b"{}")
        return 200, {}, json.dumps({"trade": {"id": "trade-1"}})

    client.set_transport_for_testing(transport)
    _ = client.commit_convert_trade("trade-1")

    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/api/v3/brokerage/convert/trade/trade-1")
    assert captured["body"] == {}
